# SPDX-License-Identifier: Apache-2.0
"""线性层「加载 → BF16 回退 apply」全路径测试 + 设备路径门控。

复现并锁定 2026-11-16 评审发现的问题：BF16 回退分支把权重当原始 FP4 解码，
但 process_weights_after_loading 在加载时已经转置权重并重排尺度，
解码器拿到的是错误布局。回退路径现在保持原始布局；路径由
``_mxfp4_path()`` 按设备能力统一判定：A5 原生、A2+显式开关回退、其余拒绝。
"""
import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]


class Param:
    """模拟 nn.Parameter：可写 .data，同时代理 .shape 等张量属性。"""

    def __init__(self, data):
        self.data = data

    def __getattr__(self, name):
        return getattr(self.data, name)


class FakeDeviceType:
    A2 = 'A2'
    A3 = 'A3'
    _310P = '_310P'
    A5 = 'A5'


def _stub_modules(device_type: str, fallback_enabled: bool = True, cast_fails: bool = False):
    mods = {}
    dev_config = types.ModuleType('vllm_ascend.device.device_config')
    dev_config.get_ascend_device_type = lambda: device_type
    mods['vllm_ascend.device.device_config'] = dev_config
    hardware = types.ModuleType('vllm_ascend.device.hardware')
    hardware.AscendDeviceType = FakeDeviceType
    mods['vllm_ascend.device.hardware'] = hardware

    envs_mod = types.ModuleType('vllm_ascend.envs')
    envs_mod.VLLM_ASCEND_FP4_MXFP4_FALLBACK = fallback_enabled
    mods['vllm_ascend.envs'] = envs_mod

    base = types.ModuleType('vllm_ascend.quantization.methods.base')

    class AscendLinearScheme:
        pass

    class AscendMoEScheme:
        pass

    class PreparedLinearInput:
        pass

    base.AscendLinearScheme = AscendLinearScheme
    base.AscendMoEScheme = AscendMoEScheme
    base.PreparedLinearInput = PreparedLinearInput
    base.QuantType = MagicMock(W4A8MXFP='W4A8MXFP')
    base.WeightSwitchGatherSpec = MagicMock()
    mods['vllm_ascend.quantization.methods.base'] = base

    registry = types.ModuleType('vllm_ascend.quantization.methods.registry')
    registry.register_scheme = lambda *a, **k: (lambda cls: cls)
    mods['vllm_ascend.quantization.methods.registry'] = registry

    utils = types.ModuleType('vllm_ascend.utils')
    utils.ACL_FORMAT_FRACTAL_NZ = 29
    utils.FP8_METHOD = 'fp8'
    utils.dispose_tensor = lambda x: None
    mods['vllm_ascend.utils'] = utils

    for name in (
        'torch_npu', 'vllm', 'vllm.config',
        'vllm.model_executor', 'vllm.model_executor.layers',
        'vllm.model_executor.layers.fused_moe',
        'vllm.model_executor.layers.fused_moe.activation',
        'vllm_ascend.ascend_config', 'vllm_ascend.ascend_forward_context',
        'vllm_ascend.ops', 'vllm_ascend.ops.fused_moe',
        'vllm_ascend.ops.fused_moe.dataclass',
        'vllm_ascend.ops.fused_moe.dataclass.fused_experts',
        'vllm_ascend.ops.fused_moe.dataclass.moe_mlp',
        'vllm_ascend.ops.fused_moe.moe_utils',
        'vllm_ascend.ops.fused_moe.routed_experts',
        'vllm_ascend.quantization', 'vllm_ascend.quantization.methods',
        'vllm_ascend.quantization.methods.w4a8',
        'vllm_ascend.device', 'vllm_ascend.device.device_op',
        'vllm_ascend.quantization.nz',
        'vllm_ascend.quantization.methods.w4a8.fp4_decode_cache',
        'vllm_ascend.quantization.methods.w4a8.fp4_int8_cube',
    ):
        mods.setdefault(name, MagicMock(name=name))

    # NZ cast 规格：把 transpose/reshape 校验建立在真实的 format_cast 调用上。
    nz = types.ModuleType('vllm_ascend.quantization.nz')

    class _NzCastSpec:
        def __init__(self, customize_dtype=None, input_dtype=None):
            self._kwargs = {}
            if customize_dtype is not None:
                self._kwargs['customize_dtype'] = customize_dtype
            if input_dtype is not None:
                self._kwargs['input_dtype'] = input_dtype

        def cast(self, tensor):
            import torch_npu as _tnpu
            return _tnpu.npu_format_cast(tensor, 29, **self._kwargs)

        def cast_(self, tensor):
            import torch_npu as _tnpu
            _tnpu.npu_format_cast_(tensor, 29, **self._kwargs)
            return tensor

    nz.NzCastSpec = _NzCastSpec
    nz.fp4_packed_nz = lambda: _NzCastSpec(
        customize_dtype=torch.float8_e4m3fn, input_dtype=torch.float4_e2m1fn_x2)
    nz.fp4_semantic_nz = lambda: _NzCastSpec(
        customize_dtype=torch.float8_e4m3fn, input_dtype=torch.float4_e2m1fn_x2)
    mods['vllm_ascend.quantization.nz'] = nz

    # format_cast 用恒等实现，让后续的 transpose / scale reshape 走真实张量运算，
    # 从而能干净地断言回退路径是否改动了布局。
    tnpu = mods['torch_npu']
    if cast_fails:
        def _failing_cast(*args, **kwargs):
            raise RuntimeError('unsupported FP4(x2)->FP8 cast')
        tnpu.npu_format_cast = _failing_cast
    else:
        tnpu.npu_format_cast = lambda tensor, *args, **kwargs: tensor

    saved = {name: sys.modules.get(name) for name in mods}
    sys.modules.update(mods)
    return saved


def load_mxfp4(device_type: str = 'A2', fallback_enabled: bool = True, cast_fails: bool = False):
    saved = _stub_modules(device_type, fallback_enabled, cast_fails)
    for name in ('vllm_ascend', 'vllm_ascend.quantization',
                 'vllm_ascend.quantization.methods', 'vllm_ascend.quantization.methods.w4a8'):
        pkg = types.ModuleType(name)
        pkg.__path__ = []
        if name not in saved:
            saved[name] = sys.modules.get(name)
        sys.modules[name] = pkg
    sys.modules['vllm_ascend'].envs = sys.modules['vllm_ascend.envs']
    target = 'vllm_ascend.quantization.methods.w4a8.w4a8_mxfp4'
    # 记录真实模块（若已由其他测试导入），退出时一并还原，避免污染同进程后续用例。
    saved[target] = sys.modules.get(target)
    sys.modules.pop(target, None)
    spec = importlib.util.spec_from_file_location(
        target, ROOT / 'vllm_ascend/quantization/methods/w4a8/w4a8_mxfp4.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, saved


def restore_modules(saved):
    for name, mod in saved.items():
        if mod is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = mod


class TestNonA5LinearPath(unittest.TestCase):
    OUT, IN_DIM, GROUP = 8, 64, 32

    def _make_layer(self, mod):
        torch.manual_seed(0)
        w_fp4 = torch.randint(0, 256, (self.OUT, self.IN_DIM // 2), dtype=torch.uint8)
        w_scale = torch.randint(119, 128, (self.OUT, self.IN_DIM // self.GROUP), dtype=torch.uint8)
        layer = SimpleNamespace(weight=Param(w_fp4), weight_scale=Param(w_scale))
        return layer, w_fp4.clone(), w_scale.clone()

    def _make_method(self, mod):
        cls = mod.AscendW4A8MXFPDynamicLinearMethod
        inst = cls.__new__(cls)
        inst.group_size = self.GROUP
        return inst

    def test_load_then_apply_matches_original_layout_reference(self):
        mod, saved = load_mxfp4('A2', fallback_enabled=True)
        try:
            layer, w_fp4, w_scale = self._make_layer(mod)
            inst = self._make_method(mod)
            inst.process_weights_after_loading(layer)
            # BF16 回退不得改变加载时的原始布局，也不得标记已变换。
            self.assertTrue(torch.equal(layer.weight.data, w_fp4))
            self.assertTrue(torch.equal(layer.weight_scale.data, w_scale))
            self.assertFalse(getattr(layer, '_mxfp4_transformed', False))

            x = torch.randn(4, self.IN_DIM, dtype=torch.bfloat16)
            bias = torch.randn(self.OUT, dtype=torch.bfloat16)
            out = inst.apply(layer, x, bias)

            w_ref = mod.dequant_fp4_to_bf16(w_fp4, w_scale, self.GROUP)
            ref = F.linear(x, w_ref, bias)
            self.assertEqual(out.shape, ref.shape)
            self.assertTrue(torch.equal(out, ref))

            # restore 在回退路径上必须是空操作（未做过变换）。
            before_w, before_s = layer.weight.data.clone(), layer.weight_scale.data.clone()
            inst.restore_weights_for_rl_loading(layer)
            self.assertTrue(torch.equal(layer.weight.data, before_w))
            self.assertTrue(torch.equal(layer.weight_scale.data, before_s))
        finally:
            restore_modules(saved)

    def _assert_path_rejected(self, device_type, fallback_enabled):
        mod, saved = load_mxfp4(device_type, fallback_enabled=fallback_enabled)
        try:
            layer, _, _ = self._make_layer(mod)
            inst = self._make_method(mod)
            x = torch.randn(4, self.IN_DIM, dtype=torch.bfloat16)
            # apply 与 process_weights_after_loading 都必须显式拒绝。
            with self.assertRaisesRegex(RuntimeError, 'not enabled on device_type'):
                inst.apply(layer, x, None)
            with self.assertRaisesRegex(RuntimeError, 'not enabled on device_type'):
                inst.process_weights_after_loading(layer)
        finally:
            restore_modules(saved)

    def test_a3_910c_is_rejected_even_with_switch(self):
        self._assert_path_rejected('A3', True)

    def test_310p_is_rejected_even_with_switch(self):
        self._assert_path_rejected('_310P', True)

    def test_a2_without_explicit_switch_is_rejected(self):
        self._assert_path_rejected('A2', False)

    def test_a5_keeps_native_transform_and_restore(self):
        mod, saved = load_mxfp4('A5')
        try:
            layer, w_fp4, w_scale = self._make_layer(mod)
            inst = self._make_method(mod)
            inst.process_weights_after_loading(layer)
            self.assertTrue(getattr(layer, '_mxfp4_transformed', False))
            # A5 必须做转置与 scale 重排（原生量化矩阵乘要求）。
            self.assertTrue(torch.equal(layer.weight.data, w_fp4.transpose(-1, -2)))
            self.assertEqual(layer.weight_scale.data.shape, (1, self.OUT, 2))

            # restore 必须完整还原，供 RL 权重重载。
            inst.restore_weights_for_rl_loading(layer)
            self.assertTrue(torch.equal(layer.weight.data, w_fp4))
            self.assertTrue(torch.equal(layer.weight_scale.data, w_scale))
            self.assertFalse(getattr(layer, '_mxfp4_transformed', False))
        finally:
            restore_modules(saved)

    def test_a5_format_cast_failure_is_explicit(self):
        mod, saved = load_mxfp4('A5', cast_fails=True)
        try:
            layer, w_fp4, w_scale = self._make_layer(mod)
            inst = self._make_method(mod)
            # 合入 382dcb27e 后 A5 走 nz.fp4_packed_nz().cast()，转换失败由
            # npu_format_cast 直接抛出（不再包一层 "half-transformed"）；关键行为
            # 仍是"显式失败、且不留下半变换布局"。
            with self.assertRaisesRegex(RuntimeError, 'unsupported FP4'):
                inst.process_weights_after_loading(layer)
            # 失败后不得留下半变换布局：权重未被转置、标记未置位。
            self.assertTrue(torch.equal(layer.weight.data, w_fp4))
            self.assertFalse(getattr(layer, '_mxfp4_transformed', False))
        finally:
            restore_modules(saved)

    def test_apply_detects_mutated_scale(self):
        mod, saved = load_mxfp4('A2', fallback_enabled=True)
        try:
            layer, w_fp4, w_scale = self._make_layer(mod)
            inst = self._make_method(mod)
            inst.process_weights_after_loading(layer)
            x = torch.randn(4, self.IN_DIM, dtype=torch.bfloat16)
            out = inst.apply(layer, x, None)
            bad = SimpleNamespace(weight=Param(w_fp4), weight_scale=Param(w_scale.clone()))
            bad.weight_scale.data[0, 0] = (bad.weight_scale.data[0, 0].to(torch.int32) + 4).to(torch.uint8)
            out_bad = inst.apply(bad, x, None)
            self.assertFalse(torch.equal(out, out_bad))
        finally:
            restore_modules(saved)


if __name__ == '__main__':
    unittest.main()
