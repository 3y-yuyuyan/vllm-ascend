#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


from typing import Any

import torch
import torch.nn.functional as F
import torch_npu
from vllm.config import get_current_vllm_config
from vllm.model_executor.layers.fused_moe.activation import MoEActivation

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import _EXTRA_CTX, use_cann_megamoe
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.ops.fused_moe.dataclass.fused_experts import MoEWeights, build_fused_experts_input
from vllm_ascend.ops.fused_moe.dataclass.moe_mlp import MoEMlpComputeInput
from vllm_ascend.ops.fused_moe.moe_utils import cumsum_group_list, maybe_normalize_mxfp_scale_layout
from vllm_ascend.ops.fused_moe.routed_experts import AscendRoutedExperts  # noqa: F401
from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ, FP8_METHOD, dispose_tensor

from ..base import (
    AscendLinearScheme,
    AscendMoEScheme,
    PreparedLinearInput,
    QuantType,
    WeightSwitchGatherSpec,
)
from ..registry import register_scheme
from .fp4_decode_cache import clear_fp4_decode_cache, decode_active_fp4_weights, initialize_fp4_decode_cache
from .fp4_int8_cube import apply_fp4_int8_gmm, initialize_fp4_int8_layer, quantize_fp4_int8_activation

# CANN uses 36 to select FP8 E4M3FN output for situ_mx_quant.
SITU_MX_DST_TYPE_E4M3FN = 36

# W4A8 MXFP4 执行路径：按"已验证的设备能力"选择，而不是宽泛的"非 A5"。
_MXFP4_PATH_NATIVE = "native"
_MXFP4_PATH_FALLBACK = "bf16_fallback"


def _mxfp4_path() -> str:
    """Return the W4A8 MXFP4 execution path for the current device/config.

    - A5: native FP4 quant matmul (needs NZ/FP8 weight layout).
    - A2 (910B family) with ``VLLM_ASCEND_FP4_MXFP4_FALLBACK=1``: BF16
      fallback that decodes packed FP4 weights at call time. Operator-level
      validation exists only on Ascend 910B4-1; enabling it on other A2
      variants is the operator's explicit responsibility.
    - A3 (910C), _310P, or A2 without the switch: raise, never silently run
      an unverified path.

    Kept as a single function so all call sites stay consistent.
    """
    from vllm_ascend import envs
    from vllm_ascend.device.device_config import get_ascend_device_type
    from vllm_ascend.device.hardware import AscendDeviceType

    device_type = get_ascend_device_type()
    if device_type == AscendDeviceType.A5:
        return _MXFP4_PATH_NATIVE
    if device_type in (AscendDeviceType.A2, AscendDeviceType.A3) and envs.VLLM_ASCEND_FP4_MXFP4_FALLBACK:
        return _MXFP4_PATH_FALLBACK
    raise RuntimeError(
        f"W4A8 MXFP4 is not enabled on device_type={device_type}. "
        "A5 uses the native FP4 quant matmul; the BF16 fallback is validated "
        "on Ascend 910B (A2) and 910_93 (A3) and must be enabled explicitly with "
        "VLLM_ASCEND_FP4_MXFP4_FALLBACK=1. Refusing to run an unverified path."
    )

# Device-resident LUT cache to eliminate Host-to-Device memory copy during CUDA Graph capture
_LUT_MAP: dict[str, torch.Tensor] = {}


def get_lut(device: torch.device) -> torch.Tensor:
    dev_key = str(device)
    lut = _LUT_MAP.get(dev_key)
    if lut is None or lut.device != device:
        lut = torch.tensor(
            [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
            dtype=torch.bfloat16,
            device=device,
        )
        _LUT_MAP[dev_key] = lut
    return lut


def dequant_fp4_to_bf16(
    w_fp4: torch.Tensor,
    w_scale: torch.Tensor,
    block_size: int = 32,
    chunk_size: int = 2,
    transpose: bool = False,
) -> torch.Tensor:
    """Dynamically decode packed MXFP4 weights and E8M0 scale to bfloat16 with minimal memory footprint and zero host-device sync."""
    lut = get_lut(w_fp4.device)
    ndim = w_fp4.ndim

    if ndim == 3:
        num_experts = w_fp4.shape[0]
        dim1 = w_fp4.shape[1]
        dim2 = w_fp4.shape[2] * 2
        if transpose:
            out = torch.empty((num_experts, dim2, dim1), dtype=torch.bfloat16, device=w_fp4.device)
        else:
            out = torch.empty((num_experts, dim1, dim2), dtype=torch.bfloat16, device=w_fp4.device)

        for i in range(0, num_experts, chunk_size):
            end = min(i + chunk_size, num_experts)
            c_fp4 = w_fp4[i:end]
            c_scale = w_scale[i:end]

            c_shape = list(c_fp4.shape)
            c_shape[-1] *= 2

            low = c_fp4 & 0x0F
            high = (c_fp4 >> 4) & 0x0F
            interleaved = torch.stack([low, high], dim=-1).reshape(c_shape)
            decoded = lut[interleaved.to(torch.int32)]

            scale_raw = c_scale.view(torch.uint8).to(torch.int32)
            powers = torch.pow(2.0, (scale_raw - 127).to(torch.float32)).to(torch.bfloat16)
            scale_float = torch.where(scale_raw == 0, torch.zeros_like(powers), powers)

            if scale_float.shape[-1] * block_size == decoded.shape[-1]:
                chunk_res = (decoded.view(*scale_float.shape, block_size) * scale_float.unsqueeze(-1)).view(c_shape)
            else:
                scale_expanded = scale_float.repeat_interleave(block_size, dim=-1)
                if scale_expanded.shape[-1] > decoded.shape[-1]:
                    scale_expanded = scale_expanded[..., : decoded.shape[-1]]
                chunk_res = decoded * scale_expanded

            if transpose:
                out[i:end].copy_(chunk_res.transpose(1, 2))
            else:
                out[i:end].copy_(chunk_res)
        return out

    elif ndim == 2:
        rows = w_fp4.shape[0]
        cols = w_fp4.shape[1] * 2
        chunk_rows = max(1, min(128, rows))
        out = torch.empty((rows, cols), dtype=torch.bfloat16, device=w_fp4.device)
        for i in range(0, rows, chunk_rows):
            end = min(i + chunk_rows, rows)
            c_fp4 = w_fp4[i:end]
            c_scale = w_scale[i:end]

            c_shape = list(c_fp4.shape)
            c_shape[-1] *= 2

            low = c_fp4 & 0x0F
            high = (c_fp4 >> 4) & 0x0F
            interleaved = torch.stack([low, high], dim=-1).reshape(c_shape)
            decoded = lut[interleaved.to(torch.int32)]

            scale_raw = c_scale.view(torch.uint8).to(torch.int32)
            powers = torch.pow(2.0, (scale_raw - 127).to(torch.float32)).to(torch.bfloat16)
            scale_float = torch.where(scale_raw == 0, torch.zeros_like(powers), powers)

            if scale_float.shape[-1] * block_size == decoded.shape[-1]:
                chunk_res = (decoded.view(*scale_float.shape, block_size) * scale_float.unsqueeze(-1)).view(c_shape)
            else:
                scale_expanded = scale_float.repeat_interleave(block_size, dim=-1)
                if scale_expanded.shape[-1] > decoded.shape[-1]:
                    scale_expanded = scale_expanded[..., : decoded.shape[-1]]
                chunk_res = decoded * scale_expanded

            out[i:end].copy_(chunk_res)
        return out
    else:
        orig_shape = list(w_fp4.shape)
        orig_shape[-1] *= 2
        low = w_fp4 & 0x0F
        high = (w_fp4 >> 4) & 0x0F
        interleaved = torch.stack([low, high], dim=-1).reshape(orig_shape)
        decoded = lut[interleaved.to(torch.int32)]
        scale_raw = w_scale.view(torch.uint8).to(torch.int32)
        powers = torch.pow(2.0, (scale_raw - 127).to(torch.float32)).to(torch.bfloat16)
        scale_float = torch.where(scale_raw == 0, torch.zeros_like(powers), powers)
        if scale_float.shape[-1] * block_size == decoded.shape[-1]:
            return (decoded.view(*scale_float.shape, block_size) * scale_float.unsqueeze(-1)).view(orig_shape)
        scale_expanded = scale_float.repeat_interleave(block_size, dim=-1)
        if scale_expanded.shape[-1] > decoded.shape[-1]:
            scale_expanded = scale_expanded[..., : decoded.shape[-1]]
        return decoded * scale_expanded


@register_scheme("W4A8_MXFP", "linear")
class AscendW4A8MXFPDynamicLinearMethod(AscendLinearScheme):
    """Linear method for Ascend W4A8_MXFP (Microscaling) quantization."""

    weight_switch_gather_specs = (
        WeightSwitchGatherSpec("weight"),
        WeightSwitchGatherSpec("weight_scale"),
    )
    weight_switch_output_gather_specs = (
        WeightSwitchGatherSpec("weight", gather_dim=1),
        WeightSwitchGatherSpec("weight_scale", gather_dim=1),
    )
    supports_weight_switch = True

    def __init__(self):
        vllm_config = get_current_vllm_config()
        self.group_size = vllm_config.quant_config.quant_description.get("group_size", 32)

    @staticmethod
    def get_weight(input_size: int, output_size: int, params_dtype: torch.dtype) -> dict[str, Any]:
        params_dict = {"weight": torch.empty(output_size, input_size // 2, dtype=torch.uint8)}
        return params_dict

    def get_pergroup_param(
        self, input_size: int, output_size: int, params_dtype: torch.dtype, layer_type: str | None = None
    ) -> dict[str, Any]:
        params_dict = {}
        params_dict["weight_scale"] = torch.empty(output_size, input_size // self.group_size, dtype=torch.uint8)
        return params_dict

    def prepare_input_for_overlap(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
    ) -> PreparedLinearInput | None:
        if x.dim() != 2:
            return None
        quantized_x, dynamic_scale = torch_npu.npu_dynamic_mx_quant(x, dst_type=torch.float8_e4m3fn)
        return PreparedLinearInput(quantized_x, dynamic_scale, x.dtype)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor | PreparedLinearInput | tuple[torch.Tensor, torch.Tensor],
        bias: torch.Tensor | None = None,
        tp_rank: int | None = 0,
    ) -> torch.Tensor:
        # BF16 回退（仅 A2 且显式开启 VLLM_ASCEND_FP4_MXFP4_FALLBACK）：
        # 权重保持加载时的原始 FP4 布局，在此按原始布局现场解码后做线性运算。
        # A5 走原生 FP4 量化矩阵乘（需要 NZ/FP8 转换后的布局）；
        # 其余设备由 _mxfp4_path() 显式拒绝。
        if _mxfp4_path() == _MXFP4_PATH_FALLBACK:
            if isinstance(x, tuple):
                x = x[0].to(torch.bfloat16)
            else:
                x = x.to(torch.bfloat16)
            w = layer.weight.data
            if w.dtype == torch.uint8:
                w = dequant_fp4_to_bf16(w, layer.weight_scale.data, self.group_size)
            elif w.dtype != torch.bfloat16:
                w = w.to(torch.bfloat16)
            if w.ndim == 2:
                if w.shape[0] == x.shape[-1]:
                    out = torch.matmul(x, w)
                    if bias is not None:
                        out = out + bias
                    return out
                elif w.shape[1] == x.shape[-1]:
                    return F.linear(x, w, bias)
            return F.linear(x, w, bias)

        if isinstance(x, PreparedLinearInput):
            quantized_x = x.quantized
            assert x.scale is not None
            dynamic_scale = x.scale
            output_dtype = x.output_dtype
        elif isinstance(x, tuple):
            quantized_x, dynamic_scale = x
            output_dtype = torch.bfloat16
        else:
            quantized_x, dynamic_scale = torch_npu.npu_dynamic_mx_quant(x, dst_type=torch.float8_e4m3fn)
            output_dtype = x.dtype

        output = torch_npu.npu_quant_matmul(
            quantized_x,
            layer.weight,
            layer.weight_scale,
            scale_dtype=torch_npu.float8_e8m0fnu,
            pertoken_scale=dynamic_scale,
            pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
            bias=bias,
            output_dtype=output_dtype,
            x2_dtype=torch_npu.float4_e2m1fn_x2,
            group_sizes=[0, 0, self.group_size],
        )

        return output

    def process_weights_after_loading(self, layer):
        """Cast the weight to NZ format and reshape its scale for NPU inference (A5 only).

        The BF16 fallback (see ``_mxfp4_path``) runs in ``apply``, which decodes
        the original FP4 layout at call time.  The weight and scale must therefore
        keep their loader-provided ``[out, in/2]`` / ``[out, in/group_size]``
        layout; transposing/reshaping them here would feed the decoder a
        layout it cannot interpret (dimension error or silent wrong results).
        """
        if getattr(layer, "_mxfp4_transformed", False):
            return
        if _mxfp4_path() == _MXFP4_PATH_FALLBACK:
            # 保持原始布局；不置位 _mxfp4_transformed，
            # restore_weights_for_rl_loading 自然成为空操作。
            return
        if not hasattr(layer, "_mxfp4_original_shapes"):
            layer._mxfp4_original_shapes = {
                "weight": tuple(layer.weight.data.shape),
                "weight_scale": tuple(layer.weight_scale.data.shape),
            }
        try:
            layer.weight.data = torch_npu.npu_format_cast(
                layer.weight.data,
                ACL_FORMAT_FRACTAL_NZ,
                customize_dtype=torch.float8_e4m3fn,
                input_dtype=torch_npu.float4_e2m1fn_x2,
            )
        except Exception as err:
            # 转换失败后继续 transpose/reshape 会留下"半变换"的权重布局，
            # A5 的原生量化矩阵乘会读到错误数据；必须显式失败。
            raise RuntimeError(
                "npu_format_cast failed for MXFP4 weight (A5 requires the "
                "FP4(x2)->FP8 NZ cast); aborting instead of continuing with a "
                "half-transformed layout. Original error: " + str(err)
            ) from err
        layer.weight.data = layer.weight.data.transpose(-1, -2)
        n, k = layer.weight_scale.shape
        layer.weight_scale.data = layer.weight_scale.data.reshape(n, k // 2, 2).transpose(-3, -2)
        layer._mxfp4_transformed = True

    def restore_weights_for_rl_loading(self, layer):
        """Undo the NZ/scale transform so the weight loader can reload ND weights.

        Reverses the transpose, casts the weight back to ND (format 2), and
        restores the scale's original shape.
        """
        if not getattr(layer, "_mxfp4_transformed", False):
            return
        layer.weight.data = layer.weight.data.transpose(-1, -2)
        layer.weight.data = torch_npu.npu_format_cast(layer.weight.data, 2)
        orig_scale_shape = layer._mxfp4_original_shapes["weight_scale"]
        layer.weight_scale.data = layer.weight_scale.data.transpose(-3, -2).reshape(orig_scale_shape)
        layer._mxfp4_transformed = False


@register_scheme("W4A8_MXFP", "moe")
class AscendW4A8MXFPDynamicFusedMoEMethod(AscendMoEScheme):
    """FusedMoe method for Ascend W4A8_DYNAMIC."""

    supports_eplb = True
    quant_type: QuantType = QuantType.W4A8MXFP
    act_quant_type: torch.dtype = torch.float8_e4m3fn
    fused_activations = frozenset({"silu", "situ"})

    def __init__(self):
        vllm_config = get_current_vllm_config()
        self.group_size = vllm_config.quant_config.quant_description.get("group_size", 32)
        ascend_config = get_ascend_config()
        self.dynamic_eplb = False if vllm_config.use_v2_model_runner else ascend_config.eplb_config.dynamic_eplb
        self.use_expert_weight_list = self.dynamic_eplb or (
            vllm_config.use_v2_model_runner is True and vllm_config.parallel_config.enable_eplb is True
        )

    @staticmethod
    def get_weight(
        num_experts: int, intermediate_size_per_partition: int, hidden_sizes: int, params_dtype: torch.dtype
    ) -> dict[str, Any]:
        param_dict = {}
        param_dict["w13_weight"] = torch.empty(
            num_experts, 2 * intermediate_size_per_partition, hidden_sizes // 2, dtype=torch.uint8
        )
        param_dict["w2_weight"] = torch.empty(
            num_experts, hidden_sizes, intermediate_size_per_partition // 2, dtype=torch.uint8
        )
        return param_dict

    def get_dynamic_quant_param(
        self, num_experts: int, intermediate_size_per_partition: int, hidden_sizes: int, params_dtype: torch.dtype
    ) -> dict[str, Any]:
        param_dict = {}
        param_dict["w13_weight_scale"] = torch.empty(
            num_experts, 2 * intermediate_size_per_partition, hidden_sizes // self.group_size, dtype=torch.uint8
        )

        param_dict["w2_weight_scale"] = torch.empty(
            num_experts, hidden_sizes, intermediate_size_per_partition // self.group_size, dtype=torch.uint8
        )
        return param_dict

    def apply(
        self,
        layer: "AscendRoutedExperts",
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: Any | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        # BF16 回退时用 None 关闭原生 MXFP4 量化参数。
        # 路径由 _mxfp4_path() 统一判定；原生 FP4 INT8 Cube 另有 pure-TP 限制，
        # 见 fp4_int8_cube.initialize_fp4_int8_layer 的 EP 校验。
        use_bf16_fallback = _mxfp4_path() == _MXFP4_PATH_FALLBACK
        moe_comm_method = _EXTRA_CTX.moe_comm_method
        return moe_comm_method.fused_experts(
            fused_experts_input=build_fused_experts_input(
                hidden_states=x,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                layer=layer,
                quant_type=self.quant_type,
                dynamic_eplb=self.use_expert_weight_list,
                expert_map=layer.ascend_expert_map,
                global_redundant_expert_num=layer.global_redundant_expert_num,
                mc2_mask=layer.ascend_mc2_mask,
                apply_router_weight_on_input=layer.apply_router_weight_on_input,
                pertoken_scale=layer.ascend_pertoken_scale,
                activation=layer.activation,
                mxfp_act_quant_type=None if use_bf16_fallback else torch.float8_e4m3fn,
                mxfp_weight_quant_type=None if use_bf16_fallback else torch_npu.float4_e2m1fn_x2,
                mxfp_scale_dtype=None if use_bf16_fallback else torch_npu.float8_e8m0fnu,
                mxfp_per_token_scale_dtype=None if use_bf16_fallback else torch_npu.float8_e8m0fnu,
                mxfp_use_bf16=True if use_bf16_fallback else (x.dtype in [torch.bfloat16, torch.float8_e4m3fn]),
            ),
            quant_method=self,
        )

    def get_fused_mc2_weights(self, layer: torch.nn.Module) -> MoEWeights:
        if self.use_expert_weight_list:
            w1 = layer.w13_weight_list
            w2 = layer.w2_weight_list
            w1_scale = layer.w13_weight_scale_list
            w2_scale = layer.w2_weight_scale_list
            return MoEWeights(
                w1=w1,
                w2=w2,
                w1_scale=w1_scale,
                w2_scale=w2_scale,
                w1_scale_bias=None,
                w2_scale_bias=None,
            )
        if _EXTRA_CTX.use_mega_moe:
            # MegaMoe consumes the non-transposed per-expert weight/scale lists
            # built in process_weights_after_loading (the non-mega path uses the
            # transposed single tensors below).
            return MoEWeights(
                w1=layer.cann_mega_moe_w13_weight_list,
                w2=layer.cann_mega_moe_w2_weight_list,
                w1_scale=layer.cann_mega_moe_w13_weight_scale_list,
                w2_scale=layer.cann_mega_moe_w2_weight_scale_list,
                w1_scale_bias=None,
                w2_scale_bias=None,
            )
        else:
            return MoEWeights(
                w1=[layer.w13_weight],
                w2=[layer.w2_weight],
                w1_scale=[layer.w13_weight_scale],
                w2_scale=[layer.w2_weight_scale],
                w1_scale_bias=None,
                w2_scale_bias=None,
            )

    @staticmethod
    def get_eplb_weight_views(layer: torch.nn.Module) -> list:
        if hasattr(layer, "w13_weight_list"):
            return [
                layer.w13_weight_list,
                layer.w2_weight_list,
                layer.w13_weight_scale_list,
                layer.w2_weight_scale_list,
            ]
        return [
            layer.w13_weight.transpose(1, 2),
            layer.w2_weight.transpose(1, 2),
            layer.w13_weight_scale.transpose(1, 2),
            layer.w2_weight_scale.transpose(1, 2),
        ]

    def process_weights_after_loading(self, layer):
        """Cast MoE weights to NZ format and reshape their scales for NPU inference.

        Records the original shapes and marks the layer transformed so
        ``restore_weights_for_rl_loading`` can reverse it before an RL reload.
        EPLB list weights are intentionally not reversible for RL reload.
        """
        if getattr(layer, "_mxfp4_transformed", False):
            return
        if _mxfp4_path() == _MXFP4_PATH_FALLBACK:
            vllm_config = get_current_vllm_config()
            initialize_fp4_int8_layer(layer, vllm_config, self.group_size)
            initialize_fp4_decode_cache(layer, vllm_config)
            layer._mxfp4_transformed = True
            return
        if not hasattr(layer, "_mxfp4_original_shapes"):
            layer._mxfp4_original_shapes = {
                "w13_weight": tuple(layer.w13_weight.data.shape),
                "w13_weight_scale": tuple(layer.w13_weight_scale.data.shape),
                "w2_weight": tuple(layer.w2_weight.data.shape),
                "w2_weight_scale": tuple(layer.w2_weight_scale.data.shape),
            }
        g, n, k = layer.w13_weight_scale.shape
        layer.w13_weight_scale.data = layer.w13_weight_scale.data.reshape(g, n, k // 2, 2)
        g, n, k = layer.w2_weight_scale.shape
        layer.w2_weight_scale.data = layer.w2_weight_scale.data.reshape(g, n, k // 2, 2)

        if self.use_expert_weight_list:
            is_situ = getattr(layer, "activation", None) in (
                MoEActivation.SITU,
                "situ",
            )
            w13_input_dtype = torch.float4_e2m1fn_x2 if is_situ else torch_npu.float4_e2m1fn_x2

            layer.w13_weight_list = []
            for weight in layer.w13_weight.data.unbind(dim=0):
                # aclnnInplaceCopy does not support DT_FLOAT4_E2M1. Clone the
                # packed bytes while the tensor is still uint8, then attach
                # the semantic FP4 dtype required by the SiTU operator.
                expert_weight = weight.clone()
                if is_situ:
                    expert_weight = expert_weight.view(torch.float4_e2m1fn_x2)

                layer.w13_weight_list.append(
                    torch_npu.npu_format_cast(
                        expert_weight,
                        ACL_FORMAT_FRACTAL_NZ,
                        customize_dtype=torch.float8_e4m3fn,
                        input_dtype=w13_input_dtype,
                    )
                )
            layer.w2_weight_list = [
                torch_npu.npu_format_cast(
                    weight.clone(),
                    ACL_FORMAT_FRACTAL_NZ,
                    customize_dtype=torch.float8_e4m3fn,
                    input_dtype=torch_npu.float4_e2m1fn_x2,
                )
                for weight in layer.w2_weight.data.unbind(dim=0)
            ]
            # GMM-SiTU consumes contiguous N-major expert lists. Ordinary GMM
            # gets zero-copy transposed views from _get_mlp_weights.
            layer.w13_weight_scale_list = [scale.clone() for scale in layer.w13_weight_scale.data.unbind(dim=0)]
            layer.w2_weight_scale_list = [scale.clone() for scale in layer.w2_weight_scale.data.unbind(dim=0)]
            del layer.w13_weight
            del layer.w2_weight
            del layer.w13_weight_scale
            del layer.w2_weight_scale
            torch.npu.empty_cache()

        # MegaMoe (FUSED_MC2 on A5) expects the weights in their original
        # (out, in) layout — weight1=(E, 2*inter, hidden//2), weight2=(E,
        # hidden, inter//2) for packed FP4 — and the reshaped (not transposed)
        # scales. Capture per-expert FRACTAL_NZ lists before the non-mega path
        # below format-casts and transposes everything.
        elif use_cann_megamoe(get_current_vllm_config()):
            layer.cann_mega_moe_w13_weight_list = [
                torch_npu.npu_format_cast(
                    weight.clone(),
                    ACL_FORMAT_FRACTAL_NZ,
                    customize_dtype=torch.float8_e4m3fn,
                    input_dtype=torch_npu.float4_e2m1fn_x2,
                )
                for weight in layer.w13_weight.data
            ]
            layer.cann_mega_moe_w2_weight_list = [
                torch_npu.npu_format_cast(
                    weight.clone(),
                    ACL_FORMAT_FRACTAL_NZ,
                    customize_dtype=torch.float8_e4m3fn,
                    input_dtype=torch_npu.float4_e2m1fn_x2,
                )
                for weight in layer.w2_weight.data
            ]
            # Checkpoint E8M0 scales are stored as bytes; CANN requires their semantic dtype.
            layer.cann_mega_moe_w13_weight_scale_list = [
                w13_weight_scale.clone().view(torch.float8_e8m0fnu)
                for w13_weight_scale in layer.w13_weight_scale.data.unbind(dim=0)
            ]
            layer.cann_mega_moe_w2_weight_scale_list = [
                w2_weight_scale.clone().view(torch.float8_e8m0fnu)
                for w2_weight_scale in layer.w2_weight_scale.data.unbind(dim=0)
            ]
            tensor_names = (
                "w13_weight",
                "w2_weight",
                "w13_weight_scale",
                "w2_weight_scale",
            )
            for tensor_name in tensor_names:
                dispose_tensor(getattr(layer, tensor_name))
        else:
            # Non-mega path (npu_grouped_matmul): format-cast to FRACTAL_NZ and
            # transpose to (in, out) so the packed FP4 K is interpreted correctly.
            w13 = layer.w13_weight.data
            w13_input_dtype = torch_npu.float4_e2m1fn_x2
            if getattr(layer, "activation", None) in (MoEActivation.SITU, "situ"):
                # The native GMSQ schema requires FP4 semantic dtype metadata.
                w13 = w13.view(torch.float4_e2m1fn_x2)
                w13_input_dtype = torch.float4_e2m1fn_x2
            layer.w13_weight.data = torch_npu.npu_format_cast(
                w13,
                ACL_FORMAT_FRACTAL_NZ,
                customize_dtype=torch.float8_e4m3fn,
                input_dtype=w13_input_dtype,
            )
            layer.w2_weight.data = torch_npu.npu_format_cast(
                layer.w2_weight.data,
                ACL_FORMAT_FRACTAL_NZ,
                customize_dtype=torch.float8_e4m3fn,
                input_dtype=torch_npu.float4_e2m1fn_x2,
            )
            layer.w13_weight.data = layer.w13_weight.data.transpose(1, 2)
            layer.w2_weight.data = layer.w2_weight.data.transpose(1, 2)
            layer.w13_weight_scale.data = layer.w13_weight_scale.data.transpose(-3, -2)
            layer.w2_weight_scale.data = layer.w2_weight_scale.data.transpose(-3, -2)
        layer._mxfp4_transformed = True

    def restore_weights_for_rl_loading(self, layer):
        """Undo the NZ/scale transform so the weight loader can reload ND weights.

        Reverses the transpose, casts the expert weights back to ND (format 2),
        and restores the scales' original shapes.
        """
        if not getattr(layer, "_mxfp4_transformed", False):
            return
        if _mxfp4_path() == _MXFP4_PATH_FALLBACK:
            clear_fp4_decode_cache(layer)
            layer._mxfp4_transformed = False
            return
        if self.use_expert_weight_list:
            raise RuntimeError("RL weight reload is not supported together with W4A8-MXFP EPLB.")
        layer.w13_weight.data = layer.w13_weight.data.transpose(1, 2)
        layer.w2_weight.data = layer.w2_weight.data.transpose(1, 2)
        layer.w13_weight.data = torch_npu.npu_format_cast(layer.w13_weight.data, 2)
        if getattr(layer, "activation", None) in (MoEActivation.SITU, "situ"):
            layer.w13_weight.data = layer.w13_weight.data.view(torch.uint8)
        layer.w2_weight.data = torch_npu.npu_format_cast(layer.w2_weight.data, 2)
        orig_shapes = layer._mxfp4_original_shapes
        layer.w13_weight_scale.data = layer.w13_weight_scale.data.transpose(-3, -2).reshape(
            orig_shapes["w13_weight_scale"]
        )
        layer.w2_weight_scale.data = layer.w2_weight_scale.data.transpose(-3, -2).reshape(
            orig_shapes["w2_weight_scale"]
        )
        layer._mxfp4_transformed = False

    def _get_mlp_weights(self, layer):
        if self.use_expert_weight_list:
            return (
                [weight.transpose(0, 1) for weight in layer.w13_weight_list],
                [scale.transpose(0, 1) for scale in layer.w13_weight_scale_list],
                [weight.transpose(0, 1) for weight in layer.w2_weight_list],
                [scale.transpose(0, 1) for scale in layer.w2_weight_scale_list],
            )
        return (
            [layer.w13_weight],
            [layer.w13_weight_scale],
            [layer.w2_weight],
            [layer.w2_weight_scale],
        )
    def supports_fused_activation(self, activation) -> bool:
        if _mxfp4_path() == _MXFP4_PATH_FALLBACK:
            return False
        return getattr(activation, "value", activation) in self.fused_activations

    def apply_gmm1_act_quant(self, mlp_compute_input: MoEMlpComputeInput):
        if _mxfp4_path() == _MXFP4_PATH_FALLBACK:
            h = self.apply_gmm1(mlp_compute_input)
            from vllm_ascend.ops.fused_moe.moe_mlp import _unified_apply_activation
            h = _unified_apply_activation(mlp_compute_input, h, self)
            return h, None

        hidden_states = mlp_compute_input.hidden_states
        hidden_states, pertoken_scale = self._quant_hidden_states(hidden_states, mlp_compute_input.dynamic_scale)
        layer = mlp_compute_input.layer
        assert layer is not None
        if (
            mlp_compute_input.activation == MoEActivation.SITU
            and mlp_compute_input.group_list_type in (0, 1)
            and self.group_size == 32
            and (mlp_compute_input.activation_situ_linear_beta or 0.0) > 0.0
        ):
            if self.use_expert_weight_list:
                w1 = layer.w13_weight_list
                w1_scale = layer.w13_weight_scale_list
            else:
                w1 = layer.w13_weight
                w1_scale = layer.w13_weight_scale
            hidden_states, out_scale, _ = DeviceOperator.npu_grouped_matmul_situ_quant(
                x=hidden_states,
                weight=w1,
                weight_scale=w1_scale,
                x_scale=pertoken_scale,
                group_list=mlp_compute_input.group_list,
                group_list_type=mlp_compute_input.group_list_type,
                beta=(
                    1.0 if mlp_compute_input.activation_situ_beta is None else mlp_compute_input.activation_situ_beta
                ),
                linear_beta=mlp_compute_input.activation_situ_linear_beta or 0.0,
                mxfp_quant_dtype=self.quant_type,
            )
            dispose_tensor(mlp_compute_input.hidden_states)
            return hidden_states, maybe_normalize_mxfp_scale_layout(out_scale)
        w1, w1_scale, _, _ = self._get_mlp_weights(layer)
        hidden_states = torch_npu.npu_grouped_matmul(
            x=[hidden_states],
            weight=w1,
            scale=None,
            antiquant_scale=w1_scale,
            scale_dtype=None,
            per_token_scale=[pertoken_scale],
            per_token_scale_dtype=torch_npu.float8_e8m0fnu,
            split_item=2,
            group_type=0,
            group_list=mlp_compute_input.group_list,
            group_list_type=mlp_compute_input.group_list_type,
            x_dtype=torch.float8_e4m3fn,
            weight_dtype=torch_npu.float4_e2m1fn_x2,
            output_dtype=torch.bfloat16,
        )[0]
        dispose_tensor(mlp_compute_input.hidden_states)
        if mlp_compute_input.activation == MoEActivation.SITU:
            # SituAndMul: run the dequantized gmm1 first, then fuse the situ
            # activation with MXFP output quantization (Kimi K3 SITU activation).

            hidden_states, swiglu_out_scale = torch.ops._C_ascend.situ_mx_quant(
                x=hidden_states,
                beta=1.0 if mlp_compute_input.activation_situ_beta is None else mlp_compute_input.activation_situ_beta,
                linear_beta=mlp_compute_input.activation_situ_linear_beta or 0.0,
                activate_left=True,
                dst_type=SITU_MX_DST_TYPE_E4M3FN,
            )
            return hidden_states, maybe_normalize_mxfp_scale_layout(swiglu_out_scale)

        # The `group_index` input for the `npu_swiglu_group_quant` operator
        # currently only supports the `count` type. In the current version, the
        # `npu_swiglu_group_quant` operator performs a summation calculation on
        # `group_index`. Therefore, under the cumsum type, the last value is directly taken
        # and Avoid executing two small operators.
        if mlp_compute_input.group_list_type == 0:
            group_index = mlp_compute_input.group_list[-1:]
        else:
            group_index = cumsum_group_list(mlp_compute_input.group_list, mlp_compute_input.group_list_type, 1)
            hidden_states, out_scale, _ = torch.ops._C_ascend.npu_swiglu_group_quant(
            hidden_states,
            topk_weight=None,
            group_index=group_index,
            dst_type=torch.float8_e4m3fn,
            quant_mode=2,
            clamp_value=mlp_compute_input.swiglu_limit,
        )
        return hidden_states, maybe_normalize_mxfp_scale_layout(out_scale)

    def apply_gmm1(self, mlp_compute_input: MoEMlpComputeInput):
        if _mxfp4_path() == _MXFP4_PATH_FALLBACK:
            layer = mlp_compute_input.layer
            assert layer is not None
            if mlp_compute_input.hidden_states.shape[0] == 0:
                return mlp_compute_input.hidden_states.new_empty(
                    (0, layer.w13_weight.shape[1]), dtype=torch.bfloat16
                )
            if getattr(layer, "_fp4_int8_decoder", None) is not None:
                hidden_states = apply_fp4_int8_gmm(
                    layer, mlp_compute_input, mlp_compute_input.hidden_states,
                    "w13", self.group_size, mlp_compute_input.dynamic_scale,
                )
                dispose_tensor(mlp_compute_input.hidden_states)
                return hidden_states
            w13_gemm, groups = decode_active_fp4_weights(
                layer, mlp_compute_input, "w13", dequant_fp4_to_bf16, self.group_size,
            )
            hidden_states = torch_npu.npu_grouped_matmul(
                x=[mlp_compute_input.hidden_states],
                weight=[w13_gemm],
                split_item=2,
                group_type=0,
                group_list=groups,
                group_list_type=mlp_compute_input.group_list_type,
                output_dtype=torch.bfloat16,
            )[0]
            del w13_gemm
            dispose_tensor(mlp_compute_input.hidden_states)
            return hidden_states

        hidden_states = mlp_compute_input.hidden_states
        hidden_states, pertoken_scale = self._quant_hidden_states(hidden_states, mlp_compute_input.dynamic_scale)
        layer = mlp_compute_input.layer
        assert layer is not None
        w1, w1_scale, _, _ = self._get_mlp_weights(layer)
        hidden_states = torch_npu.npu_grouped_matmul(
            x=[hidden_states],
            weight=w1,
            scale=w1_scale,
            per_token_scale=[pertoken_scale],
            bias=None,
            split_item=2,
            group_type=0,
            group_list=mlp_compute_input.group_list,
            group_list_type=mlp_compute_input.group_list_type,
            output_dtype=torch.bfloat16,
            scale_dtype=torch_npu.float8_e8m0fnu,
            per_token_scale_dtype=torch_npu.float8_e8m0fnu,
        )[0]
        dispose_tensor(mlp_compute_input.hidden_states)
        return hidden_states

    def apply_act_quant(self, mlp_compute_input: MoEMlpComputeInput, hidden_states: torch.Tensor):
        if _mxfp4_path() == _MXFP4_PATH_FALLBACK:
            if getattr(mlp_compute_input.layer, "_fp4_int8_decoder", None) is not None:
                return quantize_fp4_int8_activation(hidden_states)
            return hidden_states, None

        hidden_states, dynamic_scale = torch_npu.npu_dynamic_mx_quant(hidden_states, dst_type=torch.float8_e4m3fn)
        return hidden_states, maybe_normalize_mxfp_scale_layout(dynamic_scale)

    def apply_gmm2(self, mlp_compute_input: MoEMlpComputeInput, hidden_states, act_out_scale):
        if _mxfp4_path() == _MXFP4_PATH_FALLBACK:
            layer = mlp_compute_input.layer
            assert layer is not None
            if hidden_states.shape[0] == 0:
                return hidden_states.new_empty((0, layer.w2_weight.shape[1]), dtype=torch.bfloat16)
            if getattr(layer, "_fp4_int8_decoder", None) is not None:
                return apply_fp4_int8_gmm(
                    layer, mlp_compute_input, hidden_states, "w2", self.group_size, act_out_scale,
                )
            w2_gemm, groups = decode_active_fp4_weights(
                layer, mlp_compute_input, "w2", dequant_fp4_to_bf16, self.group_size,
            )
            out = torch_npu.npu_grouped_matmul(
                x=[hidden_states],
                weight=[w2_gemm],
                split_item=2,
                group_type=0,
                group_list=groups,
                group_list_type=mlp_compute_input.group_list_type,
                output_dtype=torch.bfloat16,
            )[0]
            del w2_gemm
            return out
        layer = mlp_compute_input.layer
        assert layer is not None
        _, _, w2, w2_scale = self._get_mlp_weights(layer)
        input_dtype = mlp_compute_input.hidden_states.dtype
        use_bf16 = input_dtype in [torch.bfloat16, torch.float8_e4m3fn]
        output_dtype = (
            input_dtype
            if input_dtype in [torch.bfloat16, torch.float16]
            else (torch.bfloat16 if use_bf16 else torch.float16)
        )
        return torch_npu.npu_grouped_matmul(
            x=[hidden_states],
            weight=w2,
            scale=None,
            antiquant_scale=w2_scale,
            bias=None,
            per_token_scale=[act_out_scale],
            split_item=2,
            group_list_type=mlp_compute_input.group_list_type,
            group_type=0,
            group_list=mlp_compute_input.group_list,
            output_dtype=output_dtype,
            per_token_scale_dtype=torch_npu.float8_e8m0fnu,
            x_dtype=torch.float8_e4m3fn,
            weight_dtype=torch_npu.float4_e2m1fn_x2,
        )[0]


@register_scheme(FP8_METHOD, "ds_w4a8_moe")
class AscendW4A8MXFPDSDynamicFusedMoEMethod(AscendW4A8MXFPDynamicFusedMoEMethod):
    """FusedMoe method for DS original w4a8 mxfp quantization."""

    model_dtype = None
    quant_type: QuantType = QuantType.W4A8MXFP

    def get_dynamic_quant_param(
        self, num_experts: int, intermediate_size_per_partition: int, hidden_sizes: int, params_dtype: torch.dtype
    ) -> dict[str, Any]:
        # Keep checkpoint E8M0 bytes intact for the BF16 software decoder (A2 fallback).
        # copy_(uint8 -> float8_e8m0fnu) would convert numeric values instead
        # of preserving the exponent encoding, and requires an FP8 NPU cast.

        if self.group_size <= 0:
            raise ValueError(f"FP4 group_size must be positive, got {self.group_size}.")
        if hidden_sizes % self.group_size or intermediate_size_per_partition % self.group_size:
            raise ValueError(
                f"FP4 dimensions ({hidden_sizes}, {intermediate_size_per_partition}) must be divisible "
                f"by group_size={self.group_size}."
            )
        if _mxfp4_path() == _MXFP4_PATH_FALLBACK:
            return super().get_dynamic_quant_param(
                num_experts, intermediate_size_per_partition, hidden_sizes, params_dtype
            )
        param_dict = {}
        param_dict["w13_weight_scale"] = torch.empty(
            num_experts,
            2 * intermediate_size_per_partition,
            hidden_sizes // self.group_size,
            dtype=torch.float8_e8m0fnu,
        )

        param_dict["w2_weight_scale"] = torch.empty(
            num_experts, hidden_sizes, intermediate_size_per_partition // self.group_size, dtype=torch.float8_e8m0fnu
        )
        return param_dict
