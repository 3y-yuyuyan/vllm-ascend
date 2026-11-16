import inspect
import json
from pathlib import Path
from typing import Optional

import torch
from compressed_tensors.quantization import QuantizationArgs
from vllm.logger import logger
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.quantization.utils.quant_utils import is_layer_skipped
from vllm.models.deepseek_v4 import DeepseekV4FP8Config

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.utils import FP8_METHOD

from ..methods import get_scheme_class
from ..utils import is_fused_moe_layer

# vLLM 0.27.1's is_layer_skipped has no match_mode; newer trees default to exact.
_IS_LAYER_SKIPPED_SUPPORTS_MATCH_MODE = "match_mode" in inspect.signature(is_layer_skipped).parameters


QUANTIZATION_SCHEME_MAP_TYPE = dict[str, dict[str, QuantizationArgs] | None]
DEFAULT_MXFP4_GROUP_SIZE = 32


def normalize_deepseek_quant_name(name: str) -> str:
    """Map ModelSlim checkpoint names to the Ascend DeepSeek module names."""
    name = name.removeprefix("language_model.")
    if name.startswith("layers."):
        name = "model." + name
    elif name.startswith("head."):
        name = "lm_head." + name.removeprefix("head.")
    name = name.replace(".shared_head.", ".")
    for checkpoint, runtime in (
        (".attn.", ".self_attn."),
        (".ffn.", ".mlp."),
        (".w1.", ".gate_proj."),
        (".w2.", ".down_proj."),
        (".w3.", ".up_proj."),
    ):
        name = name.replace(checkpoint, runtime)
    return name


@register_quantization_config(FP8_METHOD)
class AscendFp8Config(Fp8Config):
    """Serve checkpoints published under the generic HF ``fp8`` quant method.

    Only the block-wise flavour is supported: weights in ``float8_e4m3fn`` paired
    with a float32 ``weight_scale_inv`` holding one scale per
    ``weight_block_size`` tile. Layers the checkpoint left in bfloat16 are named
    in ``ignored_layers`` / ``modules_to_not_convert`` and fall back to the
    unquantized methods.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.quant_description = {}

    @classmethod
    def get_min_capability(cls) -> int:
        raise NotImplementedError('Ascend hardware dose not support "get_min_capability" feature.')

    def _verify_block_quantization(self) -> None:
        if self.weight_block_size is None:
            raise NotImplementedError(
                "vLLM Ascend serves native FP8 checkpoints only when their scales are stored per "
                "weight block, that is when `quantization_config.weight_block_size` is present in "
                "config.json. This checkpoint uses per-tensor or per-channel FP8 scales, which have "
                "no Ascend execution path yet. Re-quantize it with ModelSlim and serve with "
                "`--quantization ascend`, or use an llm-compressor checkpoint with "
                "`--quantization compressed-tensors`."
            )
        if self.activation_scheme != "dynamic":
            raise NotImplementedError(
                f"vLLM Ascend supports dynamic activation quantization for native FP8 checkpoints, "
                f"but this one declares `activation_scheme: {self.activation_scheme}`."
            )

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
        tid2eid=None,
    ) -> Optional["QuantizeMethodBase"]:
        # Delayed imports: both modules reach back into the quantization package.
        from vllm_ascend.ops.fused_moe.routed_experts import AscendUnquantizedFusedMoEMethod
        from vllm_ascend.ops.linear import AscendUnquantizedLinearMethod

        from ..method_adapters import (
            AscendFusedMoEMethod,
            AscendLinearMethod,
        )

        is_linear = isinstance(layer, LinearBase)
        is_moe = is_fused_moe_layer(layer)
        if not is_linear and not is_moe:
            return None

        skip_kwargs = {}
        if _IS_LAYER_SKIPPED_SUPPORTS_MATCH_MODE:
            skip_kwargs["match_mode"] = getattr(self, "ignored_layers_match_mode", "exact")
        if is_layer_skipped(
            prefix,
            self.ignored_layers,
            self.packed_modules_mapping,
            **skip_kwargs,
        ):
            if is_linear:
                return AscendUnquantizedLinearMethod()
            return AscendUnquantizedFusedMoEMethod(layer.moe_config, tid2eid)

        self._verify_block_quantization()
        logger.info_once("Using the vLLM Ascend block-wise fp8 Quantization now!")

        if is_linear:
            scheme_class = get_scheme_class(FP8_METHOD, "linear")
            assert scheme_class is not None, f"No scheme registered for {FP8_METHOD}/linear"
            return AscendLinearMethod(scheme_class(self.weight_block_size))

        if is_moe:
            scheme_class = get_scheme_class(FP8_METHOD, "moe")
            assert scheme_class is not None, f"No scheme registered for {FP8_METHOD}/moe"
            return AscendFusedMoEMethod(
                scheme_class(self.weight_block_size, layer.moe_config),
                layer.moe_config,
                tid2eid=tid2eid,
            )

        return None


@register_quantization_config("deepseek_v4_fp8")
class AscendDeepseekV4FP8Config(DeepseekV4FP8Config):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.quant_description = {}
        self._uses_modelslim_linears = False

    def maybe_update_config(self, model_name, hf_config=None, revision=None):
        # Converted W8A8 checkpoints retain their original per-layer metadata.
        # The quant_method name alone cannot describe their INT8/FP4 mixture.
        description_file = Path(model_name) / "quant_model_description.json"
        if not description_file.is_file():
            return super().maybe_update_config(model_name, hf_config=hf_config, revision=revision)
        description = json.loads(description_file.read_text())
        if not isinstance(description, dict):
            raise ValueError(f"Expected a dictionary in {description_file}.")
        has_int8_linears = any(
            key.endswith(".weight") and ".experts." not in key and value == "W8A8_DYNAMIC"
            for key, value in description.items()
        )
        if not has_int8_linears:
            return super().maybe_update_config(model_name, hf_config=hf_config, revision=revision)

        normalized = {}
        for key, value in description.items():
            normalized_key = normalize_deepseek_quant_name(key) if key.endswith(".weight") else key
            if normalized_key in normalized and normalized[normalized_key] != value:
                raise ValueError(f"Conflicting quantization descriptions for {normalized_key}.")
            normalized[normalized_key] = value
        # ModelSlim INT8 uses group_size=0 for per-channel scales. The
        # converted MXFP4 experts instead use the converter's block_size.
        # Their scheme reads group_size from this same description dictionary.
        config_file = Path(model_name) / "config.json"
        if config_file.is_file():
            model_config = json.loads(config_file.read_text())
            quant_config = model_config.get("quantization_config") or {}
        else:
            quant_config = getattr(hf_config, "quantization_config", None) or {}
        expert_group_size = quant_config.get("block_size", DEFAULT_MXFP4_GROUP_SIZE)
        if type(expert_group_size) is not int or expert_group_size <= 0:
            raise ValueError(f"Expected a positive FP4 block_size, got {expert_group_size!r}.")
        normalized["group_size"] = expert_group_size
        self.quant_description = normalized
        self._uses_modelslim_linears = True
        logger.info_once("Using ModelSlim W8A8_DYNAMIC linears with DeepSeek FP4 routed experts.")

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
        tid2eid=None,
    ) -> Optional["QuantizeMethodBase"]:
        from ..method_adapters import (
            AscendFusedMoEMethod,
            AscendLinearMethod,
        )

        if isinstance(layer, LinearBase):
            if get_ascend_device_type() != AscendDeviceType.A5 and "shared_expert" in prefix:
                from vllm_ascend.ops.linear import AscendUnquantizedLinearMethod
                return AscendUnquantizedLinearMethod()
            if getattr(self, "_uses_modelslim_linears", False):
                # Delayed imports: ModelSlim and method adapters depend on this package.
                from vllm_ascend.ops.linear import AscendUnquantizedLinearMethod

                from .modelslim_config import create_scheme_for_layer, get_quant_type_for_layer

                normalized_prefix = normalize_deepseek_quant_name(prefix + ".weight").removesuffix(".weight")
                packed_mapping = getattr(self, "packed_modules_mapping", {})
                if (
                    prefix
                    and normalized_prefix.rsplit(".", 1)[-1] not in packed_mapping
                    and normalized_prefix + ".weight" not in self.quant_description
                ):
                    raise ValueError(f"Missing original ModelSlim quantization description for linear layer {prefix}.")
                quant_type = get_quant_type_for_layer(self.quant_description, normalized_prefix, packed_mapping)
                if quant_type is None:
                    return AscendUnquantizedLinearMethod()
                if quant_type != "W8A8_DYNAMIC":
                    raise NotImplementedError(
                        f"Unsupported mixed DeepSeek linear quantization: {quant_type} ({prefix})."
                    )
                return AscendLinearMethod(create_scheme_for_layer(quant_type, normalized_prefix, "linear"))
            scheme_class = get_scheme_class(FP8_METHOD, "ds_linear")
            assert scheme_class is not None, f"No scheme registered for {FP8_METHOD}/ds_linear"
            quant_method = AscendLinearMethod(scheme_class(self.weight_block_size))
            return quant_method
        if is_fused_moe_layer(layer):
            if self.expert_dtype == "fp4":
                scheme_class = get_scheme_class(FP8_METHOD, "ds_w4a8_moe")
                assert scheme_class is not None, f"No scheme registered for {FP8_METHOD}/ds_w4a8_moe"
            else:
                raise NotImplementedError
            quant_method = AscendFusedMoEMethod(scheme_class(), layer.moe_config, tid2eid=tid2eid)
            return quant_method
        return None


@register_quantization_config("fp4_a8_int")
class AscendFP4A8IntConfig(AscendDeepseekV4FP8Config):
    @classmethod
    def get_name(cls) -> str:
        return "fp4_a8_int"

    @classmethod
    def override_quantization_method(
        cls, hf_quant_cfg, user_quant, hf_config=None
    ) -> Optional[str]:
        if user_quant == "fp4_a8_int":
            return "fp4_a8_int"
        if isinstance(hf_quant_cfg, dict) and hf_quant_cfg.get("quant_method") == "fp4_a8_int":
            return "fp4_a8_int"
        return None
