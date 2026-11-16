# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# Patch target: vllm/config/attention.py
# - DeepSeek V4 / similar Ascend sparse-attention models use an ``int8``
#   indexer K cache. Upstream's ``AttentionConfig.indexer_kv_dtype`` field is
#   typed as ``IndexerKVDType = Literal["auto", "bf16", "fp8", "mxfp4",
#   "nvfp4"]`` and rejects ``"int8"`` at CLI/config validation time with a pydantic
#   ``literal_error`` -- *before* the value ever reaches
#   ``kv_cache_dtype_str_to_dtype``, which does support ``int8`` (it maps to
#   ``torch.int8`` upstream, see ``worker/patch_kv_cache_dtype.py``).
#   This patch widens the accepted ``Literal`` to include ``"int8"`` without
#   touching the upstream vllm tree, by rebuilding the pydantic dataclass
#   schema in place.
#
# Why a platform patch:
#   ``vllm serve`` builds its argparse parser in
#   ``AsyncEngineArgs.add_cli_args`` (``vllm/engine/arg_utils.py``). After all
#   ``--attention-config`` arguments are registered it calls
#   ``current_platform.pre_register_and_update(parser)``, which runs
#   ``adapt_patch(is_global_patch=True)`` (i.e. this module). ``parse_args`` is
#   only called afterwards from the CLI ``main``. So rebuilding the schema here
#   makes the new ``Literal`` visible to the per-call
#   ``TypeAdapter(AttentionConfig).validate_json`` used by the
#   ``--attention-config.indexer_kv_dtype`` dotted-path argument -- in time for
#   validation to accept ``int8``.

import typing

import pydantic.dataclasses as _pdc
from vllm.config import attention as _attention_mod
from vllm.config.attention import AttentionConfig
from vllm.logger import logger

# The upstream Literal that gates ``indexer_kv_dtype`` (currently
# ``Literal["auto", "bf16", "fp8", "mxfp4", "nvfp4"]``). We widen it to also
# accept ``"int8"`` (DeepSeek V4 Ascend indexer cache dtype). Keep the existing
# members verbatim so non-int8 behavior is unchanged -- this includes the
# ``"auto"`` sentinel: the draft proposer's ``_create_draft_vllm_config``
# re-instantiates ``AttentionConfig`` via ``replace()``, which passes every
# field (including the default ``indexer_kv_dtype="auto"``) explicitly, so a
# Literal without ``"auto"`` raises a pydantic ``literal_error`` at engine
# startup. Deriving from the current upstream members keeps this patch in sync
# if upstream adds more formats. (The starred-subscript Literal is only
# analyzable at runtime, so mypy flags it as an invalid type alias -- the
# explicit ignore below is the same pattern used elsewhere in this package.)
_ORIG_INDEXER_KV_DTYPE = getattr(_attention_mod, "IndexerKVDType", None)
if _ORIG_INDEXER_KV_DTYPE is not None:
    _INDEXER_KV_DTYPE_WITH_INT8 = typing.Literal[*_ORIG_INDEXER_KV_DTYPE.__args__ + ("int8",)]  # type: ignore[valid-type]
else:
    _INDEXER_KV_DTYPE_WITH_INT8 = None


def _apply_indexer_kv_dtype_int8_patch() -> None:
    """Widen ``AttentionConfig.indexer_kv_dtype`` to accept ``"int8"``."""
    if _ORIG_INDEXER_KV_DTYPE is None or _INDEXER_KV_DTYPE_WITH_INT8 is None:
        return
    # Idempotency: if int8 is already accepted, nothing to do.
    existing_args = getattr(_ORIG_INDEXER_KV_DTYPE, "__args__", ())
    if "int8" in existing_args:
        return

    _attention_mod.IndexerKVDType = _INDEXER_KV_DTYPE_WITH_INT8
    AttentionConfig.__annotations__["indexer_kv_dtype"] = _INDEXER_KV_DTYPE_WITH_INT8
    AttentionConfig.__dataclass_fields__["indexer_kv_dtype"].type = _INDEXER_KV_DTYPE_WITH_INT8

    # Rebuild the pydantic dataclass schema so the new Literal is enforced.
    # ``_parent_namespace_depth=1`` makes pydantic resolve types against this
    # module's namespace (where the new Literal lives).
    _pdc.rebuild_dataclass(
        AttentionConfig,
        force=True,
        raise_errors=True,
        _parent_namespace_depth=1,
    )
    logger.info("Patched AttentionConfig.indexer_kv_dtype to accept 'int8' (DeepSeek V4 Ascend indexer K cache dtype).")


_apply_indexer_kv_dtype_int8_patch()
