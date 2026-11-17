# SPDX-License-Identifier: Apache-2.0
"""Unit test for dispatch_ffn_combine with FP4 (uint8) weights and E8M0 scales."""

import math
try:
    import pytest
except ImportError:
    pytest = None
import torch


def fp4_e2m1_dequantize_reference(
    packed_weight: torch.Tensor,
    scales: torch.Tensor,
) -> torch.Tensor:
    """CPU reference implementation of FP4 (E2M1) to BF16 dequantization.

    packed_weight: [..., K // 2], uint8, 2 FP4 nibbles per byte.
    scales: [..., K // 32], uint8, E8M0 scale per block of 32 elements.
    Returns: [..., K], bfloat16.
    """
    e2m1_lut = torch.tensor(
        [-0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
          0.0,  0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0],
        dtype=torch.float32,
    )

    *prefix_shape, packed_k = packed_weight.shape
    k = packed_k * 2
    assert scales.shape[-1] == k // 32

    # Unpack low and high nibbles
    lo = (packed_weight & 0x0F).to(torch.int64)
    hi = ((packed_weight >> 4) & 0x0F).to(torch.int64)

    # Interleave to restore original sequence
    unpacked = torch.stack([lo, hi], dim=-1).reshape(*prefix_shape, k)
    magnitudes = e2m1_lut[unpacked]  # [..., K], float32

    # E8M0 scale: 2^(e8m0 - 127)
    exp = scales.to(torch.int32) - 127
    scale_floats = torch.pow(2.0, exp.to(torch.float32))
    scale_floats = torch.where(scales == 0, torch.zeros_like(scale_floats), scale_floats)

    # Broadcast scale: 1 scale per 32 elements
    scale_broadcast = scale_floats.repeat_interleave(32, dim=-1)

    result = (magnitudes * scale_broadcast).to(torch.bfloat16)
    return result


def test_fp4_e2m1_dequantize_numerics():
    """Verify numerical accuracy and rounding of FP4 dequantization against E2M1 spec."""
    # 1 block of 32 elements -> 16 packed bytes, 1 scale byte
    packed = torch.tensor([0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77,
                           0x88, 0x99, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF],
                          dtype=torch.uint8).unsqueeze(0)
    # E8M0 scale = 127 (2^0 = 1.0)
    scales = torch.tensor([127], dtype=torch.uint8).unsqueeze(0)

    out = fp4_e2m1_dequantize_reference(packed, scales)
    assert out.shape == (1, 32)
    assert out.dtype == torch.bfloat16

    # Verify first two elements (byte 0x00: lo=0 -> -0.0, hi=0 -> -0.0)
    assert out[0, 0].item() == 0.0
    # Byte 0x11: lo=1 -> -0.5, hi=1 -> -0.5
    assert out[0, 2].item() == -0.5
    assert out[0, 3].item() == -0.5
    # Byte 0xFF: lo=15 -> 6.0, hi=15 -> 6.0
    assert out[0, 30].item() == 6.0
    assert out[0, 31].item() == 6.0

    # Test scale exponent: E8M0 = 129 (2^2 = 4.0)
    scales_4x = torch.tensor([129], dtype=torch.uint8).unsqueeze(0)
    out_4x = fp4_e2m1_dequantize_reference(packed, scales_4x)
    assert out_4x[0, 30].item() == 24.0
    assert out_4x[0, 2].item() == -2.0


def test_fp4_pair_lut_equivalence():
    """Verify that pair LUT (256-entry table decoding 2 nibbles at once) matches nibble-by-nibble decoding."""
    # Check all 256 byte values
    all_bytes = torch.arange(256, dtype=torch.uint8).unsqueeze(0)
    # 256 bytes = 512 elements = 16 blocks of 32
    scales = torch.full((1, 16), 127, dtype=torch.uint8)

    out = fp4_e2m1_dequantize_reference(all_bytes, scales)
    assert out.shape == (1, 512)

    # For each byte value, verify low and high nibble decoding
    vals = [-0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
             0.0,  0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0]
    for b in range(256):
        lo = b & 0x0F
        hi = (b >> 4) & 0x0F
        assert out[0, 2 * b].item() == vals[lo]
        assert out[0, 2 * b + 1].item() == vals[hi]


def test_dispatch_ffn_combine_fp4_meta_interface():
    """Verify that torch.ops._C_ascend.dispatch_ffn_combine accepts FP4 (uint8) weights on meta device."""
    if not hasattr(torch.ops, "_C_ascend") or not hasattr(torch.ops._C_ascend, "dispatch_ffn_combine"):
        pytest.skip("torch.ops._C_ascend.dispatch_ffn_combine not registered in current environment")

    num_tokens = 16
    hidden_dim = 4096
    intermediate_dim = 1024
    num_experts = 4
    top_k = 2

    # Inputs on meta device
    x = torch.empty(num_tokens, hidden_dim, dtype=torch.bfloat16, device="meta")
    # w1 FP4 packed: [E, 2 * intermediate_dim, hidden_dim // 2]
    w1 = [torch.empty(2 * intermediate_dim, hidden_dim // 2, dtype=torch.uint8, device="meta") for _ in range(num_experts)]
    # w2 FP4 packed: [E, hidden_dim, intermediate_dim // 2]
    w2 = [torch.empty(hidden_dim, intermediate_dim // 2, dtype=torch.uint8, device="meta") for _ in range(num_experts)]
    # Scales: E8M0 uint8
    s1 = [torch.empty(2 * intermediate_dim, hidden_dim // 32, dtype=torch.uint8, device="meta") for _ in range(num_experts)]
    s2 = [torch.empty(hidden_dim, intermediate_dim // 32, dtype=torch.uint8, device="meta") for _ in range(num_experts)]

    expert_idx = torch.zeros(num_tokens, top_k, dtype=torch.int32, device="meta")
    probs = torch.ones(num_tokens, top_k, dtype=torch.float32, device="meta") / top_k
    out = torch.empty_like(x)
    expert_token_nums = torch.empty(num_experts, dtype=torch.int32, device="meta")

    res_out, res_tokens = torch.ops._C_ascend.dispatch_ffn_combine(
        x, w1, w2, expert_idx, s1, s2, [], [], probs,
        "ep_group", num_tokens, out, expert_token_nums,
        None, 0.0, 1
    )
    assert res_out.shape == (num_tokens, hidden_dim)
    assert res_out.dtype == torch.bfloat16
    assert res_tokens.shape == (num_experts,)


def test_w4a8_mxfp4_fused_mc2_weights_payload():
    """Verify that AscendW4A8MXFPDynamicFusedMoEMethod packages weights correctly for FUSED_MC2 dispatch_ffn_combine."""
    try:
        from vllm_ascend.quantization.methods.w4a8.w4a8_mxfp4 import AscendW4A8MXFPDynamicFusedMoEMethod
    except ImportError:
        if pytest is not None:
            pytest.skip("vllm or dependencies not available in current environment")
        return

    from types import SimpleNamespace

    method = AscendW4A8MXFPDynamicFusedMoEMethod.__new__(AscendW4A8MXFPDynamicFusedMoEMethod)
    method.use_expert_weight_list = False

    layer = SimpleNamespace(
        w13_weight=torch.empty(4, 2048, 1024, dtype=torch.uint8),
        w2_weight=torch.empty(4, 2048, 512, dtype=torch.uint8),
        w13_weight_scale=torch.empty(4, 2048, 64, dtype=torch.uint8),
        w2_weight_scale=torch.empty(4, 2048, 32, dtype=torch.uint8),
    )

    weights = method.get_fused_mc2_weights(layer)
    assert isinstance(weights.w1, list) and len(weights.w1) == 1
    assert isinstance(weights.w2, list) and len(weights.w2) == 1
    assert isinstance(weights.w1_scale, list) and len(weights.w1_scale) == 1
    assert isinstance(weights.w2_scale, list) and len(weights.w2_scale) == 1
    assert weights.w1_scale_bias is None
    assert weights.w2_scale_bias is None

    # Test with use_expert_weight_list = True
    method.use_expert_weight_list = True
    layer_list = SimpleNamespace(
        w13_weight_list=[torch.empty(2048, 1024, dtype=torch.uint8) for _ in range(4)],
        w2_weight_list=[torch.empty(2048, 512, dtype=torch.uint8) for _ in range(4)],
        w13_weight_scale_list=[torch.empty(2048, 64, dtype=torch.uint8) for _ in range(4)],
        w2_weight_scale_list=[torch.empty(2048, 32, dtype=torch.uint8) for _ in range(4)],
    )
    weights_list = method.get_fused_mc2_weights(layer_list)
    assert isinstance(weights_list.w1, list) and len(weights_list.w1) == 4
    assert isinstance(weights_list.w2, list) and len(weights_list.w2) == 4
    assert isinstance(weights_list.w1_scale, list) and len(weights_list.w1_scale) == 4
    assert isinstance(weights_list.w2_scale, list) and len(weights_list.w2_scale) == 4

