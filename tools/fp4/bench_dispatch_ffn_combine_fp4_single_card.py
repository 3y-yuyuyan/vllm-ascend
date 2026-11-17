#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Single-card benchmark for dispatch_ffn_combine_fp4 (FP4 All-to-All + GMM1 + SwiGLU + GMM2 + Combine).

Measures latency, throughput, and TFLOPS on a single Ascend NPU (910B / 910_93) with world_size = 1.
"""

import argparse
import random
import sys
import time
import torch

try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
except ImportError:
    torch_npu = None


def get_hccl_comm_name(pg, rank=0):
    if hasattr(pg, "_get_backend"):
        backend = pg._get_backend(torch.device("npu"))
        if hasattr(backend, "get_hccl_comm_name"):
            return backend.get_hccl_comm_name(rank)
    if hasattr(pg, "get_hccl_comm_name"):
        return pg.get_hccl_comm_name(rank)
    from torch.distributed.distributed_c10d import _get_default_group
    default_pg = _get_default_group()
    if hasattr(default_pg, "_get_backend"):
        return default_pg._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
    return "hccl_world_group"


def setup_single_card_hccl(device_id=0):
    import torch.distributed as dist
    torch.npu.set_device(device_id)
    port = 29500 + random.randint(0, 10000)
    init_method = f"tcp://127.0.0.1:{port}"
    dist.init_process_group(
        backend="hccl",
        init_method=init_method,
        rank=0,
        world_size=1,
    )
    from torch.distributed.distributed_c10d import _get_default_group
    default_pg = _get_default_group()
    comm_name = get_hccl_comm_name(default_pg, 0)
    return comm_name


def run_benchmark(args):
    from vllm_ascend.utils import enable_custom_op
    enable_custom_op()

    device = torch.device(f"npu:{args.device}")
    torch.npu.set_device(device)
    comm_name = setup_single_card_hccl(args.device)

    m = args.tokens
    k = args.hidden_size
    n = args.intermediate_size
    e = args.experts
    topk = args.topk
    k2 = n // 2
    n2 = k

    print("=" * 60)
    print(f"Benchmarking dispatch_ffn_combine_fp4 (Single Card NPU:{args.device})")
    print(f"M (Tokens)         : {m}")
    print(f"K (Hidden Size)    : {k}")
    print(f"N (Intermediate)   : {n} (w1 shape: [{e}, {2 * n}, {k // 2}] FP4)")
    print(f"Experts per Rank   : {e}")
    print(f"Top-K              : {topk}")
    print(f"Warmup / Repeats   : {args.warmup} / {args.repeats}")
    print("=" * 60)

    # 1. Inputs on NPU
    x = torch.randn(m, k, dtype=torch.bfloat16, device=device)

    # FP4 packed weights (uint8, 2 elements per byte along K)
    w1_list = [torch.randint(0, 256, (2 * n, k // 2), dtype=torch.uint8, device=device) for _ in range(e)]
    w2_list = [torch.randint(0, 256, (k, k2 // 2), dtype=torch.uint8, device=device) for _ in range(e)]

    # E8M0 scale factors (uint8, 1 scale per 32 elements along K)
    scale1_list = [torch.randint(120, 135, (2 * n, k // 32), dtype=torch.uint8, device=device) for _ in range(e)]
    scale2_list = [torch.randint(120, 135, (k, k2 // 32), dtype=torch.uint8, device=device) for _ in range(e)]

    # Routing
    expert_idx = torch.randint(0, e, (m, topk), dtype=torch.int32, device=device)
    probs = torch.ones(m, topk, dtype=torch.float32, device=device) / topk

    out = torch.empty(m, k, dtype=torch.bfloat16, device=device)
    expert_token_nums = torch.empty(e, dtype=torch.int32, device=device)

    # Max output tokens buffer
    max_output_size = max(m * topk, 512)

    # Warmup
    for _ in range(args.warmup):
        torch.ops._C_ascend.dispatch_ffn_combine(
            x, w1_list, w2_list, expert_idx, scale1_list, scale2_list,
            [], [], probs, comm_name, max_output_size, out,
            expert_token_nums, None, 0.0, 1,
        )
    torch.npu.synchronize(device)

    # Timing
    latencies = []
    for _ in range(args.repeats):
        t0 = time.perf_counter()
        torch.ops._C_ascend.dispatch_ffn_combine(
            x, w1_list, w2_list, expert_idx, scale1_list, scale2_list,
            [], [], probs, comm_name, max_output_size, out,
            expert_token_nums, None, 0.0, 1,
        )
        torch.npu.synchronize(device)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)  # ms

    latencies.sort()
    median_lat = latencies[len(latencies) // 2]
    mean_lat = sum(latencies) / len(latencies)
    p90_lat = latencies[int(len(latencies) * 0.9)]
    p99_lat = latencies[int(len(latencies) * 0.99)]

    # FLOPS calculation:
    # GMM1: 2 * M * topk * K * (2 * N)
    # GMM2: 2 * M * topk * (N / 2) * K
    # Total GEMM FLOPs: 2 * M * topk * K * (2.5 * N) = 5 * M * topk * K * N
    total_flops = 5.0 * m * topk * k * n
    tflops = (total_flops / (median_lat / 1000.0)) / 1e12
    tokens_per_sec = m / (median_lat / 1000.0)

    print(f"Results:")
    print(f"  Median Latency   : {median_lat:.3f} ms")
    print(f"  Mean Latency     : {mean_lat:.3f} ms")
    print(f"  P90 Latency      : {p90_lat:.3f} ms")
    print(f"  P99 Latency      : {p99_lat:.3f} ms")
    print(f"  Effective TFLOPS : {tflops:.2f} TFLOPS")
    print(f"  Throughput       : {tokens_per_sec:.1f} tokens/s")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Single Card Benchmark for dispatch_ffn_combine_fp4")
    parser.add_argument("--device", type=int, default=0, help="NPU device ID")
    parser.add_argument("--tokens", "-m", type=int, default=64, help="Batch tokens (M)")
    parser.add_argument("--hidden-size", "-k", type=int, default=4096, help="Hidden dimension (K)")
    parser.add_argument("--intermediate-size", "-n", type=int, default=1024, help="Intermediate size (N)")
    parser.add_argument("--experts", "-e", type=int, default=8, help="Number of experts on this card")
    parser.add_argument("--topk", type=int, default=8, help="Top-K active experts per token")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--repeats", type=int, default=100, help="Measurement iterations")
    args = parser.parse_args()

    if torch_npu is None or not hasattr(torch, "npu") or not torch.npu.is_available():
        print("ERROR: torch_npu or NPU device not available on this machine.")
        print("Please run this script inside the Ascend NPU container or on an Ascend 910B/910_93 host:")
        print("  python3 tools/fp4/bench_dispatch_ffn_combine_fp4_single_card.py --tokens 64 --experts 8")
        sys.exit(1)

    run_benchmark(args)


if __name__ == "__main__":
    main()
