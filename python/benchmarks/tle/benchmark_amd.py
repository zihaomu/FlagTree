"""Reproducible A/B microbenchmarks for TLE on AMD GPUs."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import torch

import triton
import triton.language as tl
import triton.experimental.tle.language as tle


QUANTILES = (0.5, 0.2, 0.8)
BENCH_WARMUP_MS = 25
BENCH_REP_MS = 100
RANDOM_SEED = 0
BOOTSTRAP_SAMPLES = 10_000
REPO_ROOT = Path(__file__).resolve().parents[3]
NUM_WARPS_CONFIGS = {
    "matched": {
        "cumsum": {"triton": 4, "tle": 4},
        "axpy": {"triton": 4, "tle": 4},
        "copy": {"triton": 4, "tle": 4},
        "gather": {"triton": 4, "tle": 4},
        "matmul": {"triton": 4, "tle": 4},
    },
    "gfx1201-tuned": {
        "cumsum": {"triton": 4, "tle": 4},
        "axpy": {"triton": 4, "tle": 1},
        "copy": {"triton": 2, "tle": 2},
        "gather": {"triton": 2, "tle": 2},
        "matmul": {"triton": 1, "tle": 1},
    },
}



@triton.jit
def _tle_cumsum_kernel(
    x_ptr,
    out_ptr,
    n,
    row_stride,
    BLOCK: tl.constexpr,
    REVERSE: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + row * row_stride + offsets, mask=mask, other=0)
    exclusive, _ = tle.cumsum(x, axis=0, reverse=REVERSE)
    tl.store(out_ptr + row * row_stride + offsets, exclusive, mask=mask)


@triton.jit
def _triton_cumsum_kernel(
    x_ptr,
    out_ptr,
    n,
    row_stride,
    BLOCK: tl.constexpr,
    REVERSE: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + row * row_stride + offsets, mask=mask, other=0)
    inclusive = tl.cumsum(x, axis=0, reverse=REVERSE)
    tl.store(out_ptr + row * row_stride + offsets, inclusive - x, mask=mask)


@triton.jit
def _tle_local_ptr_axpy_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    numel,
    alpha,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel

    smem_tile = tle.gpu.alloc(
        [BLOCK],
        dtype=tl.float32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    smem_ptrs = tle.gpu.local_ptr(smem_tile, (tl.arange(0, BLOCK), ))

    x_values = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    tl.store(smem_ptrs, x_values, mask=mask)
    shared_values = tl.load(smem_ptrs, mask=mask, other=0.0)
    y_values = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    updated = shared_values * alpha + y_values
    tl.store(smem_ptrs, updated, mask=mask)
    tl.store(
        out_ptr + offsets,
        tl.load(smem_ptrs, mask=mask, other=0.0),
        mask=mask,
    )


@triton.jit
def _triton_axpy_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    numel,
    alpha,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel
    x_values = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y_values = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x_values * alpha + y_values, mask=mask)


@triton.jit
def _tle_copy_2d_kernel(
    x_ptr,
    out_ptr,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
):
    tile_offset = tl.program_id(0) * ROWS * COLS
    rows = tl.arange(0, ROWS)[:, None]
    cols = tl.arange(0, COLS)[None, :]
    offsets = tile_offset + rows * COLS + cols
    smem = tle.gpu.alloc(
        [ROWS, COLS],
        dtype=tl.float32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    tle.gpu.copy(x_ptr + offsets, smem, [ROWS, COLS])
    tl.store(out_ptr + offsets, tl.load(tle.gpu.local_ptr(smem)))


@triton.jit
def _triton_copy_2d_kernel(
    x_ptr,
    out_ptr,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
):
    tile_offset = tl.program_id(0) * ROWS * COLS
    rows = tl.arange(0, ROWS)[:, None]
    cols = tl.arange(0, COLS)[None, :]
    offsets = tile_offset + rows * COLS + cols
    tl.store(out_ptr + offsets, tl.load(x_ptr + offsets))


@triton.jit
def _tle_axis_gather_kernel(
    x_ptr,
    out_ptr,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    SLICE: tl.constexpr,
):
    tile = tl.program_id(0)
    input_tile_offset = tile * ROWS * COLS
    output_tile_offset = tile * ROWS * SLICE
    rows = tl.arange(0, ROWS)[:, None]
    cols = tl.arange(0, COLS)[None, :]
    smem = tle.gpu.alloc(
        [ROWS, COLS],
        dtype=tl.float32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    tle.gpu.copy(
        x_ptr + input_tile_offset + rows * COLS + cols,
        smem,
        [ROWS, COLS],
    )
    gather_rows = tl.broadcast_to(rows, (ROWS, SLICE))
    gather_cols = tl.broadcast_to(
        1 + tl.arange(0, SLICE)[None, :],
        (ROWS, SLICE),
    )
    values = tl.load(tle.gpu.local_ptr(smem, (gather_rows, gather_cols)))
    output_offsets = output_tile_offset + rows * SLICE + tl.arange(0, SLICE)[None, :]
    tl.store(out_ptr + output_offsets, values)


@triton.jit
def _triton_axis_gather_kernel(
    x_ptr,
    out_ptr,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    SLICE: tl.constexpr,
):
    tile = tl.program_id(0)
    input_tile_offset = tile * ROWS * COLS
    output_tile_offset = tile * ROWS * SLICE
    rows = tl.arange(0, ROWS)[:, None]
    cols = 1 + tl.arange(0, SLICE)[None, :]
    input_offsets = input_tile_offset + rows * COLS + cols
    output_offsets = output_tile_offset + rows * SLICE + tl.arange(0, SLICE)[None, :]
    tl.store(out_ptr + output_offsets, tl.load(x_ptr + input_offsets))


@triton.jit
def _tle_tiled_matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SLICE_WIDTH: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    smem_a = tle.gpu.alloc(
        [BLOCK_M, BLOCK_K],
        dtype=tl.float16,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    smem_b = tle.gpu.alloc(
        [BLOCK_K, BLOCK_N],
        dtype=tl.float16,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_tile in range(0, K, BLOCK_K):
        offsets_k = k_tile + tl.arange(0, BLOCK_K)
        a_offsets = offsets_m[:, None] * K + offsets_k[None, :]
        b_offsets = offsets_k[:, None] * N + offsets_n[None, :]
        tle.gpu.copy(a_ptr + a_offsets, smem_a, [BLOCK_M, BLOCK_K])
        tle.gpu.copy(b_ptr + b_offsets, smem_b, [BLOCK_K, BLOCK_N])
        for slice_start in range(0, BLOCK_K, SLICE_WIDTH):
            a_rows = tl.broadcast_to(
                tl.arange(0, BLOCK_M)[:, None],
                (BLOCK_M, SLICE_WIDTH),
            )
            a_cols = tl.broadcast_to(
                slice_start + tl.arange(0, SLICE_WIDTH)[None, :],
                (BLOCK_M, SLICE_WIDTH),
            )
            b_rows = tl.broadcast_to(
                slice_start + tl.arange(0, SLICE_WIDTH)[:, None],
                (SLICE_WIDTH, BLOCK_N),
            )
            b_cols = tl.broadcast_to(
                tl.arange(0, BLOCK_N)[None, :],
                (SLICE_WIDTH, BLOCK_N),
            )
            a_values = tl.load(tle.gpu.local_ptr(smem_a, (a_rows, a_cols)))
            b_values = tl.load(tle.gpu.local_ptr(smem_b, (b_rows, b_cols)))
            accumulator += tl.dot(a_values, b_values, out_dtype=tl.float32)
    c_offsets = offsets_m[:, None] * N + offsets_n[None, :]
    tl.store(c_ptr + c_offsets, accumulator)


@triton.jit
def _triton_tiled_matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SLICE_WIDTH: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_tile in range(0, K, BLOCK_K):
        for slice_start in range(0, BLOCK_K, SLICE_WIDTH):
            offsets_k = k_tile + slice_start + tl.arange(0, SLICE_WIDTH)
            a_offsets = offsets_m[:, None] * K + offsets_k[None, :]
            b_offsets = offsets_k[:, None] * N + offsets_n[None, :]
            a_values = tl.load(a_ptr + a_offsets)
            b_values = tl.load(b_ptr + b_offsets)
            accumulator += tl.dot(a_values, b_values, out_dtype=tl.float32)
    c_offsets = offsets_m[:, None] * N + offsets_n[None, :]
    tl.store(c_ptr + c_offsets, accumulator)


def _git_output(*args: str) -> str:
    result = subprocess.run(
        ("git", "-C", str(REPO_ROOT), *args),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _environment() -> dict[str, object]:
    target = triton.runtime.driver.active.get_current_target()
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_output("rev-parse", "HEAD"),
        "git_branch": _git_output("branch", "--show-current"),
        "git_dirty": bool(_git_output("status", "--porcelain")),
        "torch_version": torch.__version__,
        "rocm_version": torch.version.hip,
        "triton_version": triton.__version__,
        "device": torch.cuda.get_device_name(),
        "backend": target.backend,
        "arch": target.arch,
        "warp_size": target.warp_size,
    }


def _summarize_rounds(rounds: list[dict[str, float]]) -> dict[str, object]:
    p50_values = [round_result["p50_ms"] for round_result in rounds]
    p50_median = statistics.median(p50_values)
    p50_mean = statistics.mean(p50_values)
    coefficient_of_variation = (
        statistics.pstdev(p50_values) / p50_mean if len(p50_values) > 1 else 0.0
    )
    return {
        "rounds": rounds,
        "p50_ms_median": p50_median,
        "p50_us_median": p50_median * 1_000.0,
        "p50_coefficient_of_variation": coefficient_of_variation,
    }


def _bootstrap_speedup_ci(
    triton_rounds: list[dict[str, float]],
    tle_rounds: list[dict[str, float]],
) -> list[float]:
    rng = random.Random(RANDOM_SEED)
    sample_count = len(triton_rounds)
    estimates = []
    for _ in range(BOOTSTRAP_SAMPLES):
        indices = [rng.randrange(sample_count) for _ in range(sample_count)]
        triton_median = statistics.median(
            triton_rounds[index]["p50_ms"] for index in indices
        )
        tle_median = statistics.median(
            tle_rounds[index]["p50_ms"] for index in indices
        )
        estimates.append(triton_median / tle_median)

    estimates.sort()
    lower_index = math.floor(0.025 * (BOOTSTRAP_SAMPLES - 1))
    upper_index = math.ceil(0.975 * (BOOTSTRAP_SAMPLES - 1))
    return [estimates[lower_index], estimates[upper_index]]


def _measure_pair(
    triton_launch: Callable[[], None],
    tle_launch: Callable[[], None],
    rounds: int,
) -> dict[str, object]:
    samples: dict[str, list[dict[str, float]]] = {"triton": [], "tle": []}
    providers = {"triton": triton_launch, "tle": tle_launch}

    triton_launch()
    tle_launch()
    torch.cuda.synchronize()

    for round_index in range(rounds):
        order = ("triton", "tle") if round_index % 2 == 0 else ("tle", "triton")
        for provider in order:
            p50_ms, p20_ms, p80_ms = triton.testing.do_bench(
                providers[provider],
                warmup=BENCH_WARMUP_MS,
                rep=BENCH_REP_MS,
                quantiles=QUANTILES,
            )
            samples[provider].append({
                "p20_ms": float(p20_ms),
                "p50_ms": float(p50_ms),
                "p80_ms": float(p80_ms),
            })

    summaries = {
        provider: _summarize_rounds(provider_rounds)
        for provider, provider_rounds in samples.items()
    }
    summaries["speedup_vs_triton"] = (
        summaries["triton"]["p50_ms_median"]
        / summaries["tle"]["p50_ms_median"]
    )
    summaries["speedup_95_ci"] = _bootstrap_speedup_ci(
        samples["triton"], samples["tle"]
    )
    return summaries


def _run_cumsum(
    rows: int,
    n: int,
    block: int,
    rounds: int,
    num_warps: dict[str, int],
) -> dict[str, object]:
    if n > block or not math.log2(block).is_integer():
        raise ValueError("cumsum requires n <= BLOCK and a power-of-two BLOCK")
    x = torch.randint(-16, 17, (rows, block), device="cuda", dtype=torch.int32)
    triton_out = torch.empty_like(x)
    tle_out = torch.empty_like(x)

    triton_launch = lambda: _triton_cumsum_kernel[(rows, )](
        x,
        triton_out,
        n,
        block,
        BLOCK=block,
        REVERSE=False,
        num_warps=num_warps["triton"],
        num_stages=1,
    )
    tle_launch = lambda: _tle_cumsum_kernel[(rows, )](
        x,
        tle_out,
        n,
        block,
        BLOCK=block,
        REVERSE=False,
        num_warps=num_warps["tle"],
        num_stages=1,
    )

    triton_launch()
    tle_launch()
    torch.testing.assert_close(tle_out[:, :n], triton_out[:, :n])

    return {
        "case": "cumsum",
        "parameters": {
            "rows": rows,
            "n": n,
            "block": block,
            "dtype": "int32",
            "reverse": False,
            "num_warps": num_warps,
        },
        "correct": True,
        "measurements": _measure_pair(triton_launch, tle_launch, rounds),
    }


def _run_axpy(
    numel: int,
    block: int,
    rounds: int,
    num_warps: dict[str, int],
) -> dict[str, object]:
    x = torch.randn(numel, device="cuda", dtype=torch.float32)
    y = torch.randn_like(x)
    triton_out = torch.empty_like(x)
    tle_out = torch.empty_like(x)
    alpha = 0.75
    grid = (triton.cdiv(numel, block), )

    triton_launch = lambda: _triton_axpy_kernel[grid](
        x, y, triton_out, numel, alpha, BLOCK=block, num_warps=num_warps["triton"]
    )
    tle_launch = lambda: _tle_local_ptr_axpy_kernel[grid](
        x, y, tle_out, numel, alpha, BLOCK=block, num_warps=num_warps["tle"]
    )

    triton_launch()
    tle_launch()
    torch.testing.assert_close(tle_out, triton_out)

    return {
        "case": "local_ptr_axpy",
        "parameters": {
            "numel": numel,
            "block": block,
            "dtype": "float32",
            "alpha": alpha,
            "num_warps": num_warps,
            "expected_role": "negative_control",
        },
        "correct": True,
        "measurements": _measure_pair(triton_launch, tle_launch, rounds),
    }


def _run_copy(
    tiles: int,
    rows: int,
    cols: int,
    rounds: int,
    num_warps: dict[str, int],
) -> dict[str, object]:
    numel = tiles * rows * cols
    x = torch.randn(numel, device="cuda", dtype=torch.float32)
    triton_out = torch.empty_like(x)
    tle_out = torch.empty_like(x)
    grid = (tiles, )

    triton_launch = lambda: _triton_copy_2d_kernel[grid](
        x, triton_out, ROWS=rows, COLS=cols, num_warps=num_warps["triton"]
    )
    tle_launch = lambda: _tle_copy_2d_kernel[grid](
        x, tle_out, ROWS=rows, COLS=cols, num_warps=num_warps["tle"]
    )
    triton_launch()
    tle_launch()
    torch.testing.assert_close(tle_out, triton_out)

    return {
        "case": "local_ptr_copy_2d",
        "parameters": {
            "tiles": tiles,
            "rows": rows,
            "cols": cols,
            "dtype": "float32",
            "num_warps": num_warps,
            "expected_role": "negative_control",
        },
        "correct": True,
        "measurements": _measure_pair(triton_launch, tle_launch, rounds),
    }


def _run_gather(
    tiles: int,
    rows: int,
    cols: int,
    slice_width: int,
    rounds: int,
    num_warps: dict[str, int],
) -> dict[str, object]:
    if 1 + slice_width > cols:
        raise ValueError("gather requires 1 + SLICE <= COLS")
    x = torch.randn(tiles * rows * cols, device="cuda", dtype=torch.float32)
    triton_out = torch.empty(
        tiles * rows * slice_width,
        device="cuda",
        dtype=torch.float32,
    )
    tle_out = torch.empty_like(triton_out)
    grid = (tiles, )

    triton_launch = lambda: _triton_axis_gather_kernel[grid](
        x,
        triton_out,
        ROWS=rows,
        COLS=cols,
        SLICE=slice_width,
        num_warps=num_warps["triton"],
    )
    tle_launch = lambda: _tle_axis_gather_kernel[grid](
        x,
        tle_out,
        ROWS=rows,
        COLS=cols,
        SLICE=slice_width,
        num_warps=num_warps["tle"],
    )
    triton_launch()
    tle_launch()
    torch.testing.assert_close(tle_out, triton_out)

    return {
        "case": "local_ptr_axis_gather",
        "parameters": {
            "tiles": tiles,
            "rows": rows,
            "cols": cols,
            "slice": slice_width,
            "dtype": "float32",
            "num_warps": num_warps,
            "expected_role": "negative_control_reuse_1",
        },
        "correct": True,
        "measurements": _measure_pair(triton_launch, tle_launch, rounds),
    }


def _run_matmul(
    m: int,
    n: int,
    k: int,
    rounds: int,
    num_warps: dict[str, int],
) -> dict[str, object]:
    block_m = 32
    block_n = 32
    block_k = 32
    slice_width = 16
    if m % block_m or n % block_n or k % block_k:
        raise ValueError("matmul dimensions must be divisible by their block sizes")
    a = torch.randn((m, k), device="cuda", dtype=torch.float16)
    b = torch.randn((k, n), device="cuda", dtype=torch.float16)
    triton_out = torch.empty((m, n), device="cuda", dtype=torch.float32)
    tle_out = torch.empty_like(triton_out)
    grid = (m // block_m, n // block_n)
    constants = {
        "M": m,
        "N": n,
        "K": k,
        "BLOCK_M": block_m,
        "BLOCK_N": block_n,
        "BLOCK_K": block_k,
        "SLICE_WIDTH": slice_width,
        "num_stages": 1,
    }
    triton_launch = lambda: _triton_tiled_matmul_kernel[grid](
        a, b, triton_out, **constants, num_warps=num_warps["triton"]
    )
    tle_launch = lambda: _tle_tiled_matmul_kernel[grid](
        a, b, tle_out, **constants, num_warps=num_warps["tle"]
    )
    triton_launch()
    tle_launch()
    expected = a.float() @ b.float()
    torch.testing.assert_close(triton_out, expected, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(tle_out, expected, atol=5e-2, rtol=5e-2)

    return {
        "case": "local_ptr_tiled_matmul",
        "parameters": {
            "m": m,
            "n": n,
            "k": k,
            "dtype": "float16",
            "accumulator_dtype": "float32",
            "block_m": block_m,
            "block_n": block_n,
            "block_k": block_k,
            "slice_width": slice_width,
            "num_warps": num_warps,
        },
        "correct": True,
        "measurements": _measure_pair(triton_launch, tle_launch, rounds),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        choices=("all", "cumsum", "axpy", "copy", "gather", "matmul"),
        default="all",
    )
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument(
        "--config-mode",
        choices=tuple(NUM_WARPS_CONFIGS),
        default="matched",
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.rounds < 1:
        parser.error("--rounds must be at least 1")

    torch.manual_seed(RANDOM_SEED)
    num_warps = NUM_WARPS_CONFIGS[args.config_mode]
    results: list[dict[str, object]] = []
    if args.case in ("all", "cumsum"):
        rows = 256 if args.quick else 4096
        results.append(
            _run_cumsum(rows, 128, 128, args.rounds, num_warps["cumsum"])
        )
    if args.case in ("all", "axpy"):
        numel = 1 << (16 if args.quick else 24)
        results.append(
            _run_axpy(numel, 64, args.rounds, num_warps["axpy"])
        )
    if args.case in ("all", "copy"):
        tiles = 256 if args.quick else 16384
        results.append(
            _run_copy(tiles, 8, 8, args.rounds, num_warps["copy"])
        )
    if args.case in ("all", "gather"):
        tiles = 256 if args.quick else 16384
        results.append(
            _run_gather(tiles, 8, 8, 4, args.rounds, num_warps["gather"])
        )
    if args.case in ("all", "matmul"):
        size = 64 if args.quick else 512
        k = 64 if args.quick else 256
        results.append(
            _run_matmul(size, size, k, args.rounds, num_warps["matmul"])
        )

    report = {
        "schema_version": 2,
        "environment": _environment(),
        "measurement_config": {
            "config_mode": args.config_mode,
            "rounds": args.rounds,
            "warmup_ms": BENCH_WARMUP_MS,
            "rep_ms": BENCH_REP_MS,
            "quantiles": list(QUANTILES),
            "provider_order": "alternating",
            "random_seed": RANDOM_SEED,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "confidence_level": 0.95,
            "confidence_method": "paired bootstrap of ratio of p50 medians",
        },
        "results": results,
    }
    serialized = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
