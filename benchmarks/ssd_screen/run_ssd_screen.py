# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU driver for the Phase 0a SSD screen (vllm-project/vllm#36037).

Sweeps speculative-decoding configs on real hardware, profiles each one, and
reports the draft/verify split that decides whether SSD could ever pay for its
extra GPU. Requires no vLLM source changes: the `gpu_model_runner: draft` and
`gpu_model_runner: forward` scopes already exist and are enabled by
VLLM_CUSTOM_SCOPES_FOR_PROFILING=1.

Each config runs in its own subprocess, since speculative config is fixed at
engine construction and a fresh process avoids CUDA state leaking between runs.

Usage:

    python run_ssd_screen.py \\
        --model Qwen/Qwen3-32B \\
        --tp 4 \\
        --spec-config '{"method":"ngram","num_speculative_tokens":8,
                        "prompt_lookup_max":4}' \\
        --spec-config '{"method":"eagle3","model":"...","num_speculative_tokens":8}' \\
        --batch-sizes 1,8,32

Add --dry-run to validate the matrix and flags without touching a GPU.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ssd_screen import analyze_trace, screen  # noqa: E402

SCOPE_ENV = "VLLM_CUSTOM_SCOPES_FOR_PROFILING"


@dataclass
class RunSpec:
    model: str
    spec_config: dict
    batch_size: int
    tp: int
    input_len: int
    output_len: int
    temperature: float
    gpu_memory_utilization: float
    max_model_len: int | None
    delay_iters: int

    @property
    def label(self) -> str:
        m = self.spec_config.get("method", "?")
        k = self.spec_config.get("num_speculative_tokens", "?")
        return f"{m} K={k}, b={self.batch_size}"


def _make_prompts(n: int, input_len: int, seed: int = 0) -> list[str]:
    """Distinct pseudo-random prompts, so prefix caching cannot collapse them."""
    import random

    rng = random.Random(seed)
    vocab = [f"w{i}" for i in range(4096)]
    return [
        " ".join(rng.choice(vocab) for _ in range(input_len)) for _ in range(n)
    ]


def run_worker(spec: RunSpec, trace_dir: Path, profile_iters: int) -> dict:
    """Run one config end to end inside this process and dump a trace."""
    if os.environ.get(SCOPE_ENV) != "1":
        raise SystemExit(
            f"{SCOPE_ENV}=1 must be set before importing vLLM, otherwise the "
            "draft/forward scopes compile out to nullcontext."
        )

    from vllm import LLM, SamplingParams
    from vllm.config.profiler import ProfilerConfig

    trace_dir.mkdir(parents=True, exist_ok=True)

    engine_kwargs = dict(
        model=spec.model,
        tensor_parallel_size=spec.tp,
        speculative_config=spec.spec_config,
        max_num_seqs=spec.batch_size,
        gpu_memory_utilization=spec.gpu_memory_utilization,
        # A full batch of identical-length decodes is the point; prefix caching
        # would let sequences share work and distort the per-step split.
        enable_prefix_caching=False,
        disable_log_stats=False,
        profiler_config=ProfilerConfig(
            profiler="torch",
            torch_profiler_dir=str(trace_dir),
            # Stack tracing is on by default and adds enough per-op overhead to
            # skew the very ratio being measured.
            torch_profiler_with_stack=False,
            torch_profiler_use_gzip=True,
            # Skip the prefill iterations that open the profiled generate().
            # Chunked prefill can span several, though the median over the
            # decode window makes an imperfect value harmless.
            delay_iterations=spec.delay_iters,
            max_iterations=profile_iters,
        ),
    )
    if spec.max_model_len is not None:
        engine_kwargs["max_model_len"] = spec.max_model_len

    llm = LLM(**engine_kwargs)
    prompts = _make_prompts(spec.batch_size, spec.input_len)

    # Warm up: triggers CUDA graph capture and torch.compile so neither lands
    # inside the profiled window.
    llm.generate(
        prompts,
        SamplingParams(temperature=0.0, max_tokens=16, ignore_eos=True),
        use_tqdm=False,
    )

    sampling = SamplingParams(
        temperature=spec.temperature,
        max_tokens=spec.output_len,
        ignore_eos=True,
    )
    llm.start_profile()
    t0 = time.perf_counter()
    outs = llm.generate(prompts, sampling, use_tqdm=False)
    elapsed = time.perf_counter() - t0
    llm.stop_profile()

    generated = sum(len(o.outputs[0].token_ids) for o in outs)
    return {
        "elapsed_s": elapsed,
        "generated_tokens": generated,
        "tok_per_s": generated / elapsed if elapsed > 0 else 0.0,
    }


def find_trace(trace_dir: Path) -> Path:
    """Newest trace in the directory, preferring the tp0 rank."""
    traces = sorted(
        trace_dir.glob("*.pt.trace.json*"), key=lambda p: p.stat().st_mtime
    )
    if not traces:
        raise FileNotFoundError(f"no trace written under {trace_dir}")
    for t in reversed(traces):
        if "tp0" in t.name or "rank0" in t.name:
            return t
    return traces[-1]


def run_matrix(args: argparse.Namespace) -> list[dict]:
    out_root = Path(args.output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    results = []

    specs = [
        RunSpec(
            model=args.model,
            spec_config=cfg,
            batch_size=b,
            tp=args.tp,
            input_len=args.input_len,
            output_len=args.output_len,
            temperature=args.temperature,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            delay_iters=args.delay_iters,
        )
        for cfg in args.spec_config
        for b in args.batch_sizes
    ]

    print(f"{len(specs)} configs to run\n")
    for i, spec in enumerate(specs, 1):
        tag = f"{spec.spec_config.get('method', 'spec')}_k" \
              f"{spec.spec_config.get('num_speculative_tokens', 0)}_b{spec.batch_size}"
        trace_dir = out_root / tag
        print(f"[{i}/{len(specs)}] {spec.label}")

        if args.dry_run:
            results.append({"label": spec.label, "dry_run": True})
            continue

        payload = {"spec": asdict(spec), "trace_dir": str(trace_dir),
                   "profile_iters": args.profile_iters}
        env = {**os.environ, SCOPE_ENV: "1"}
        proc = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                json.dumps(payload),
            ],
            env=env,
            capture_output=not args.verbose,
            text=True,
        )
        if proc.returncode != 0:
            tail = (proc.stderr or "")[-2000:] if not args.verbose else "(see above)"
            print(f"    FAILED (exit {proc.returncode})\n{tail}\n")
            results.append({"label": spec.label, "error": f"exit {proc.returncode}"})
            continue

        try:
            trace = find_trace(trace_dir)
            draft_us, verify_us, n = analyze_trace(trace, use_gpu_time=True)
        except (FileNotFoundError, SystemExit) as e:
            print(f"    trace analysis failed: {e}\n")
            results.append({"label": spec.label, "error": str(e)})
            continue

        v = screen(draft_us, verify_us, args.tp, args.draft_gpus, args.p_hit)
        verdict = asdict(v)
        verdict["verdict"] = verdict.pop("label")
        row = {
            "label": spec.label,
            "method": spec.spec_config.get("method"),
            "num_speculative_tokens": spec.spec_config.get("num_speculative_tokens"),
            "batch_size": spec.batch_size,
            "iterations": n,
            "draft_ms": draft_us / 1000,
            "verify_ms": verify_us / 1000,
            "trace": str(trace),
            **verdict,
        }
        results.append(row)
        print(
            f"    draft {row['draft_ms']:.2f} ms / verify {row['verify_ms']:.2f} ms"
            f"  ->  {v.draft_frac:.1%} draft share, {v.label}\n"
        )

    return results


def print_summary(results: list[dict], p_hit: float, tp: int, draft_gpus: int) -> None:
    ok = [r for r in results if "draft_frac" in r]
    if not ok:
        print("no successful runs to summarize")
        return
    threshold = draft_gpus / (tp + draft_gpus)
    print(
        f"\n  target TP={tp} + {draft_gpus} draft GPU  ->  break-even draft "
        f"share {threshold:.0%} at p_hit=1, {ok[0]['threshold_p']:.0%} "
        f"at p_hit={p_hit:g}\n"
    )
    print(
        f"  {'config':<22} {'draft ms':>8} {'verify ms':>9} {'draft%':>7}"
        f" {'perGPU@1':>9} {'perGPU@' + format(p_hit, 'g'):>9}   verdict"
    )
    print(f"  {'-' * 22} {'-' * 8} {'-' * 9} {'-' * 7} {'-' * 9} {'-' * 9}   {'-' * 8}")
    for r in ok:
        print(
            f"  {r['label']:<22} {r['draft_ms']:>8.2f} {r['verify_ms']:>9.2f}"
            f" {r['draft_frac']:>7.1%} {r['per_gpu']:>8.2f}x"
            f" {r['per_gpu_p']:>8.2f}x   {r['verdict']}"
        )
    failed = [r for r in results if "error" in r]
    if failed:
        print(f"\n  {len(failed)} config(s) failed:")
        for r in failed:
            print(f"    {r['label']}: {r['error']}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the Phase 0a SSD screen across spec-decode configs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--worker", help=argparse.SUPPRESS)
    p.add_argument("--model", help="target model")
    p.add_argument("--tp", type=int, default=1, help="target tensor parallel size")
    p.add_argument(
        "--spec-config",
        action="append",
        default=[],
        help="vLLM speculative_config as JSON; repeat for each config to screen",
    )
    p.add_argument("--batch-sizes", default="1,8,32", help="comma-separated")
    p.add_argument("--input-len", type=int, default=512)
    p.add_argument("--output-len", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--profile-iters", type=int, default=64)
    p.add_argument(
        "--delay-iters",
        type=int,
        default=4,
        help="engine iterations to skip after start_profile, to clear prefill",
    )
    p.add_argument("--draft-gpus", type=int, default=1)
    p.add_argument("--p-hit", type=float, default=0.7)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--max-model-len", type=int, default=None)
    p.add_argument("--output-dir", default="./ssd_screen_out")
    p.add_argument("--json", help="write full results to this path")
    p.add_argument("--verbose", action="store_true", help="stream vLLM logs")
    p.add_argument("--dry-run", action="store_true", help="validate without a GPU")
    args = p.parse_args()

    if args.worker:
        return args
    if not args.model:
        p.error("--model is required")
    if not args.spec_config:
        p.error(
            "at least one --spec-config is required, e.g.\n"
            "  --spec-config '{\"method\":\"ngram\","
            "\"num_speculative_tokens\":8,\"prompt_lookup_max\":4}'"
        )
    try:
        args.spec_config = [json.loads(c) for c in args.spec_config]
    except json.JSONDecodeError as e:
        p.error(f"bad --spec-config JSON: {e}")
    args.batch_sizes = [int(b) for b in args.batch_sizes.split(",")]
    return args


def main() -> int:
    args = parse_args()

    if args.worker:
        payload = json.loads(args.worker)
        spec = RunSpec(**payload["spec"])
        stats = run_worker(
            spec, Path(payload["trace_dir"]), payload["profile_iters"]
        )
        print(json.dumps(stats))
        return 0

    results = run_matrix(args)
    if not args.dry_run:
        print_summary(results, args.p_hit, args.tp, args.draft_gpus)
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
