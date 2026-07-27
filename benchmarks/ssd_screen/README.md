# Phase 0a runbook — SSD break-even screen on real hardware

For [vllm-project/vllm#36037](https://github.com/vllm-project/vllm/issues/36037). Decides whether Speculative Speculative Decoding could ever repay the extra GPU it needs, before anyone writes engine code.

**The question:** SSD's best case is hiding draft latency entirely, so the speedup ceiling is `1 + draft/verify`. Its drafter needs a dedicated GPU. With a TP=4 target that is 25% more hardware, so drafting must be **>20% of step time** just to break even on tok/s/GPU. Measure the ratio; if it fails, nothing downstream can rescue it.

## Files

| File | Role |
|---|---|
| `ssd_screen.py` | The screen itself: trace parsing + break-even arithmetic. Stdlib only, no vLLM import. |
| `run_ssd_screen.py` | GPU driver. Sweeps configs, profiles each in its own subprocess, prints the table. |

**No vLLM source changes are required.** The scopes `gpu_model_runner: draft` and `gpu_model_runner: forward` already exist ([`gpu_model_runner.py`](../../vllm/v1/worker/gpu_model_runner.py) around lines 4605 and 4436); `VLLM_CUSTOM_SCOPES_FOR_PROFILING=1` turns them from `nullcontext` into real profiler regions. Nothing outside this directory is touched.

## Setup on a Lambda Cloud instance

Pick an instance whose GPU count is `tp + draft_gpus` for the config you want to screen — the whole point is costing the extra drafter GPU, so an 8×H100 node screens a TP=4 target with room to spare.

```bash
git clone -b ssd-phase0a-screen https://github.com/xikronz/vllm.git && cd vllm
```

Per `AGENTS.md`, everything goes through `uv` and `.venv` — never system `python3` or bare `pip`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh && uv venv --python 3.12 && source .venv/bin/activate
```

The precompiled wheel is the fast path and is enough here, since no C++/CUDA in vLLM is being modified:

```bash
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto
```

Gated models need a token before the sweep:

```bash
export HF_TOKEN=your_token_here
```

## Run it

Start with the cheapest possible sanity check — no draft model to download, ~2 minutes, works on a single GPU:

```bash
.venv/bin/python benchmarks/ssd_screen/run_ssd_screen.py --model Qwen/Qwen3-0.6B --tp 1 --spec-config '{"method":"ngram","num_speculative_tokens":8,"prompt_lookup_max":4}' --batch-sizes 1 --output-len 128 --verbose
```

That should print a table with a FAIL for ngram (correct — see Gotchas). If it does, the profiling path works end to end.

Then the real sweep. Substitute your target and drafters:

```bash
.venv/bin/python benchmarks/ssd_screen/run_ssd_screen.py --model meta-llama/Llama-3.1-70B-Instruct --tp 4 --draft-gpus 1 --spec-config '{"method":"eagle3","model":"yuhuili/EAGLE3-LLaMA3.1-Instruct-8B","num_speculative_tokens":8}' --spec-config '{"method":"draft_model","model":"meta-llama/Llama-3.2-1B-Instruct","num_speculative_tokens":8}' --batch-sizes 1,8,32 --json results.json
```

Long sweeps outlive an SSH session; run under `tmux` so a dropped connection does not kill the run.

`--dry-run` validates the matrix without a GPU. `--verbose` streams vLLM logs when a config fails.

### Flags that matter

| Flag | Why |
|---|---|
| `--tp` | Target TP size. Sets the break-even threshold: `draft_gpus/(tp+draft_gpus)`. |
| `--draft-gpus` | GPUs SSD would add. The whole cost side of the ledger. |
| `--batch-sizes` | Pinned via `max_num_seqs` + exactly that many prompts. |
| `--p-hit` | Cache hit rate for the realistic column (default 0.7; paper measures 0.65–0.90). |
| `--profile-iters` | Decode iterations to profile (default 64). More is steadier. |
| `--delay-iters` | Iterations skipped after `start_profile` to clear prefill (default 4). |

## Reading the output

```
  config                 draft ms verify ms  draft%  perGPU@1 perGPU@0.7   verdict
  mtp K=8, b=1               4.64     11.54   28.7%     1.12x     1.00x   PASS
  dflash K=16, b=1           0.90     11.46    7.3%     0.86x     0.84x   FAIL
```

- **`perGPU@1`** — hard upper bound, assumes a perfect speculation cache. **Below 1.00× here, no amount of tuning helps.** This is the number that kills configs.
- **`perGPU@0.7`** — prices in cache misses: only `p_hit·draft` is actually hidden.
- **PASS** clears both. **MARGINAL** clears the bound but not the realistic point — measure `p_hit` (Phase 1) before deciding. **FAIL** is terminal.

Exit code is non-zero when the single-config screen fails, so it composes in scripts.

Compare against the numbers already in the issue thread at any time — no GPU needed:

```bash
.venv/bin/python benchmarks/ssd_screen/ssd_screen.py --reference
```

## How the measurement works

Each `draft` scope is paired with the nearest **preceding** `forward` scope — in `execute_model` the order is forward → sample → draft — which isolates decode steps and drops prefill-only forwards. Reported values are **medians** over the profiled window, so a stray prefill or a scheduler hiccup cannot move them.

Timing is **GPU kernel time attributed through trace correlation ids**, not host wall time. This matters: with CUDA graphs and async launches the CPU-side region duration understates the target forward, which would bias the ratio *toward* SSD and manufacture a false PASS. `--host-time` switches to wall clock.

## Gotchas

- **`VLLM_CUSTOM_SCOPES_FOR_PROFILING=1` must be set before vLLM is imported.** The driver sets it in each subprocess; if you drive vLLM yourself and forget, the scopes silently become `nullcontext` and the tool reports "no draft scopes" rather than a wrong number.
- **CPU-side drafters read ~0 GPU time.** `method: ngram` (the CPU `NgramProposer`) legitimately consumes no GPU, so it will FAIL the screen — which is the correct answer, since SSD has nothing to hide there. Use `--host-time` if you want its wall-clock cost. `ngram_proposer_gpu` does show GPU time.
- **Stack tracing is disabled deliberately.** `torch_profiler_with_stack` defaults to `True` and its per-op overhead would distort the very ratio being measured. The driver sets it `False`.
- **Prefix caching is disabled** so a batch of sequences cannot share work and skew the per-step split.
- **`ignore_eos=True`** keeps the batch full for the whole run; otherwise sequences finish early and the effective batch size drifts below what you asked for.
- **One process per config** — speculative config is fixed at engine construction, and a fresh process avoids CUDA state leaking between runs.

## What this does and does not settle

Settles: whether the draft/verify ratio leaves enough room for SSD to pay for its GPU. A FAIL is decisive.

Does **not** settle: the actual speedup. That needs `p_hit` (Phase 1) and the accepted-length effect (`E_hit/E_SD`), which cuts the other way. Nor does it model the batch-size collapse — at large batch a miss on *any* sequence stalls the whole batch, so the effective rate is `p_hit^b` and near-break-even rows at b=32 are optimistic.

Report `tok/s/GPU`, not `tok/s`. The maintainer's objection is economic: a 5-GPU deployment no longer fits two replicas on an 8×H100 node.

## Status — read before proposing this upstream

This branch exists to **run experiments on a personal fork**, not as an upstream contribution.

`AGENTS.md` §1 forbids low-value standalone PRs, and these scripts import no vLLM internals — as-is they would be exactly that. They belong in `vllm-project/vllm` only bundled with substantive work, most plausibly the Phase 1 `p_hit` instrument, which *does* touch vLLM and is independently useful for any drafter. If this ever becomes a PR, `AGENTS.md` also requires the duplicate-work checks, a human who understands and defends every line, and an explicit statement that AI assistance was used.

**Tested:** module imports, arg validation, `--dry-run`, worker payload round-trip, trace discovery (gzipped, tp0-preferred), the full sweep path against synthetic traces, and the missing-trace failure path. Synthetic MTP- and DFlash-shaped inputs reproduce the reference table exactly.

**Not tested:** anything against a real GPU or a real vLLM-emitted trace. The synthetic traces match the torch schema as read from `vllm/profiler/wrapper.py`, but the first real run is the actual test. Most likely breakage is trace file naming or the `delay_iterations`/`max_iterations` interaction — hence `--verbose` on the smoke test.
