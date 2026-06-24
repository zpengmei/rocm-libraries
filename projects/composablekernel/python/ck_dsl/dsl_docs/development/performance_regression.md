# Performance Regression Tracking

CK DSL measures TFLOPS/GB/s/ms on every run but used to discard the numbers, so a
change to shared code (`core/lower_llvm.py`, a helper, the dispatcher) or a kernel
recipe could quietly slow kernels while still passing correctness.
`ck_dsl.benchmark.regression_store` keeps an **append-only history** and flags a
TFLOPS drop for a given kernel + GPU + shape.

## The key

Regressions are measured against `(arch, kernel_name, M, N, K)` - all three affect
TFLOPS. Shapes stay fixed (they come from real workloads); the *code* is what
varies between runs. `arch` is captured at run time
(`runtime.hip_module.get_device_arch`), so any AMD GPU works and arches coexist in
one history.

## Recording

Both entry points append to the same CSV (replace `<gfx>` with your target; the
record side auto-detects the arch it ran on):

```bash
# sweep - many dispatcher variants at once
python3 -m ck_dsl.benchmark.gemm.fp16_rcr_sweep \
  --arch <gfx> --shape '512,512,512:demo' --run \
  --history history.csv --output-dir /tmp/sweep

# manifest runner - every example funnels through it, so all examples record
python3 -m ck_dsl.run_manifest kernel.hsaco manifest.json \
  --shape 128,4096,4096 --history history.csv
```

Each row carries a shared `run_id` (`<utc-timestamp>_<commit>`) so runs can be
diffed; `commit` comes from CI env (`GITHUB_SHA`), then `git`, then `"unknown"`.

## Detecting

```bash
# exits non-zero on a regression - CI-gateable
python3 -m ck_dsl.benchmark.regression_store --history history.csv \
  --compare --threshold 0.05
```

`compare` flags any key whose `tflops` dropped more than `threshold` (default 5%)
between the two most recent runs; new kernels with no baseline are skipped. The
sweep also accepts `--compare`/`--threshold` to record-and-check in one step.

## Try it (no GPU)

The store is pure data - seed two runs and detect the regression on a laptop:

```python
import ck_dsl.benchmark.regression_store as rs
row = dict(arch="gfx_demo", kernel_name="gemm_x", M=512, N=512, K=512, flop=2*512**3)
rs.append_results("h.csv", rs.stamp_rows([dict(row, ms=1.0, tflops=18.0)],
                  rid="2026-01-01T00:00:00+00:00_old", source="sweep"))
rs.append_results("h.csv", rs.stamp_rows([dict(row, ms=1.3, tflops=13.8)],
                  rid="2026-01-02T00:00:00+00:00_new", source="sweep"))
```

```bash
python3 -m ck_dsl.benchmark.regression_store --history h.csv --compare
# REGRESSION arch=gfx_demo kernel_name=gemm_x M=512 N=512 K=512: tflops 18 -> 13.8 (-23.3%)
```

## Notes

- **Fidelity guard.** `run_manifest` also records `flop`, so
  `consistency_error(row)` checks `tflops ≈ flop/1e9/ms` for any kernel kind -
  catching wrong-column/formula/unit bugs in a real history.
- **Threshold vs noise.** TFLOPS wobbles run-to-run (bimodality, cold-cache first
  runs). Set the threshold above the measured spread to avoid false alarms; to
  test that a *change* is captured, vary the shape (large, deterministic) rather
  than editing the kernel (small, noisy).
- **Backend.** Zero-dependency CSV behind `append_results`/`load_results`; swap
  those two for Parquet+DuckDB at scale without touching the compare logic.

## Validated on gfx1201 (RDNA4)

End-to-end on real hardware: the sweep and `run_manifest` both recorded gfx1201
runs, and `consistency_error` was `0.0` across every recorded row (capture is
faithful). Two *identical* sweep runs showed one kernel swing `21.7 -> 19.8`
TFLOPS (`-8.6%`) from run-to-run noise alone - it tripped a 5% threshold but
passed cleanly at 15%. Confirms the rule above: set the threshold above your
measured noise floor.

## Open questions (team / Yaswanth)

- CI baseline storage: committed golden file vs TheRock run artifact?
- Are `pyarrow`/`duckdb` acceptable deps later, or keep the store zero-dep?
- Sweep as the CI regression harness, or a separate curated example set?
