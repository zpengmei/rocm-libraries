"""Append-only CSV performance-regression store for CK DSL sweeps and examples.

Backend-thin on purpose: ``append_results`` / ``load_results`` / ``compare`` have
no GPU or sweep dependency, so the comparison logic is unit-testable with
synthetic rows. Swap CSV for Parquet later by reimplementing only
``append_results`` / ``load_results`` — the key, compare, and consistency logic
are untouched.

A row is one measured ``(arch, kernel_name, M, N, K)`` point tagged with a
``run_id`` (so successive runs can be diffed). The grouping key is kernel + GPU +
shape because all three independently affect TFLOPS.
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

# kernel + GPU + shape — the grouping key a regression is measured against.
DEFAULT_KEY = ("arch", "kernel_name", "M", "N", "K")


# --- run metadata helpers ----------------------------------------------------


def utc_timestamp() -> str:
    """ISO-8601 UTC timestamp, e.g. ``2026-06-24T17:03:11+00:00``."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def git_commit(cwd: Optional[str] = None) -> str:
    """Short commit hash: CI env first, then ``git``, then ``"unknown"``.

    CI runners expose the SHA via env (``GITHUB_SHA`` on GitHub Actions / TheRock),
    where a ``git`` call may be unavailable or point at the wrong tree.
    """
    for var in ("GITHUB_SHA", "CI_COMMIT_SHA", "COMMIT_SHA"):
        sha = os.environ.get(var)
        if sha:
            return sha[:11]
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def run_id(commit: Optional[str] = None) -> str:
    """Identity for one sweep/example invocation: ``<timestamp>_<commit>``.

    Timestamp-first so lexicographic order == chronological order, which is what
    :func:`split_latest_run` relies on.
    """
    return f"{utc_timestamp()}_{commit or git_commit()}"


def stamp_rows(
    rows: Iterable[Mapping],
    *,
    rid: Optional[str] = None,
    commit: Optional[str] = None,
    source: str = "",
) -> list[dict]:
    """Return copies of ``rows`` with run_id/timestamp/commit/source filled in.

    A single ``run_id`` is shared across all rows of one invocation so the whole
    batch is one comparable unit.
    """
    commit = commit or git_commit()
    rid = rid or run_id(commit)
    ts = utc_timestamp()
    stamped = []
    for r in rows:
        row = dict(r)
        row["run_id"] = rid
        row["timestamp"] = ts
        row["commit"] = commit
        if source:
            row["source"] = source
        stamped.append(row)
    return stamped


# --- storage (the swappable backend) -----------------------------------------


def append_results(
    path, rows: Iterable[Mapping], *, fieldnames: Optional[Sequence[str]] = None
) -> None:
    """Append rows to a CSV, writing the header only when the file is new.

    When the file already exists its existing header is reused, so callers may
    pass rows with extra/missing keys across versions without corrupting columns
    (missing keys are blank; unknown keys raise, surfacing schema drift early).
    """
    path = Path(path)
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    if exists:
        with path.open(newline="") as f:
            header = next(csv.reader(f), None)
        fieldnames = header or list(rows[0].keys())
    elif fieldnames is None:
        fieldnames = list(rows[0].keys())
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        for r in rows:
            writer.writerow(r)


def load_results(path) -> list[dict]:
    """Read every row back as a list of dicts (empty list if file is absent)."""
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


# --- keying / comparison -----------------------------------------------------


def _key(row: Mapping, key_cols: Sequence[str]) -> tuple:
    return tuple(str(row.get(c, "")) for c in key_cols)


def latest_by_key(rows: Iterable[Mapping], *, key_cols=DEFAULT_KEY) -> dict:
    """One row per key — the last occurrence in file (append) order wins."""
    out: dict = {}
    for r in rows:
        out[_key(r, key_cols)] = r
    return out


def split_latest_run(
    history: Iterable[Mapping], *, run_col: str = "run_id"
) -> tuple[list[dict], list[dict]]:
    """Split history into (baseline_rows, current_rows) by the two most recent runs.

    "Current" is the run with the largest ``run_id``; "baseline" is the next one
    down. Returns ``([], current)`` when only one run exists (nothing to compare).
    """
    rows = list(history)
    run_ids = sorted({str(r.get(run_col, "")) for r in rows if r.get(run_col)})
    if not run_ids:
        return [], rows
    current_id = run_ids[-1]
    baseline_id = run_ids[-2] if len(run_ids) >= 2 else None
    current = [r for r in rows if str(r.get(run_col, "")) == current_id]
    baseline = (
        [r for r in rows if str(r.get(run_col, "")) == baseline_id]
        if baseline_id is not None
        else []
    )
    return baseline, current


def compare(
    baseline: Iterable[Mapping],
    current: Iterable[Mapping],
    *,
    key_cols=DEFAULT_KEY,
    metric: str = "tflops",
    threshold: float = 0.05,
) -> list[dict]:
    """Flag current rows whose ``metric`` dropped more than ``threshold`` vs baseline.

    Keys present only in ``current`` (new kernels) are skipped — there is nothing
    to regress against. Returns one dict per regression with the key fields, the
    baseline/current metric, and the relative delta (negative = slower).
    """
    base = latest_by_key(baseline, key_cols=key_cols)
    regressions: list[dict] = []
    for r in current:
        k = _key(r, key_cols)
        if k not in base:
            continue
        try:
            b = float(base[k][metric])
            c = float(r[metric])
        except (TypeError, ValueError, KeyError):
            continue
        if b <= 0:
            continue
        delta = (c - b) / b
        if delta < -threshold:
            regressions.append(
                {
                    **{col: r.get(col, "") for col in key_cols},
                    f"baseline_{metric}": b,
                    f"current_{metric}": c,
                    "delta": delta,
                }
            )
    return regressions


def consistency_error(row: Mapping) -> Optional[float]:
    """Relative error between stored ``tflops`` and ``flop/1e9/ms`` from the same row.

    Kernel-agnostic fidelity guard: if the stored tflops was written into the
    wrong column, computed with the wrong formula, or carries a unit bug, this
    diverges. Uses the ``flop`` count the runner already records, so it holds for
    any kernel kind (not just GEMM's ``2*M*N*K``). Falls back to ``2*M*N*K`` when
    no ``flop`` column is present. Returns None when the row lacks the fields.
    """
    try:
        ms = float(row["ms"])
        stored = float(row["tflops"])
    except (TypeError, ValueError, KeyError):
        return None
    if ms <= 0 or stored <= 0:
        return None
    flop: Optional[float]
    try:
        flop = float(row["flop"])
    except (TypeError, ValueError, KeyError):
        flop = None
    if flop is None:
        try:
            flop = 2.0 * float(row["M"]) * float(row["N"]) * float(row["K"])
        except (TypeError, ValueError, KeyError):
            return None
    if flop <= 0:
        return None
    recomputed = flop / 1e9 / ms
    return abs(stored - recomputed) / recomputed


# --- CLI (CI entry point) ----------------------------------------------------


def _format_regression(reg: Mapping, metric: str) -> str:
    key = " ".join(f"{k}={reg[k]}" for k in DEFAULT_KEY if k in reg)
    b = reg.get(f"baseline_{metric}")
    c = reg.get(f"current_{metric}")
    pct = reg.get("delta", 0.0) * 100.0
    return f"  REGRESSION {key}: {metric} {b:.4g} -> {c:.4g} ({pct:+.1f}%)"


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--history", required=True, type=Path, help="history CSV path")
    ap.add_argument(
        "--compare",
        action="store_true",
        help="diff the latest run vs the previous run and report regressions",
    )
    ap.add_argument("--metric", default="tflops")
    ap.add_argument("--threshold", type=float, default=0.05)
    ns = ap.parse_args(argv)

    history = load_results(ns.history)
    if not history:
        print(f"no history at {ns.history}")
        return 0
    if not ns.compare:
        runs = sorted({r.get("run_id", "") for r in history})
        print(f"{len(history)} rows across {len(runs)} runs in {ns.history}")
        return 0

    baseline, current = split_latest_run(history)
    if not baseline:
        print("only one run present — nothing to compare against")
        return 0
    regressions = compare(
        baseline, current, metric=ns.metric, threshold=ns.threshold
    )
    if not regressions:
        print(f"no {ns.metric} regressions > {ns.threshold:.0%} ({len(current)} rows)")
        return 0
    print(f"{len(regressions)} {ns.metric} regression(s) > {ns.threshold:.0%}:")
    for reg in regressions:
        print(_format_regression(reg, ns.metric))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
