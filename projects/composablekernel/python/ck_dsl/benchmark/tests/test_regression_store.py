# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""No-GPU unit tests for the performance-regression store.

All synthetic — no kernels, no GPU, no sweep. Proves the storage/compare logic in
isolation. Run: ``python3 -m unittest test_regression_store``.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

try:  # in-repo (PYTHONPATH=python) vs standalone (laptop staging dir)
    from ck_dsl.benchmark import regression_store as rs
except ImportError:
    import regression_store as rs


def _row(*, arch="gfx1201", kernel="k1", M=512, N=512, K=512, tflops=18.0,
         ms=1.0, run_id="r1", **extra):
    row = {
        "run_id": run_id,
        "arch": arch,
        "kernel_name": kernel,
        "M": M,
        "N": N,
        "K": K,
        "ms": ms,
        "tflops": tflops,
    }
    row.update(extra)
    return row


class TestAppendLoad(unittest.TestCase):
    def setUp(self):
        self.path = Path(tempfile.mkdtemp(prefix="rs_test_")) / "history.csv"

    def test_append_creates_header_then_appends_without_duplicating(self):
        rs.append_results(self.path, [_row(run_id="r1")])
        rs.append_results(self.path, [_row(run_id="r2")])
        text = self.path.read_text()
        # header appears exactly once; two data rows follow.
        self.assertEqual(text.count("kernel_name"), 1)
        rows = rs.load_results(self.path)
        self.assertEqual(len(rows), 2)
        self.assertEqual({r["run_id"] for r in rows}, {"r1", "r2"})

    def test_value_round_trips_exactly(self):
        # capture fidelity: the number written is the number read back.
        rs.append_results(self.path, [_row(tflops=18.498730799372698)])
        rows = rs.load_results(self.path)
        self.assertEqual(rows[0]["tflops"], "18.498730799372698")

    def test_empty_rows_is_noop(self):
        rs.append_results(self.path, [])
        self.assertFalse(self.path.exists())

    def test_existing_header_is_reused_on_schema_drift(self):
        rs.append_results(self.path, [_row()])
        # a later row with an unknown extra column must not corrupt columns.
        rs.append_results(self.path, [_row(run_id="r2", surprise="x")])
        rows = rs.load_results(self.path)
        self.assertEqual(len(rows), 2)
        self.assertNotIn("surprise", rows[0])


class TestCompare(unittest.TestCase):
    def test_flags_drop(self):
        base = [_row(tflops=18.0)]
        cur = [_row(tflops=14.0)]
        regs = rs.compare(base, cur, threshold=0.05)
        self.assertEqual(len(regs), 1)
        self.assertLess(regs[0]["delta"], -0.05)

    def test_ignores_within_threshold(self):
        base = [_row(tflops=18.0)]
        cur = [_row(tflops=17.5)]  # ~2.8% drop
        self.assertEqual(rs.compare(base, cur, threshold=0.05), [])

    def test_ignores_improvement(self):
        base = [_row(tflops=18.0)]
        cur = [_row(tflops=20.0)]
        self.assertEqual(rs.compare(base, cur, threshold=0.05), [])

    def test_skips_new_kernel_with_no_baseline(self):
        base = [_row(kernel="k1")]
        cur = [_row(kernel="k2", tflops=1.0)]  # no baseline for k2
        self.assertEqual(rs.compare(base, cur, threshold=0.05), [])

    def test_key_is_arch_kernel_shape(self):
        # different shape => different key => not compared even if much slower.
        base = [_row(M=512, tflops=18.0)]
        cur = [_row(M=1024, tflops=1.0)]
        self.assertEqual(rs.compare(base, cur, threshold=0.05), [])
        # different arch => different key.
        base2 = [_row(arch="gfx950", tflops=18.0)]
        cur2 = [_row(arch="gfx1201", tflops=1.0)]
        self.assertEqual(rs.compare(base2, cur2, threshold=0.05), [])


class TestConsistency(unittest.TestCase):
    def test_correct_row_has_tiny_error(self):
        # tflops == flop/1e9/ms exactly.
        flop = 2.0 * 512 * 512 * 512
        ms = 1.0
        row = _row(M=512, N=512, K=512, ms=ms, tflops=flop / 1e9 / ms, flop=flop)
        err = rs.consistency_error(row)
        self.assertIsNotNone(err)
        self.assertLess(err, 1e-9)

    def test_corrupted_row_has_large_error(self):
        flop = 2.0 * 512 * 512 * 512
        # gbps accidentally written into the tflops column.
        row = _row(M=512, N=512, K=512, ms=1.0, tflops=49.6, flop=flop)
        err = rs.consistency_error(row)
        self.assertIsNotNone(err)
        self.assertGreater(err, 0.1)

    def test_falls_back_to_2mnk_without_flop_column(self):
        ms = 1.0
        tflops = 2.0 * 512 * 512 * 512 / 1e9 / ms
        row = _row(M=512, N=512, K=512, ms=ms, tflops=tflops)  # no flop col
        err = rs.consistency_error(row)
        self.assertIsNotNone(err)
        self.assertLess(err, 1e-9)

    def test_missing_fields_returns_none(self):
        self.assertIsNone(rs.consistency_error({"tflops": 10.0}))


class TestSplitLatestRun(unittest.TestCase):
    def test_picks_two_most_recent_runs(self):
        history = [
            _row(run_id="2026-06-22T00:00:00+00:00_aaa", tflops=18.0),
            _row(run_id="2026-06-23T00:00:00+00:00_bbb", tflops=18.0),
            _row(run_id="2026-06-24T00:00:00+00:00_ccc", tflops=14.0),
        ]
        baseline, current = rs.split_latest_run(history)
        self.assertEqual({r["run_id"] for r in current}, {"2026-06-24T00:00:00+00:00_ccc"})
        self.assertEqual({r["run_id"] for r in baseline}, {"2026-06-23T00:00:00+00:00_bbb"})

    def test_single_run_has_no_baseline(self):
        history = [_row(run_id="only")]
        baseline, current = rs.split_latest_run(history)
        self.assertEqual(baseline, [])
        self.assertEqual(len(current), 1)

    def test_end_to_end_split_then_compare_flags_regression(self):
        history = [
            _row(run_id="2026-06-23T00:00:00+00:00_old", tflops=18.0),
            _row(run_id="2026-06-24T00:00:00+00:00_new", tflops=14.0),
        ]
        baseline, current = rs.split_latest_run(history)
        regs = rs.compare(baseline, current, threshold=0.05)
        self.assertEqual(len(regs), 1)


class TestStampRows(unittest.TestCase):
    def test_shares_one_run_id_and_sets_source(self):
        rows = rs.stamp_rows([_row(), _row(kernel="k2")], rid="R", commit="c0ffee",
                             source="sweep")
        self.assertTrue(all(r["run_id"] == "R" for r in rows))
        self.assertTrue(all(r["commit"] == "c0ffee" for r in rows))
        self.assertTrue(all(r["source"] == "sweep" for r in rows))


if __name__ == "__main__":
    unittest.main()
