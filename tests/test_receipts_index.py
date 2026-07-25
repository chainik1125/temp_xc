"""The claim→artifact receipts index must recompute clean (rebuttal
insurance — drift between artifacts and quoted claims breaks the suite,
not the rebuttal). See `experiments/explorations/task_hunt/RECEIPTS.md`."""

from experiments.explorations.task_hunt import receipts_check as rc


def test_all_receipts_recompute_to_quoted_values():
    rows, n_fail = rc.check(rc.build_receipts())
    failing = [r["id"] for r in rows if r["_verdict"] != "PASS"]
    assert n_fail == 0, f"receipts FAIL: {failing}"
