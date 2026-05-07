"""Tests for ``temp_bench.data.nlp.probe_tasks`` — pure helpers.

The HF-loading paths are integration-tested via ``--smoke`` runs;
this file covers the deterministic helpers + class-list constants.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from temp_bench.data.nlp.probe_tasks import (
    AG_NEWS_CLASSES,
    AMAZON_CATEGORY_CLASSES,
    AMAZON_SENTIMENT_CLASSES,
    BIAS_IN_BIOS_CLASSES,
    EUROPARL_LANGS,
    GITHUB_CODE_LANGS,
    SEED,
    TEST_SIZE,
    TRAIN_SIZE,
    ProbingTask,
    _balance_prelabeled,
    _balanced_binary_task,
)


# ── SAEBench class-list parity (decisions.md § 11) ────────────────────────


def test_saebench_class_lists_match_spec():
    """Hardcoded SAEBench class lists per docs/components/c3.md."""
    # bias_in_bios: 3 sets × 5 profs = 15 tasks
    assert len(BIAS_IN_BIOS_CLASSES) == 3
    assert all(len(s) == 5 for s in BIAS_IN_BIOS_CLASSES)
    assert BIAS_IN_BIOS_CLASSES[0] == [0, 1, 2, 6, 9]
    assert BIAS_IN_BIOS_CLASSES[1] == [11, 13, 14, 18, 19]
    assert BIAS_IN_BIOS_CLASSES[2] == [20, 21, 22, 25, 26]

    # ag_news: 4 classes
    assert AG_NEWS_CLASSES == [0, 1, 2, 3]

    # amazon_categories: 5 hardcoded (faithfulness fix #3)
    assert AMAZON_CATEGORY_CLASSES == ["1", "2", "3", "5", "6"]
    assert "4" not in AMAZON_CATEGORY_CLASSES, "Cat 4 is intentionally absent"
    assert "6" in AMAZON_CATEGORY_CLASSES, (
        "Cat 6 must be present (faithfulness fix vs wasteland)"
    )

    # amazon_sentiment: BOTH 1 and 5 (faithfulness fix #2)
    assert AMAZON_SENTIMENT_CLASSES == [1, 5]

    # europarl: 5 SAEBench language IDs
    assert EUROPARL_LANGS == ["en", "fr", "de", "es", "nl"]

    # github_code: 5 SAEBench languages (faithfulness fix #1)
    assert GITHUB_CODE_LANGS == ["C", "Python", "HTML", "Java", "PHP"]


def test_saebench_total_tasks_is_38():
    """SAEBench+CT (n=38) per decisions.md § 11."""
    bib = sum(len(s) for s in BIAS_IN_BIOS_CLASSES)
    saebench = (
        bib
        + len(AG_NEWS_CLASSES)
        + len(AMAZON_CATEGORY_CLASSES)
        + len(AMAZON_SENTIMENT_CLASSES)
        + len(EUROPARL_LANGS)
        + len(GITHUB_CODE_LANGS)
    )
    assert saebench == 36, f"SAEBench=36 expected; got {saebench}"
    # +2 cross-token (winogrande + wsc)
    total = saebench + 2
    assert total == 38, f"SAEBench+CT n=38 expected; got {total}"


# ── _balanced_binary_task contract ────────────────────────────────────────


def test_balanced_binary_task_50_50_split():
    """Output labels should be ~50/50 positive/negative."""
    rng = random.Random(0)
    classes = [0] * 1000 + [1] * 1000 + [2] * 1000
    texts = [f"text_{c}_{i}" for i, c in enumerate(classes)]
    tr_t, tr_l, te_t, te_l = _balanced_binary_task(
        texts, classes, positive=1, rng=rng,
    )
    # ~50/50 split
    pos_frac_tr = tr_l.mean()
    pos_frac_te = te_l.mean()
    assert 0.4 < pos_frac_tr < 0.6, f"train pos_frac out of range: {pos_frac_tr}"
    assert 0.3 < pos_frac_te < 0.7, f"test pos_frac out of range: {pos_frac_te}"
    # Train + test sizes obey caps
    assert len(tr_t) <= TRAIN_SIZE
    assert len(te_t) <= TEST_SIZE
    # Lengths agree between texts and labels
    assert len(tr_t) == len(tr_l)
    assert len(te_t) == len(te_l)


def test_balanced_binary_task_seeded():
    """Same seed → same split; different seed → different split."""
    classes = [0, 0, 1, 1, 2, 2] * 100
    texts = [f"t{i}" for i in range(len(classes))]
    rng_a = random.Random(42)
    rng_b = random.Random(42)
    rng_c = random.Random(7)
    a = _balanced_binary_task(texts, classes, 1, rng_a)
    b = _balanced_binary_task(texts, classes, 1, rng_b)
    c = _balanced_binary_task(texts, classes, 1, rng_c)
    assert a[0] == b[0], "Same seed must produce identical text order"
    assert a[0] != c[0], "Different seed must produce different text order"


def test_balanced_binary_task_no_pos():
    """If positive class is absent, n=0 and outputs are empty."""
    rng = random.Random(0)
    classes = [0] * 100
    texts = [f"t{i}" for i in range(100)]
    tr_t, tr_l, te_t, te_l = _balanced_binary_task(texts, classes, 1, rng)
    assert len(tr_t) == 0
    assert len(te_t) == 0


# ── _balance_prelabeled (crosstoken helper) ───────────────────────────────


def test_balance_prelabeled_seeded():
    rng_a = random.Random(0)
    rng_b = random.Random(0)
    texts = [f"t{i}" for i in range(50)]
    labels = [i % 2 for i in range(50)]
    a = _balance_prelabeled(texts, labels, rng_a)
    b = _balance_prelabeled(texts, labels, rng_b)
    assert a[0] == b[0]
    assert (a[1] == b[1]).all()


def test_balance_prelabeled_split_obeys_caps():
    rng = random.Random(0)
    texts = [f"t{i}" for i in range(100)]
    labels = [i % 2 for i in range(100)]
    tr_t, tr_l, te_t, te_l = _balance_prelabeled(texts, labels, rng,
                                                  max_train=10, max_test=5)
    assert len(tr_t) <= 10
    assert len(te_t) <= 5


# ── ProbingTask dataclass ────────────────────────────────────────────────


def test_probing_task_dataclass():
    pt = ProbingTask(
        dataset_key="test",
        task_name="test_task",
        train_texts=["a", "b"],
        train_labels=np.asarray([0, 1], dtype=np.int64),
        test_texts=["c"],
        test_labels=np.asarray([0], dtype=np.int64),
    )
    assert pt.dataset_key == "test"
    assert pt.task_name == "test_task"
    assert pt.train_labels.dtype == np.int64
