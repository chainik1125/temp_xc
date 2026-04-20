"""Cross-token probing datasets for Phase 5 sub-phase 5.4.

Two binary coreference / pronoun-resolution tasks where the answer
REQUIRES information from >= 2 token positions (the referent mention
and the pronoun). A last-token probe cannot solve these without the
LM having already resolved coreference into the final-token embedding.

Tasks:
    - WinoGrande binary: given a fill-in-the-blank sentence with two
      candidate completions (option1, option2), the model sees each
      completion and we probe whether the COMPLETED sentence is the
      preferred continuation. The binary label is which option was
      correct.
    - SuperGLUE WSC: span1 and span2 are noun phrases in the same
      sentence; the label is True if span2 refers to span1 (pronoun
      resolution) and False otherwise.

Both tasks are structurally multi-token-dependent by construction,
satisfying the sub-phase 5.4 brief: "requires a multi-token window by
construction."

Splits follow SAEBench sizes: train 4000 / test 1000 when available;
otherwise the entire HF test split.
"""

from __future__ import annotations

import os
import random
from collections import Counter
from dataclasses import dataclass

import numpy as np

os.environ.setdefault("HF_DATASETS_TRUST_REMOTE_CODE", "1")


@dataclass
class ProbingTask:
    dataset_key: str
    task_name: str
    train_texts: list[str]
    train_labels: np.ndarray
    test_texts: list[str]
    test_labels: np.ndarray


def _balance(texts, labels, rng, max_train=4000, max_test=1000):
    """Shuffle + 80/20 split from a pre-labelled list."""
    idx = list(range(len(texts)))
    rng.shuffle(idx)
    texts = [texts[i] for i in idx]
    labels = np.asarray([labels[i] for i in idx], dtype=np.int64)
    n_train = min(max_train, int(len(texts) * 0.8))
    return (
        texts[:n_train], labels[:n_train],
        texts[n_train:n_train + max_test], labels[n_train:n_train + max_test],
    )


def _load_winogrande(rng: random.Random) -> list[ProbingTask]:
    """WinoGrande: fill the blank with option1 or option2.

    We construct 2 prompts per example — one for each option — and label
    the correct completion 1, the incorrect 0. This makes the task
    binary: is THIS sentence the correct resolution? A last-token probe
    has to score the full sentence; both sentences end similarly so the
    answer can only come from the body of the sentence.
    """
    from datasets import load_dataset
    try:
        ds = load_dataset(
            "winogrande", "winogrande_xl",
            split="validation", streaming=False,
        )
    except Exception as e:
        print(f"  winogrande FAIL: {e}")
        return []
    texts: list[str] = []
    labels: list[int] = []
    for row in ds:
        sentence = row["sentence"]
        opt1 = row["option1"]
        opt2 = row["option2"]
        answer = row.get("answer")
        if answer not in ("1", "2"):
            continue
        # Fill the blank (underscore) with each option
        for i, opt in enumerate([opt1, opt2], start=1):
            filled = sentence.replace("_", opt)
            texts.append(filled)
            labels.append(1 if str(i) == answer else 0)
    if not texts:
        return []
    tr_t, tr_l, te_t, te_l = _balance(texts, labels, rng)
    return [ProbingTask(
        dataset_key="winogrande",
        task_name="winogrande_correct_completion",
        train_texts=tr_t, train_labels=tr_l,
        test_texts=te_t, test_labels=te_l,
    )]


def _load_super_glue_wsc(rng: random.Random) -> list[ProbingTask]:
    """SuperGLUE WSC: does span2 refer to span1?"""
    from datasets import load_dataset
    try:
        ds = load_dataset(
            "aps/super_glue", "wsc", split="train", streaming=False,
        )
    except Exception as e:
        print(f"  wsc FAIL: {e}")
        return []
    texts: list[str] = []
    labels: list[int] = []
    for row in ds:
        txt = row.get("text")
        lbl = row.get("label")
        if isinstance(txt, str) and lbl in (0, 1):
            texts.append(txt)
            labels.append(int(lbl))
    if not texts:
        return []
    tr_t, tr_l, te_t, te_l = _balance(texts, labels, rng)
    return [ProbingTask(
        dataset_key="wsc",
        task_name="wsc_coreference",
        train_texts=tr_t, train_labels=tr_l,
        test_texts=te_t, test_labels=te_l,
    )]


def load_all_crosstoken_tasks(seed: int = 42) -> list[ProbingTask]:
    rng = random.Random(seed)
    tasks: list[ProbingTask] = []
    for loader in [_load_winogrande, _load_super_glue_wsc]:
        try:
            tasks.extend(loader(rng))
        except Exception as e:
            print(f"  {loader.__name__} FAIL: {e}")
    print(f"Built {len(tasks)} cross-token probing tasks")
    return tasks
