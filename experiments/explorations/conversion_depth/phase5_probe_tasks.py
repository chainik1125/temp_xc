"""SAEBench+CT probing-task loaders (n=38) — self-contained port.

Ported byte-faithfully from
``origin/final:purified/src/temp_bench/data/nlp/probe_tasks.py`` for the
conversion-depth raw-activation probing phase (gemma-2-2b base-vs-it).
The original imported nothing from ``temp_bench.*`` — only stdlib +
numpy + lazy ``datasets`` — so the loaders and class lists below are
verbatim. Additions on top of the original:

- a module-level ``LOAD_ERRORS`` dict (one recording line per existing
  ``except`` block) so dataset-load failures surface in the summary;
- a per-loader SIGALRM timeout (600 s) in the two ``load_all_*``
  aggregators so a hung download is killed and recorded, not waited on;
- ``EXPECTED_TASKS`` (the locked 38-task composition) and a
  ``__main__`` that dumps every task's texts/labels to
  ``/workspace/conv_depth_caches/probe_tasks/<task_name>.npz`` plus a
  ``summary.json``.

Original module docstring follows.

----

SAEBench+CT probing-task loaders (n=38).

Locked task suite per ``decisions.md`` § 11 + ``docs/components/c3.md``.
Composition (8 datasets, 36 binary one-vs-rest tasks) reproduces the
upstream SAEBench class lists from
``sae_bench/sae_bench_utils/dataset_info.py::chosen_classes_per_dataset``,
plus 2 cross-token coreference tasks (WinoGrande + SuperGLUE WSC).

Sourced (in spirit) from
``origin/han-phase7-unification @ 94119bc0:experiments/phase5_downstream_utility/probing/``
(``probe_datasets.py`` + ``crosstoken_datasets.py``). Three SAEBench-
faithfulness fixes vs the wasteland 36-task loader:

1. **github-code**: switched from ``code_search_net`` (python/java/
   javascript/go) to SAEBench's ``codeparrot/github-code`` with the 5
   SAEBench languages (C, Python, HTML, Java, PHP). Loader requires
   ``trust_remote_code=True`` and ``datasets<4`` (pinned in
   ``pyproject.toml``). The HF web viewer for this dataset is disabled
   because it uses a Python loading script, NOT because the data is
   gated. The streaming filter ``languages=[...]`` does NOT actually
   filter — we ``continue`` on language mismatch after iter.
2. **amazon_sentiment**: emits BOTH 1.0-vs-rest AND 5.0-vs-rest
   binaries (wasteland only emitted 5.0).
3. **amazon_categories**: hardcoded classes ``["1","2","3","5","6"]``
   with a deterministic non-streaming pull (wasteland streaming-top-5
   missed cat6).

In addition to those 3 documented fixes, all class lists across all
datasets are now **hardcoded from SAEBench** (rather than wasteland's
"top-N most frequent in stream") to guarantee deterministic
reproducibility across runs.

Public API
----------

- :func:`load_all_probing_tasks(seed=42)` → 36 SAEBench tasks
- :func:`load_all_crosstoken_tasks(seed=42)` → 2 CT tasks
- :func:`load_all_saebench_ct_tasks(seed=42)` → all 38 tasks (the
  paper's task suite)
"""

from __future__ import annotations

import os
import random
import signal
from collections import Counter
from dataclasses import dataclass

import numpy as np

os.environ.setdefault("HF_HOME", "/workspace/hf")
os.environ.setdefault("HF_DATASETS_TRUST_REMOTE_CODE", "1")


TRAIN_SIZE = 4000
TEST_SIZE = 1000
SEED = 42

# Per-loader wall-clock budget (added for the port; the original had none).
LOADER_TIMEOUT_S = 600

# dataset-family (loader) name -> error string, recorded by the except
# blocks so the __main__ summary can attribute missing tasks.
LOAD_ERRORS: dict[str, str] = {}

# ── SAEBench class lists (hardcoded from chosen_classes_per_dataset) ──

BIAS_IN_BIOS_CLASSES = [
    [0, 1, 2, 6, 9],          # set1
    [11, 13, 14, 18, 19],     # set2
    [20, 21, 22, 25, 26],     # set3
]
AMAZON_CATEGORY_CLASSES = ["1", "2", "3", "5", "6"]
AMAZON_SENTIMENT_CLASSES = [1, 5]   # both 1-star and 5-star (faithfulness fix)
AG_NEWS_CLASSES = [0, 1, 2, 3]
EUROPARL_LANGS = ["en", "fr", "de", "es", "nl"]
EUROPARL_PAIRS = ["en-fr", "de-en", "en-es", "en-it", "en-nl"]
EUROPARL_PAIR_TO_TARGET = {
    "en-fr": "fr", "de-en": "de", "en-es": "es", "en-it": "it", "en-nl": "nl",
}
GITHUB_CODE_LANGS = ["C", "Python", "HTML", "Java", "PHP"]


@dataclass
class ProbingTask:
    """One binary classification probing task.

    Attributes:
        dataset_key: SAEBench dataset family (e.g. ``"bias_in_bios_set1"``)
        task_name: globally unique task ID (e.g. ``"bias_in_bios_set1_prof0"``)
        train_texts, test_texts: raw text strings
        train_labels, test_labels: ``(N,)`` int64 0/1 arrays
    """

    dataset_key: str
    task_name: str
    train_texts: list[str]
    train_labels: np.ndarray
    test_texts: list[str]
    test_labels: np.ndarray


# ── Internal: balanced binary task helper ──────────────────────────────────


def _balanced_binary_task(
    texts: list[str],
    classes: list,
    positive,
    rng: random.Random,
    max_train: int = TRAIN_SIZE,
    max_test: int = TEST_SIZE,
) -> tuple[list[str], np.ndarray, list[str], np.ndarray]:
    """Construct a positive-vs-rest binary split with class-balanced negatives.

    Same algorithm as the wasteland helper:
      - filter to positives + sample equal number of negatives
      - shuffle deterministically with ``rng``
      - 80/20 split on the smaller of (max_train, len * 0.8)
    """
    pos_idx = [i for i, c in enumerate(classes) if c == positive]
    neg_idx = [i for i, c in enumerate(classes) if c != positive]
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)
    n = min(len(pos_idx), len(neg_idx), (max_train + max_test) // 2)
    pos_idx = pos_idx[:n]
    neg_idx = neg_idx[:n]
    all_idx = pos_idx + neg_idx
    rng.shuffle(all_idx)
    labels = np.asarray(
        [1 if classes[i] == positive else 0 for i in all_idx], dtype=np.int64
    )
    chosen_texts = [texts[i] for i in all_idx]
    n_train = min(max_train, int(len(chosen_texts) * 0.8))
    return (
        chosen_texts[:n_train],
        labels[:n_train],
        chosen_texts[n_train:n_train + max_test],
        labels[n_train:n_train + max_test],
    )


def _balance_prelabeled(
    texts: list[str],
    labels: list[int],
    rng: random.Random,
    max_train: int = TRAIN_SIZE,
    max_test: int = TEST_SIZE,
) -> tuple[list[str], np.ndarray, list[str], np.ndarray]:
    """Shuffle + 80/20 split a pre-labelled list (used by crosstoken tasks)."""
    idx = list(range(len(texts)))
    rng.shuffle(idx)
    texts = [texts[i] for i in idx]
    labels_arr = np.asarray([labels[i] for i in idx], dtype=np.int64)
    n_train = min(max_train, int(len(texts) * 0.8))
    return (
        texts[:n_train], labels_arr[:n_train],
        texts[n_train:n_train + max_test], labels_arr[n_train:n_train + max_test],
    )


# ── SAEBench-faithful loaders (8 datasets, 36 tasks) ──────────────────────


def _load_bias_in_bios(rng: random.Random) -> list[ProbingTask]:
    """3 sets × 5 hardcoded SAEBench professions = 15 tasks."""
    from datasets import load_dataset
    print("[probe_tasks] LabHC/bias_in_bios...")
    ds = load_dataset("LabHC/bias_in_bios", split="test", streaming=True)

    needed = set()
    for s in BIAS_IN_BIOS_CLASSES:
        needed.update(s)

    texts: list[str] = []
    professions: list[int] = []
    for i, row in enumerate(ds):
        if i >= 30_000:
            break
        t = row.get("hard_text")
        p = row.get("profession")
        if isinstance(t, str) and isinstance(p, int) and p in needed and len(t) > 20:
            texts.append(t)
            professions.append(p)

    counts = Counter(professions)
    print(f"  loaded {len(texts)} bios across {len(counts)} target professions: {dict(counts)}")

    tasks: list[ProbingTask] = []
    for set_idx, profs in enumerate(BIAS_IN_BIOS_CLASSES, start=1):
        for prof in profs:
            tr_t, tr_l, te_t, te_l = _balanced_binary_task(
                texts, professions, prof, rng,
            )
            tasks.append(ProbingTask(
                dataset_key=f"bias_in_bios_set{set_idx}",
                task_name=f"bias_in_bios_set{set_idx}_prof{prof}",
                train_texts=tr_t, train_labels=tr_l,
                test_texts=te_t, test_labels=te_l,
            ))
    return tasks


def _load_ag_news(rng: random.Random) -> list[ProbingTask]:
    """4 hardcoded SAEBench classes = 4 tasks."""
    from datasets import load_dataset
    print("[probe_tasks] fancyzhx/ag_news...")
    ds = load_dataset("fancyzhx/ag_news", split="test", streaming=True)
    texts: list[str] = []
    labels: list[int] = []
    for i, row in enumerate(ds):
        if i >= 10_000:
            break
        t = row.get("text")
        lb = row.get("label")
        if isinstance(t, str) and isinstance(lb, int) and len(t) > 20:
            texts.append(t)
            labels.append(lb)
    class_names = {0: "world", 1: "sports", 2: "business", 3: "scitech"}
    out: list[ProbingTask] = []
    for cls in AG_NEWS_CLASSES:
        tr_t, tr_l, te_t, te_l = _balanced_binary_task(texts, labels, cls, rng)
        out.append(ProbingTask(
            dataset_key="ag_news",
            task_name=f"ag_news_{class_names[cls]}",
            train_texts=tr_t, train_labels=tr_l,
            test_texts=te_t, test_labels=te_l,
        ))
    return out


def _load_amazon_categories(rng: random.Random) -> list[ProbingTask]:
    """5 hardcoded SAEBench categories = 5 tasks (faithfulness fix #3).

    Non-streaming pull with shuffle so cat6 is reached. Wasteland
    streaming-top-5 missed cat6; my first attempt at non-streaming hit
    cat6 only at the tail of the split (1 row in first 80k). Shuffle
    fixes ordering bias deterministically.
    """
    from datasets import load_dataset
    print("[probe_tasks] canrager/amazon_reviews_mcauley_1and5 (categories)...")
    try:
        ds = load_dataset(
            "canrager/amazon_reviews_mcauley_1and5",
            split="train", streaming=False,
        ).shuffle(seed=42)
    except Exception as e:
        print(f"  FAIL: {e}")
        LOAD_ERRORS["amazon_categories"] = f"{type(e).__name__}: {e}"
        return []

    needed = set(AMAZON_CATEGORY_CLASSES)
    PER_CAT_TARGET = 5000   # need ≥ TRAIN+TEST = 5000 to hit max budget
    texts: list[str] = []
    cats: list[str] = []
    seen_per_cat: Counter = Counter()
    for row in ds:
        if all(seen_per_cat[c] >= PER_CAT_TARGET for c in AMAZON_CATEGORY_CLASSES):
            break
        t = row.get("text") or row.get("review_body")
        c = row.get("category")
        if c is None:
            c = row.get("main_category")
        # Normalize to str — categories are stored as either str or int
        c_str = str(c) if c is not None else None
        if (
            isinstance(t, str) and c_str in needed and len(t) > 20
            and seen_per_cat[c_str] < PER_CAT_TARGET
        ):
            texts.append(t)
            cats.append(c_str)
            seen_per_cat[c_str] += 1

    counts = Counter(cats)
    print(f"  loaded {len(texts)} reviews across categories: {dict(counts)}")

    out: list[ProbingTask] = []
    for cat in AMAZON_CATEGORY_CLASSES:
        if counts.get(cat, 0) < 200:
            print(f"  SKIP cat{cat}: only {counts.get(cat, 0)} samples")
            continue
        tr_t, tr_l, te_t, te_l = _balanced_binary_task(texts, cats, cat, rng)
        out.append(ProbingTask(
            dataset_key="amazon_reviews",
            task_name=f"amazon_reviews_cat{cat}",
            train_texts=tr_t, train_labels=tr_l,
            test_texts=te_t, test_labels=te_l,
        ))
    return out


def _load_amazon_sentiment(rng: random.Random) -> list[ProbingTask]:
    """Both 1-star-vs-5-star AND 5-star-vs-1-star binaries = 2 tasks
    (faithfulness fix #2). Wasteland only emitted 5-star.

    Note: with only 2 classes, the 1-star binary (positive=1) is the
    label-flip of the 5-star binary (positive=5). They share the SAME
    text pool but the labels are inverted. Reported as 2 separate
    tasks for SAEBench parity.
    """
    from datasets import load_dataset
    print("[probe_tasks] canrager/amazon_reviews_mcauley_1and5 (sentiment)...")
    try:
        ds = load_dataset(
            "canrager/amazon_reviews_mcauley_1and5",
            split="train", streaming=True,
        )
    except Exception as e:
        print(f"  FAIL: {e}")
        LOAD_ERRORS["amazon_sentiment"] = f"{type(e).__name__}: {e}"
        return []

    texts: list[str] = []
    stars: list[int] = []
    for i, row in enumerate(ds):
        if i >= 20_000:
            break
        t = row.get("text") or row.get("review_body")
        s = row.get("rating")
        if s is None:
            s = row.get("stars") or row.get("star")
        if isinstance(t, str) and isinstance(s, (int, float)) and len(t) > 20:
            si = int(s)
            if si in AMAZON_SENTIMENT_CLASSES:
                texts.append(t)
                stars.append(si)

    if not stars:
        return []

    out: list[ProbingTask] = []
    for star in AMAZON_SENTIMENT_CLASSES:
        tr_t, tr_l, te_t, te_l = _balanced_binary_task(texts, stars, star, rng)
        out.append(ProbingTask(
            dataset_key="amazon_reviews_sentiment",
            task_name=f"amazon_reviews_sentiment_{star}star",
            train_texts=tr_t, train_labels=tr_l,
            test_texts=te_t, test_labels=te_l,
        ))
    return out


def _load_europarl(rng: random.Random) -> list[ProbingTask]:
    """5 SAEBench language ID binaries (en, fr, de, es, nl)."""
    from datasets import load_dataset
    print("[probe_tasks] Helsinki-NLP/europarl (5 language IDs)...")
    # The HF europarl pairs are alphabetised: de-en not en-de.
    lang_texts: dict[str, list[str]] = {lg: [] for lg in EUROPARL_LANGS}
    for cfg in EUROPARL_PAIRS:
        target = EUROPARL_PAIR_TO_TARGET[cfg]
        try:
            ds = load_dataset(
                "Helsinki-NLP/europarl", cfg,
                split="train", streaming=True,
            )
        except Exception as e:
            print(f"  FAIL {cfg}: {e}")
            LOAD_ERRORS[f"europarl_{cfg}"] = f"{type(e).__name__}: {e}"
            continue
        for i, row in enumerate(ds):
            if i >= 2500:
                break
            t = row.get("translation", {})
            if isinstance(t, dict):
                en = t.get("en")
                other = t.get(target)
                if isinstance(en, str) and len(en) > 20:
                    lang_texts["en"].append(en)
                if (
                    isinstance(other, str) and len(other) > 20
                    and target in lang_texts
                ):
                    lang_texts[target].append(other)

    out: list[ProbingTask] = []
    for target in EUROPARL_LANGS:
        pos = lang_texts.get(target, [])
        neg: list[str] = []
        for other_lang, other_texts in lang_texts.items():
            if other_lang != target:
                neg.extend(other_texts)
        if not pos or not neg:
            print(f"  SKIP europarl_{target}: pos={len(pos)} neg={len(neg)}")
            continue
        rng.shuffle(pos)
        rng.shuffle(neg)
        n = min(len(pos), len(neg), (TRAIN_SIZE + TEST_SIZE) // 2)
        all_texts = pos[:n] + neg[:n]
        labels = np.asarray([1] * n + [0] * n, dtype=np.int64)
        order = rng.sample(range(len(all_texts)), len(all_texts))
        all_texts = [all_texts[i] for i in order]
        labels = labels[order]
        n_train = min(TRAIN_SIZE, int(len(all_texts) * 0.8))
        out.append(ProbingTask(
            dataset_key="europarl",
            task_name=f"europarl_{target}",
            train_texts=all_texts[:n_train],
            train_labels=labels[:n_train],
            test_texts=all_texts[n_train:n_train + TEST_SIZE],
            test_labels=labels[n_train:n_train + TEST_SIZE],
        ))
    return out


def _load_github_code(
    rng: random.Random,
    *,
    target_per_lang: int = 3000,
    max_scan: int = 200_000,
) -> list[ProbingTask]:
    """5 SAEBench programming languages (C, Python, HTML, Java, PHP)
    — faithfulness fix #1.

    Switched from wasteland's ``code_search_net`` (4 langs) to SAEBench's
    ``codeparrot/github-code``. Critical gotchas:

    - The ``languages=[...]`` constructor arg does NOT actually filter
      the stream. We MUST ``continue`` on mismatch after iter. Tested
      2026-05-03: out of 20 streamed samples with ``languages=['C']`` we
      got 9 JS, 6 C, 5 other.
    - Loader is a Python script — needs ``trust_remote_code=True``
      (set globally via ``HF_DATASETS_TRUST_REMOTE_CODE=1`` env var
      at module top). Pinned ``datasets<4`` in pyproject because v4+
      removed the trust_remote_code switch.
    - Streaming distribution observed 2026-05-03: in 100 samples, Python
      = 3 (lowest of our 5). To get 3000 Python samples we need to
      scan ~100k rows. ``max_scan=200_000`` is the safety cap.
    """
    from datasets import load_dataset
    print(f"[probe_tasks] codeparrot/github-code ({len(GITHUB_CODE_LANGS)} langs)...")

    per_lang: dict[str, list[str]] = {lg: [] for lg in GITHUB_CODE_LANGS}
    try:
        ds = load_dataset(
            "codeparrot/github-code",
            streaming=True,
            trust_remote_code=True,
            split="train",
        )
    except Exception as e:
        print(f"  github-code load FAIL: {e}")
        LOAD_ERRORS["github_code"] = f"{type(e).__name__}: {e}"
        return []

    needed = set(GITHUB_CODE_LANGS)
    n_seen = 0
    n_kept = 0
    for i, row in enumerate(ds):
        if i >= max_scan:
            break
        n_seen += 1
        lang = row.get("language")
        if lang not in needed:
            continue
        if len(per_lang[lang]) >= target_per_lang:
            # All done for this lang
            if all(len(per_lang[lg]) >= target_per_lang for lg in GITHUB_CODE_LANGS):
                break
            continue
        code = row.get("code")
        if isinstance(code, str) and 50 < len(code) < 4000:
            per_lang[lang].append(code)
            n_kept += 1

    print(f"  scanned {n_seen} rows, kept {n_kept} samples")
    print(f"  per-lang counts: { {lg: len(v) for lg, v in per_lang.items()} }")

    texts_flat: list[str] = []
    langs_flat: list[str] = []
    for lg in GITHUB_CODE_LANGS:
        texts_flat.extend(per_lang[lg])
        langs_flat.extend([lg] * len(per_lang[lg]))

    if not texts_flat:
        print("  github-code: no samples — tasks skipped")
        return []

    out: list[ProbingTask] = []
    for lg in GITHUB_CODE_LANGS:
        if sum(1 for x in langs_flat if x == lg) < 100:
            print(f"  SKIP github_code_{lg}: too few samples")
            continue
        tr_t, tr_l, te_t, te_l = _balanced_binary_task(
            texts_flat, langs_flat, lg, rng,
        )
        out.append(ProbingTask(
            dataset_key="github_code",
            task_name=f"github_code_{lg}",
            train_texts=tr_t, train_labels=tr_l,
            test_texts=te_t, test_labels=te_l,
        ))
    return out


# ── Cross-token coreference loaders (2 tasks) ──────────────────────────────


def _load_winogrande(rng: random.Random) -> list[ProbingTask]:
    """WinoGrande: fill the blank with option1 or option2; binary label =
    is THIS sentence the correct resolution? Cross-token by construction —
    the answer requires resolving the pronoun against the body.
    """
    from datasets import load_dataset
    print("[probe_tasks] winogrande (validation)...")
    try:
        ds = load_dataset(
            "winogrande", "winogrande_xl",
            split="validation", streaming=False,
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"  winogrande FAIL: {e}")
        LOAD_ERRORS["winogrande"] = f"{type(e).__name__}: {e}"
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
        for i, opt in enumerate([opt1, opt2], start=1):
            filled = sentence.replace("_", opt)
            texts.append(filled)
            labels.append(1 if str(i) == answer else 0)
    if not texts:
        return []
    tr_t, tr_l, te_t, te_l = _balance_prelabeled(texts, labels, rng)
    return [ProbingTask(
        dataset_key="winogrande",
        task_name="winogrande_correct_completion",
        train_texts=tr_t, train_labels=tr_l,
        test_texts=te_t, test_labels=te_l,
    )]


def _load_super_glue_wsc(rng: random.Random) -> list[ProbingTask]:
    """SuperGLUE WSC: does span2 refer to span1? Binary coref."""
    from datasets import load_dataset
    print("[probe_tasks] aps/super_glue (wsc)...")
    try:
        ds = load_dataset(
            "aps/super_glue", "wsc", split="train", streaming=False,
        )
    except Exception as e:
        print(f"  wsc FAIL: {e}")
        LOAD_ERRORS["wsc"] = f"{type(e).__name__}: {e}"
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
    tr_t, tr_l, te_t, te_l = _balance_prelabeled(texts, labels, rng)
    return [ProbingTask(
        dataset_key="wsc",
        task_name="wsc_coreference",
        train_texts=tr_t, train_labels=tr_l,
        test_texts=te_t, test_labels=te_l,
    )]


# ── Per-loader timeout wrapper (added for the port) ────────────────────────


def _call_with_timeout(loader, rng: random.Random, timeout_s: int = LOADER_TIMEOUT_S):
    """Run ``loader(rng)`` under a SIGALRM deadline; raise TimeoutError on expiry.

    Main-thread only. Converts a hung dataset download into an exception
    the aggregators' existing try/except records.
    """

    def _on_alarm(signum, frame):
        raise TimeoutError(f"{loader.__name__} exceeded {timeout_s}s")

    old_handler = signal.signal(signal.SIGALRM, _on_alarm)
    signal.alarm(timeout_s)
    try:
        return loader(rng)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


# ── Public API ────────────────────────────────────────────────────────────


def load_all_probing_tasks(seed: int = SEED) -> list[ProbingTask]:
    """The 36 SAEBench binary one-vs-rest probing tasks."""
    rng = random.Random(seed)
    tasks: list[ProbingTask] = []
    for loader in (
        _load_bias_in_bios,
        _load_ag_news,
        _load_amazon_categories,
        _load_amazon_sentiment,
        _load_europarl,
        _load_github_code,
    ):
        try:
            tasks.extend(_call_with_timeout(loader, rng))
        except Exception as e:
            print(f"  {loader.__name__} FAIL: {e}")
            LOAD_ERRORS.setdefault(
                loader.__name__.removeprefix("_load_"),
                f"{type(e).__name__}: {e}",
            )
    n_ds = len({t.dataset_key for t in tasks})
    print(f"[probe_tasks] built {len(tasks)} SAEBench tasks across {n_ds} datasets")
    return tasks


def load_all_crosstoken_tasks(seed: int = SEED) -> list[ProbingTask]:
    """The 2 cross-token coreference tasks (WinoGrande + WSC)."""
    rng = random.Random(seed)
    tasks: list[ProbingTask] = []
    for loader in (_load_winogrande, _load_super_glue_wsc):
        try:
            tasks.extend(_call_with_timeout(loader, rng))
        except Exception as e:
            print(f"  {loader.__name__} FAIL: {e}")
            LOAD_ERRORS.setdefault(
                loader.__name__.removeprefix("_load_"),
                f"{type(e).__name__}: {e}",
            )
    print(f"[probe_tasks] built {len(tasks)} cross-token tasks")
    return tasks


def load_all_saebench_ct_tasks(seed: int = SEED) -> list[ProbingTask]:
    """The full 38-task SAEBench+CT suite (n=38).

    Locked in ``decisions.md`` § 11 + ``docs/components/c3.md``.
    Composition (all-or-nothing — if any loader fails, count drops):

      - bias_in_bios (3 sets × 5 profs)        = 15
      - ag_news (4 classes)                    =  4
      - amazon_categories (5 hardcoded cats)   =  5
      - amazon_sentiment (1-star + 5-star)     =  2
      - europarl (5 lang IDs)                  =  5
      - github_code (5 langs codeparrot)       =  5
      - winogrande_correct_completion (CT)     =  1
      - wsc_coreference (CT)                   =  1
                                              ───
                                                 38
    """
    return load_all_probing_tasks(seed=seed) + load_all_crosstoken_tasks(seed=seed)


# ── __main__: dump texts + labels to /workspace/conv_depth_caches ──────────

# The locked 38-task composition: task_name -> dataset family (the
# LOAD_ERRORS key its loader records under).
EXPECTED_TASKS: dict[str, str] = {
    **{
        f"bias_in_bios_set{si}_prof{p}": "bias_in_bios"
        for si, profs in enumerate(BIAS_IN_BIOS_CLASSES, start=1)
        for p in profs
    },
    **{f"ag_news_{nm}": "ag_news" for nm in ("world", "sports", "business", "scitech")},
    **{f"amazon_reviews_cat{c}": "amazon_categories" for c in AMAZON_CATEGORY_CLASSES},
    **{
        f"amazon_reviews_sentiment_{s}star": "amazon_sentiment"
        for s in AMAZON_SENTIMENT_CLASSES
    },
    **{f"europarl_{lg}": "europarl" for lg in EUROPARL_LANGS},
    **{f"github_code_{lg}": "github_code" for lg in GITHUB_CODE_LANGS},
    "winogrande_correct_completion": "winogrande",
    "wsc_coreference": "wsc",
}

OUT_DIR = "/workspace/conv_depth_caches/probe_tasks"


def _main() -> None:
    import json

    os.makedirs(OUT_DIR, exist_ok=True)
    tasks = load_all_saebench_ct_tasks(seed=42)
    by_name = {t.task_name: t for t in tasks}

    summary: dict[str, dict] = {}
    for task_name, family in EXPECTED_TASKS.items():
        t = by_name.get(task_name)
        if t is None:
            err = LOAD_ERRORS.get(family)
            if err is None:
                # e.g. europarl records per-config errors
                err = "; ".join(
                    f"{k}: {v}" for k, v in LOAD_ERRORS.items()
                    if k.startswith(family) or family in k
                ) or "task not produced (loader returned no task, no error recorded)"
            summary[task_name] = {"ok": False, "error": err}
            continue
        np.savez(
            os.path.join(OUT_DIR, f"{task_name}.npz"),
            texts_train=np.asarray(t.train_texts, dtype=np.str_),
            texts_test=np.asarray(t.test_texts, dtype=np.str_),
            y_train=t.train_labels.astype(np.int64),
            y_test=t.test_labels.astype(np.int64),
        )
        summary[task_name] = {
            "n_train": int(len(t.train_texts)),
            "n_test": int(len(t.test_texts)),
            "pos_frac_train": float(t.train_labels.mean()),
            "pos_frac_test": float(t.test_labels.mean()),
            "ok": True,
        }

    unexpected = sorted(set(by_name) - set(EXPECTED_TASKS))
    if unexpected:
        print(f"[probe_tasks] WARNING: unexpected task names: {unexpected}")

    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    n_ok = sum(1 for v in summary.values() if v.get("ok"))
    print(f"[probe_tasks] wrote {n_ok}/{len(summary)} tasks to {OUT_DIR}")


if __name__ == "__main__":
    _main()
