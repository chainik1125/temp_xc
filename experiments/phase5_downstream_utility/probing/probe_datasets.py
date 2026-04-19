"""SAEBench sparse-probing dataset loaders — independent reimplementation.

SAEBench's probing uses 8 datasets, each producing several binary
one-vs-rest classification tasks. We reproduce the same task layout
without depending on SAEBench internally.

Per SAEBench (`sae_bench/evals/sparse_probing`):

    - LabHC/bias_in_bios   → 3 class-sets of 5 professions each.
    - canrager/amazon_reviews_mcauley_1and5   → 5 product categories.
    - canrager/amazon_reviews_mcauley_1and5 (stars column) → sentiment.
    - fancyzhx/ag_news     → 4 topic classes.
    - Helsinki-NLP/europarl   → 5 language-ID binaries.
    - bigcode/the-stack-smol   → 5 programming-language binaries
      (used as substitute for the HF-deprecated github-code dataset).

Each loader returns a list of `ProbingTask`. Splits use SAEBench sizes
(train=4000, test=1000) with seed 42 by default. Class-balanced
negatives are sampled via `_balanced_binary_task`.
"""

from __future__ import annotations

import os
import random
from collections import Counter
from dataclasses import dataclass

import numpy as np

os.environ.setdefault("HF_HOME", "/workspace/hf_cache")
os.environ.setdefault("HF_DATASETS_TRUST_REMOTE_CODE", "1")


TRAIN_SIZE = 4000
TEST_SIZE = 1000
SEED = 42


@dataclass
class ProbingTask:
    dataset_key: str
    task_name: str
    train_texts: list[str]
    train_labels: np.ndarray
    test_texts: list[str]
    test_labels: np.ndarray


def _balanced_binary_task(
    texts: list[str],
    classes: list,
    positive,
    rng: random.Random,
    max_train: int = TRAIN_SIZE,
    max_test: int = TEST_SIZE,
):
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


def _load_bias_in_bios(rng: random.Random) -> list[ProbingTask]:
    from datasets import load_dataset
    print("Loading LabHC/bias_in_bios...")
    ds = load_dataset("LabHC/bias_in_bios", split="test", streaming=True)
    texts: list[str] = []
    professions: list[int] = []
    for i, row in enumerate(ds):
        if i >= 20_000:
            break
        t = row.get("hard_text")
        p = row.get("profession")
        if isinstance(t, str) and isinstance(p, int) and len(t) > 20:
            texts.append(t)
            professions.append(p)

    counts = Counter(professions).most_common()
    top = [p for p, _ in counts[:15]]
    print(f"  top-15 professions: {top}")

    tasks: list[ProbingTask] = []
    for set_idx, start in enumerate([0, 5, 10]):
        for prof in top[start:start + 5]:
            tr_t, tr_l, te_t, te_l = _balanced_binary_task(
                texts, professions, prof, rng
            )
            tasks.append(ProbingTask(
                dataset_key=f"bias_in_bios_set{set_idx + 1}",
                task_name=f"bias_in_bios_set{set_idx + 1}_prof{prof}",
                train_texts=tr_t, train_labels=tr_l,
                test_texts=te_t, test_labels=te_l,
            ))
    return tasks


def _load_ag_news(rng: random.Random) -> list[ProbingTask]:
    from datasets import load_dataset
    print("Loading fancyzhx/ag_news...")
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
    for cls in range(4):
        tr_t, tr_l, te_t, te_l = _balanced_binary_task(
            texts, labels, cls, rng
        )
        out.append(ProbingTask(
            dataset_key="ag_news",
            task_name=f"ag_news_{class_names[cls]}",
            train_texts=tr_t, train_labels=tr_l,
            test_texts=te_t, test_labels=te_l,
        ))
    return out


def _load_amazon_reviews(rng: random.Random) -> list[ProbingTask]:
    from datasets import load_dataset
    print("Loading canrager/amazon_reviews_mcauley_1and5 (categories)...")
    try:
        ds = load_dataset(
            "canrager/amazon_reviews_mcauley_1and5",
            split="train", streaming=True,
        )
    except Exception as e:
        print(f"  FAIL: {e}")
        return []
    texts: list[str] = []
    cats: list = []
    for i, row in enumerate(ds):
        if i >= 20_000:
            break
        t = row.get("text") or row.get("review_body")
        c = row.get("category") or row.get("main_category")
        # HF `canrager/amazon_reviews_mcauley_1and5` stores category as int
        if isinstance(t, str) and c is not None and len(t) > 20:
            texts.append(t)
            cats.append(c)
    if not texts:
        return []
    top = [c for c, _ in Counter(cats).most_common(5)]
    out: list[ProbingTask] = []
    for cat in top:
        tr_t, tr_l, te_t, te_l = _balanced_binary_task(
            texts, cats, cat, rng
        )
        # cat may be str or int depending on dataset version
        cat_slug = str(cat).replace(" ", "_")
        out.append(ProbingTask(
            dataset_key="amazon_reviews",
            task_name=f"amazon_reviews_cat{cat_slug}",
            train_texts=tr_t, train_labels=tr_l,
            test_texts=te_t, test_labels=te_l,
        ))
    return out


def _load_amazon_reviews_sentiment(rng: random.Random) -> list[ProbingTask]:
    from datasets import load_dataset
    print("Loading canrager/amazon_reviews_mcauley_1and5 (sentiment)...")
    try:
        ds = load_dataset(
            "canrager/amazon_reviews_mcauley_1and5",
            split="train", streaming=True,
        )
    except Exception as e:
        print(f"  FAIL: {e}")
        return []
    texts, stars = [], []
    for i, row in enumerate(ds):
        if i >= 20_000:
            break
        t = row.get("text") or row.get("review_body")
        # HF canrager/amazon_reviews_mcauley_1and5 uses a `rating` column
        s = row.get("rating")
        if s is None:
            s = row.get("stars") or row.get("star")
        if isinstance(t, str) and isinstance(s, (int, float)) and len(t) > 20:
            si = int(s)
            if si in (1, 5):
                texts.append(t)
                stars.append(si)
    if not stars:
        return []
    tr_t, tr_l, te_t, te_l = _balanced_binary_task(texts, stars, 5, rng)
    return [ProbingTask(
        dataset_key="amazon_reviews_sentiment",
        task_name="amazon_reviews_sentiment_5star",
        train_texts=tr_t, train_labels=tr_l,
        test_texts=te_t, test_labels=te_l,
    )]


def _load_europarl(rng: random.Random) -> list[ProbingTask]:
    from datasets import load_dataset
    print("Loading Helsinki-NLP/europarl (5 lang pairs)...")
    # HF europarl pairs are alphabetized: de-en not en-de, el-en not en-el, etc.
    configs = ["en-fr", "de-en", "en-es", "en-it", "en-nl"]
    target_langs = ["fr", "de", "es", "it", "nl"]
    lang_texts: dict[str, list[str]] = {"en": []}
    for cfg, lang in zip(configs, target_langs):
        lang_texts.setdefault(lang, [])
        try:
            ds = load_dataset(
                "Helsinki-NLP/europarl", cfg,
                split="train", streaming=True,
            )
        except Exception as e:
            print(f"  FAIL {cfg}: {e}")
            continue
        for i, row in enumerate(ds):
            if i >= 2500:
                break
            t = row.get("translation", {})
            if isinstance(t, dict):
                en = t.get("en")
                other = t.get(lang)
                if isinstance(en, str) and len(en) > 20:
                    lang_texts["en"].append(en)
                if isinstance(other, str) and len(other) > 20:
                    lang_texts[lang].append(other)

    out: list[ProbingTask] = []
    for target in target_langs:
        pos = lang_texts.get(target, [])
        neg: list[str] = []
        for other in lang_texts:
            if other != target:
                neg.extend(lang_texts[other])
        if not pos or not neg:
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


def _load_github_code(rng: random.Random) -> list[ProbingTask]:
    from datasets import load_dataset
    print("Loading bigcode/the-stack-smol (5 langs)...")
    langs = ["python", "java", "javascript", "cpp", "go"]
    per_lang: dict[str, list[str]] = {lg: [] for lg in langs}
    for lg in langs:
        try:
            ds = load_dataset(
                "bigcode/the-stack-smol",
                data_dir=f"data/{lg}",
                split="train", streaming=True,
            )
            for i, row in enumerate(ds):
                if i >= 3000:
                    break
                c = row.get("content")
                if isinstance(c, str) and 50 < len(c) < 4000:
                    per_lang[lg].append(c)
        except Exception as e:
            print(f"  FAIL {lg}: {e}")

    texts_flat: list[str] = []
    langs_flat: list[str] = []
    for lg in langs:
        texts_flat.extend(per_lang[lg])
        langs_flat.extend([lg] * len(per_lang[lg]))
    if not texts_flat:
        return []

    out: list[ProbingTask] = []
    for lg in langs:
        if sum(1 for x in langs_flat if x == lg) < 100:
            continue
        tr_t, tr_l, te_t, te_l = _balanced_binary_task(
            texts_flat, langs_flat, lg, rng
        )
        out.append(ProbingTask(
            dataset_key="github_code",
            task_name=f"github_code_{lg}",
            train_texts=tr_t, train_labels=tr_l,
            test_texts=te_t, test_labels=te_l,
        ))
    return out


def load_all_probing_tasks(seed: int = SEED) -> list[ProbingTask]:
    rng = random.Random(seed)
    tasks: list[ProbingTask] = []
    for loader in [
        _load_bias_in_bios,
        _load_ag_news,
        _load_amazon_reviews,
        _load_amazon_reviews_sentiment,
        _load_europarl,
        _load_github_code,
    ]:
        try:
            tasks.extend(loader(rng))
        except Exception as e:
            print(f"  {loader.__name__} FAIL: {e}")
    n_ds = len({t.dataset_key for t in tasks})
    print(f"Built {len(tasks)} binary probing tasks across {n_ds} datasets")
    return tasks
