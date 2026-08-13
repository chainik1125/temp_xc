"""Cache concept-dataset activations for sparse probing (one-off).

Three datasets (SAEBench-adjacent, scaled): ag_news (4-class topic),
amazon_polarity (2-class sentiment), dbpedia_14 (14-class ontology).
For each: 2400 train + 800 test sequences, first 64 tokens, Gemma-2-2B
layer-12 resid-post activations. Writes /vol/concepts/concept_{name}.pt
with acts_{train,test} (n, 64, d) bf16 and y_{train,test}.
"""

from __future__ import annotations

import pathlib

import torch

DATASETS = {
    "ag_news": ("fancyzhx/ag_news", "text", "label"),
    "amazon_polarity": ("fancyzhx/amazon_polarity", "content", "label"),
    "dbpedia": ("fancyzhx/dbpedia_14", "content", "label"),
}


def run(out_dir: str = "/vol", n_train: int = 2400, n_test: int = 800,
        ctx: int = 64, model_name: str = "google/gemma-2-2b",
        layer: int = 12) -> dict:
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, output_hidden_states=True
    ).to(dev).eval()
    out = pathlib.Path(out_dir) / "concepts"
    out.mkdir(parents=True, exist_ok=True)

    summary = {}
    for name, (hf_name, text_col, lab_col) in DATASETS.items():
        # NB: several HF classification sets are sorted by label — shuffle
        # the stream or the head is single-class.
        ds = load_dataset(hf_name, split="train", streaming=True).shuffle(
            seed=0, buffer_size=100_000)
        texts, labels = [], []
        for row in ds:
            t = row[text_col].strip()
            if len(t) > 40:
                texts.append(t)
                labels.append(int(row[lab_col]))
            if len(texts) >= n_train + n_test:
                break
        acts = []
        with torch.no_grad():
            for b0 in range(0, len(texts), 32):
                enc = tok(texts[b0 : b0 + 32], return_tensors="pt",
                          padding="max_length", truncation=True, max_length=ctx)
                ids = enc["input_ids"].to(dev)
                hs = model(ids).hidden_states[layer + 1]
                acts.append(hs.to("cpu", torch.bfloat16))
        A = torch.cat(acts)
        y = torch.tensor(labels)
        torch.save({
            "acts_train": A[:n_train], "y_train": y[:n_train],
            "acts_test": A[n_train:], "y_test": y[n_train:],
        }, out / f"concept_{name}.pt")
        summary[name] = {"n": len(texts), "classes": int(y.max()) + 1}
        print(f"cached {name}: {summary[name]}", flush=True)
    return summary
