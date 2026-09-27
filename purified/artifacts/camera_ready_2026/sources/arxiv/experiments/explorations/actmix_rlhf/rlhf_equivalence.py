"""RLHF relu-mix ↔ btk-only equivalence checker (CARD § 7 A3).

runpod-1's rm_equivalence tensor-compare core (torch.equal on every
shared checkpoint tensor, manifest-resolved) with row-matching
adapted to the RLHF lanes: this card's rows carry no eval_cfg arm
key, so pairs are formed by (arch twin-map, seed, T/k overrides) on
canonical leaderboard rows for the RLHF evaluator + datasource.
Latest row per key wins. Emits RLHF_EQUIVALENCE.{md,json} here.

Ruling (mac-local c6e464881): IDENTICAL on all pairs ⇒ the queued
relu-mix overnight card is CANCELLED (certificate + btk-only curve
= the both-arms deliverable); DIVERGES ⇒ train relu-mix only on
measured-divergent configs.

Run:  .venv/bin/python -m experiments.explorations.actmix_rlhf.rlhf_equivalence
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from experiments.explorations.actmix_rlhf.cells import DATASOURCE

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]

TWIN = {"batchtopk_sae": "batchtopk_sae_btkonly",
        "txc_batchtopk_post": "txc_batchtopk_post_btkonly",
        "tsae": "tsae_btkonly"}


def _rows():
    out = {}
    with (ROOT / "results" / "leaderboard.jsonl").open() as fh:
        for line in fh:
            r = json.loads(line)
            tc = r.get("training_cfg") or {}
            if (r.get("evaluator_name") != "rlhf"
                    or r.get("datasource") != DATASOURCE
                    or not tc.get("n_steps")):
                continue
            ov = tc.get("arch_hparams_override") or {}
            key = (r["arch"], int(r["seed"]), ov.get("T"), ov.get("k_pos"))
            out[key] = r  # append order -> latest wins
    return out


def _ckpt(train_key):
    for line in (ROOT / "checkpoints" /
                 "manifest.jsonl").read_text().splitlines():
        try:
            j = json.loads(line)
        except Exception:
            continue
        if j.get("train_key") == train_key and "local_path" in j:
            return Path(j["local_path"])
    raise FileNotFoundError(train_key)


def main():
    rows = _rows()
    pairs = []
    for (arch, seed, T, k), r in sorted(rows.items(),
                                        key=lambda x: str(x[0])):
        if arch not in TWIN:
            continue
        twin = rows.get((TWIN[arch], seed, T, k))
        pid = {"arch": arch, "seed": seed, "T": T, "k_pos": k}
        if twin is None:
            pairs.append({**pid, "verdict": "NO-TWIN"})
            continue
        a = load_file(_ckpt(r["train_key"]))
        b = load_file(_ckpt(twin["train_key"]))
        common = sorted(set(a) & set(b))
        diffs = [kk for kk in common if not torch.equal(a[kk], b[kk])]
        extra = sorted(set(a) ^ set(b))
        d = (r["metrics"]["preference_auc_k20"]
             - twin["metrics"]["preference_auc_k20"])
        pairs.append({**pid,
                      "relumix_train_key": r["train_key"],
                      "btk_train_key": twin["train_key"],
                      "tensors_compared": len(common),
                      "tensor_mismatches": diffs,
                      "bookkeeping_only_keys": extra,
                      "pref_auc_k20_delta": d,
                      "verdict": "IDENTICAL" if not diffs else "DIVERGES"})
    n_id = sum(1 for p in pairs if p["verdict"] == "IDENTICAL")
    n_cmp = sum(1 for p in pairs if p["verdict"] != "NO-TWIN")
    md = ["# RLHF relu-mix ↔ btk-only weight equivalence (CARD § 7 A3)",
          "", f"{n_id}/{n_cmp} compared pairs IDENTICAL "
          "(torch.equal on every shared tensor).", "",
          "| arch | seed | T | k_pos | tensors | verdict | Δauc_k20 | extra keys |",
          "|---|---|---|---|---|---|---|---|"]
    for p in pairs:
        md.append("| {arch} | {seed} | {T} | {k_pos} | {tc} | **{v}** | "
                  "{d} | {e} |".format(
                      tc=p.get("tensors_compared", "—"), v=p["verdict"],
                      d=("{:+.2e}".format(p["pref_auc_k20_delta"])
                         if "pref_auc_k20_delta" in p else "—"),
                      e=",".join(p.get("bookkeeping_only_keys", [])) or "—",
                      **{k2: p.get(k2) for k2 in
                         ("arch", "seed", "T", "k_pos")}))
    (HERE / "RLHF_EQUIVALENCE.md").write_text("\n".join(md) + "\n")
    (HERE / "results" / "rlhf_equivalence.json").write_text(
        json.dumps(pairs, indent=1))
    print("\n".join(md))


if __name__ == "__main__":
    main()
