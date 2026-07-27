"""P1-RM equivalence checker (protocol (a) of the 19:49 halt proposal).

For every TRAINED relu-mix row in the canonical leaderboard, find its
btk-only twin (same base arch, seed, T), load both checkpoints from
the manifest, and compare EVERY tensor with ``torch.equal``. Emits
``RM_EQUIVALENCE.md`` + ``rm_equivalence.json`` next to this file:
one row per (arch, seed, T) pair — tensors-equal verdict, per-tensor
detail on any mismatch, metric deltas from the leaderboard rows, and
the bookkeeping-only key differences (e.g. ``threshold_set``).

Mechanical + CPU-only; no training, no eval. Run:
  .venv/bin/python -m experiments.probing.actmix.rm_equivalence
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]

TWIN = {
    "batchtopk_sae": "batchtopk_sae_btkonly",
    "txc_batchtopk_pre": "txc_batchtopk_pre_btkonly",
    "txc_batchtopk_post": "txc_batchtopk_post_btkonly",
}

# ALIAS EXCLUSION LIST (assigned 184ebd47a; published in
# RM_EQUIVALENCE.md). These are UNTRAINED-twin train_keys (n_steps=0)
# whose byte-equal metric clusters aliased an external trained-row
# join into a phantom "T8-exact" pair. They are legitimate rows —
# excluded from ARM-DIFF joins, not from the record. Two clusters of
# one physical untrained model each:
#   pre-T8-untrained : 27e5b452ad79957d, 3b99316b93e9bea2, a19178296a960d32
#   post-T8-untrained: 4cdb346b79c0ecf6, 73da804cf540cd56, 84a423f9dd529c0f
ALIAS_EXCLUDE = frozenset({
    "27e5b452ad79957d", "3b99316b93e9bea2", "a19178296a960d32",
    "4cdb346b79c0ecf6", "73da804cf540cd56", "84a423f9dd529c0f",
})


def _rows():
    out = {}
    with (ROOT / "results" / "leaderboard.jsonl").open() as fh:
        for line in fh:
            r = json.loads(line)
            ec = r.get("eval_cfg") or {}
            if ec.get("smoke") or ec.get("positive_control"):
                continue          # control rows are the instrument's, not the grid's
            if not (r.get("training_cfg") or {}).get("n_steps"):
                continue
            if r.get("train_key") in ALIAS_EXCLUDE:
                continue
            arm = ec.get("arm")
            if arm not in ("relu-mix", "btk-only"):
                continue
            T = (r["training_cfg"].get("arch_hparams_override") or {}).get("T")
            key = (arm, r["arch"], int(r["seed"]), T, int(ec["k_feat"]))
            # HOUSE RULE (184ebd47a): surface duplicate keys — silent
            # last-write-wins pooling is how the T8 phantom was made.
            if key in out and out[key]["train_key"] != r["train_key"]:
                print(f"[equiv] DUPLICATE slot {key}: train_keys "
                      f"{out[key]['train_key']} vs {r['train_key']} "
                      "(keeping latest; verify provenance)")
            out[key] = r
    return out


def _ckpt(train_key: str) -> Path:
    for line in (ROOT / "checkpoints" / "manifest.jsonl").read_text().splitlines():
        try:
            j = json.loads(line)
        except Exception:
            continue
        if j.get("train_key") == train_key and "local_path" in j:
            return Path(j["local_path"])
    raise FileNotFoundError(train_key)


def main():
    rows = _rows()
    pairs, seen = [], set()
    for (arm, arch, seed, T, k), r in sorted(rows.items()):
        if arm != "relu-mix" or arch not in TWIN:
            continue
        pid = (arch, seed, T)
        if pid in seen:
            continue
        seen.add(pid)
        twin = rows.get(("btk-only", TWIN[arch], seed, T, k)) or rows.get(
            ("btk-only", TWIN[arch], seed, T, 5 if k == 20 else 20))
        if twin is None:
            pairs.append({"pair": pid, "verdict": "NO-TWIN"})
            continue
        try:
            a = load_file(_ckpt(r["train_key"]))
            b = load_file(_ckpt(twin["train_key"]))
        except FileNotFoundError:
            # Cross-pod cell (e.g. runpod-a's shard 2): weights not local.
            # Fallback: machine-precision metric equality across the full
            # per-task vector — necessary-not-sufficient, marked as such.
            am, bm = r["metrics"], twin["metrics"]
            tk = [kk for kk in am if kk.startswith("auc__")]
            mism = [kk for kk in tk if am.get(kk) != bm.get(kk)]
            pairs.append({
                "pair": pid, "rm_train_key": r["train_key"],
                "btk_train_key": twin["train_key"],
                "tensors_compared": 0,
                "per_task_metric_mismatches": mism,
                "mean_auc_delta_at_shared_k":
                    am["mean_auc"] - bm["mean_auc"],
                "verdict": ("METRIC-IDENTICAL (weights remote)"
                            if not mism and am["mean_auc"] == bm["mean_auc"]
                            else "METRIC-DIVERGES (weights remote)"),
            })
            continue
        common = sorted(set(a) & set(b))
        diffs = [kk for kk in common if not torch.equal(a[kk], b[kk])]
        extra = sorted(set(a) ^ set(b))
        dm = (r["metrics"]["mean_auc"] - twin["metrics"]["mean_auc"])
        pairs.append({
            "pair": pid, "rm_train_key": r["train_key"],
            "btk_train_key": twin["train_key"],
            "tensors_compared": len(common),
            "tensor_mismatches": diffs,
            "bookkeeping_only_keys": extra,
            "mean_auc_delta_at_shared_k": dm,
            "verdict": "IDENTICAL" if not diffs else "DIVERGES",
        })
    n_id = sum(1 for p in pairs if p.get("verdict") == "IDENTICAL")
    md = ["# P1-RM ↔ btk-only weight-equivalence table "
          "(auto: rm_equivalence.py, protocol (a))", "",
          f"{n_id}/{len(pairs)} pairs IDENTICAL (torch.equal on every "
          "shared tensor).", "",
          "## ALIAS EXCLUSION LIST (184ebd47a assignment — any future "
          "arm-diff must exclude these train_keys)", "",
          "Untrained-twin keys (n_steps=0; legitimate rows) whose "
          "byte-equal clusters aliased an external trained-row join "
          "into the phantom 'T8-exact' pair. One physical untrained "
          "model per cluster:", "",
          "- pre-T8-untrained: `27e5b452ad79957d`, `3b99316b93e9bea2`, "
          "`a19178296a960d32`",
          "- post-T8-untrained: `4cdb346b79c0ecf6`, `73da804cf540cd56`, "
          "`84a423f9dd529c0f`", "",
          "House rule enforced by this checker: joins filter n_steps>0 "
          "AND this list, and DUPLICATE slot keys are surfaced, never "
          "silently pooled. (Prose correction on the 22:36 entry, "
          "receipts authoritative: +8.75e-3 was T8's k5 delta; k20 = "
          "+1.02e-2; largest |Δ| overall is T6's −1.63e-2.)", "",
          "| arch | seed | T | tensors | verdict | Δauc | extra keys |",
          "|---|---|---|---|---|---|---|"]
    for p in pairs:
        arch, seed, T = p["pair"]
        md.append("| {} | {} | {} | {} | **{}** | {} | {} |".format(
            arch, seed, T, p.get("tensors_compared", "—"), p["verdict"],
            ("{:+.2e}".format(p["mean_auc_delta_at_shared_k"])
             if "mean_auc_delta_at_shared_k" in p else "—"),
            ",".join(p.get("bookkeeping_only_keys", [])) or "—"))
    (HERE / "RM_EQUIVALENCE.md").write_text("\n".join(md) + "\n")
    (HERE / "rm_equivalence.json").write_text(json.dumps(pairs, indent=1))
    print("\n".join(md))


if __name__ == "__main__":
    main()
