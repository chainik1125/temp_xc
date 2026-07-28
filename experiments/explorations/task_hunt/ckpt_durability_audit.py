"""Blast radius of the `train_cached=True` fiction — how many delivered
leaderboard rows rest on weights that no longer exist ANYWHERE durable?

mac-d found ONE instance (sycgen anchors); the hub verified the MECHANISM
(runner.py:141 returns train_cached=True as a hardcoded literal, so line
153's checkpoint_exists is unreachable on the short-circuit path). Neither
measured the EXTENT. This does, for $0, read-only, from local files.

Method — mirrors `cache.checkpoint_exists` exactly, both branches:
  (a) local  checkpoints/<train_key>/model.safetensors on THIS machine
  (b) durable  a manifest entry for train_key with `hf_url` SET

Why (b) is the load-bearing one: `local_path` in the manifest records a
path on WHATEVER machine trained it (rows here read /home/elysium/...),
so it is a historical note, not an existence proof. Pod volumes are
ephemeral and the 07-25 force majeure already destroyed one generation of
them. `hf_url` is the only store that survives a pod dying.

WHAT THIS CAN AND CANNOT SEE. From this mac I cannot inspect any pod's
disk. So a train_key with no hf_url is *not proven gone* — it is proven
**not durably recoverable**: it survives only if some specific machine is
still alive and still holds it. That is a bound, and it is the bound that
matters, because the inference this whole exercise invalidated was
"a row exists, therefore the weights do".
"""

from __future__ import annotations

import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(subprocess.run(["git", "rev-parse", "--show-toplevel"],
                           capture_output=True, text=True,
                           check=True).stdout.strip())
LB = ROOT / "results" / "leaderboard.jsonl"
MF = ROOT / "checkpoints" / "manifest.jsonl"
CKPT = ROOT / "checkpoints"


def rows(p):
    with p.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main():
    # ---- manifest: which train_keys have a DURABLE copy -----------------
    durable, manifest_seen, mf_ts = set(), set(), {}
    for m in rows(MF):
        tk = m.get("train_key")
        if not tk:
            continue
        manifest_seen.add(tk)
        mf_ts.setdefault(tk, m.get("ts"))
        if m.get("hf_url"):
            durable.add(tk)

    # ---- leaderboard: the delivered record ------------------------------
    lb_keys = defaultdict(lambda: {"n": 0, "exp": Counter(), "ds": Counter(),
                                   "arch": Counter(), "ts": None})
    for r in rows(LB):
        tk = r.get("train_key")
        if not tk:
            continue
        e = lb_keys[tk]
        e["n"] += 1
        e["exp"][r.get("experiment")] += 1
        e["ds"][r.get("datasource")] += 1
        e["arch"][r.get("arch")] += 1
        ts = r.get("ts")
        if ts and (e["ts"] is None or ts > e["ts"]):
            e["ts"] = ts

    local = {tk for tk in lb_keys if (CKPT / tk / "model.safetensors").exists()}

    n_keys = len(lb_keys)
    ok = {tk for tk in lb_keys if tk in durable or tk in local}
    gone = [tk for tk in lb_keys if tk not in ok]
    orphan = [tk for tk in gone if tk not in manifest_seen]

    print(f"leaderboard rows      {sum(e['n'] for e in lb_keys.values()):>7,}")
    print(f"distinct train_keys   {n_keys:>7,}")
    print(f"  durable (hf_url)    {len(durable & set(lb_keys)):>7,}")
    print(f"  local on this mac   {len(local):>7,}")
    print(f"  NOT durably recoverable {len(gone):>4,}"
          f"  ({100 * len(gone) / n_keys:.1f}% of train_keys,"
          f" {100 * sum(lb_keys[t]['n'] for t in gone) / sum(e['n'] for e in lb_keys.values()):.1f}% of rows)")
    print(f"    of which no manifest entry AT ALL: {len(orphan):,}")

    print("\n--- exposure by paper section (rows whose weights are not durable) ---")
    by_exp, by_exp_ok = Counter(), Counter()
    for tk, e in lb_keys.items():
        for exp, c in e["exp"].items():
            (by_exp_ok if tk in ok else by_exp)[exp] += c
    print(f"{'section':<16}{'rows exposed':>14}{'rows covered':>14}{'% exposed':>11}")
    for exp in sorted(set(by_exp) | set(by_exp_ok),
                      key=lambda x: -by_exp[x]):
        tot = by_exp[exp] + by_exp_ok[exp]
        print(f"{str(exp):<16}{by_exp[exp]:>14,}{by_exp_ok[exp]:>14,}"
              f"{100 * by_exp[exp] / tot:>10.1f}%")

    print("\n--- exposure by datasource (top 15 by exposed rows) ---")
    by_ds = Counter()
    for tk in gone:
        for ds, c in lb_keys[tk]["ds"].items():
            by_ds[ds] += c
    for ds, c in by_ds.most_common(15):
        print(f"  {str(ds):<44}{c:>7,}")

    print("\n--- when were the exposed keys last written? ---")
    by_month = Counter(
        (lb_keys[tk]["ts"] or "unknown")[:10] for tk in gone)
    for d, c in sorted(by_month.items()):
        print(f"  {d:<14}{c:>6,} train_keys")

    out = ROOT / "checkpoints" / "durability_audit.json"
    payload = {
        "generated_by": "mac-c", "method": "mirrors cache.checkpoint_exists",
        "n_train_keys": n_keys, "n_durable": len(durable & set(lb_keys)),
        "n_local_this_mac": len(local), "n_not_durable": len(gone),
        "n_orphan_no_manifest": len(orphan),
        "exposed_rows_by_section": dict(by_exp),
        "covered_rows_by_section": dict(by_exp_ok),
        "exposed_rows_by_datasource": dict(by_ds),
        "caveat": ("cannot see pod disks; 'not durable' means recoverable "
                   "only if a specific machine is still alive, NOT proven gone"),
    }
    print(f"\n(payload ready, {len(json.dumps(payload))} bytes -> {out})")
    return payload


if __name__ == "__main__":
    main()
