#!/usr/bin/env python3
"""mac-d repatriation merge: append-only, dup-key-checked.

Appends incoming JSONL rows whose <key> is absent from the target file.
Rows whose key already exists must be byte-identical (the pod's clone
carries the full history); a same-key-different-content row is a CONFLICT —
loudly reported, never appended, never overwritten (row corrections belong
to the instrument owner, 184ebd47a precedent).

Default is a dry-run; pass --apply to write.
"""
import argparse
import json

p = argparse.ArgumentParser()
p.add_argument("--incoming", required=True)
p.add_argument("--target", required=True)
p.add_argument("--key", required=True, help="eval_key (leaderboard) / train_key (manifest)")
p.add_argument("--apply", action="store_true")
a = p.parse_args()


def load(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.strip():
                rows.append((line, json.loads(line)))
    return rows


tgt = load(a.target)
inc = load(a.incoming)
seen = {}
for line, row in tgt:
    k = row.get(a.key)
    if k is not None:
        seen.setdefault(k, line)

new, dup_same, conflicts, nokey = [], 0, [], 0
for line, row in inc:
    k = row.get(a.key)
    if k is None:
        nokey += 1
    elif k not in seen:
        new.append(line)
        seen[k] = line
    elif seen[k] == line:
        dup_same += 1
    else:
        conflicts.append(k)

print(
    f"incoming={len(inc)} target={len(tgt)} new={len(new)} "
    f"dup-identical={dup_same} CONFLICTS={len(conflicts)} nokey={nokey}"
)
for k in conflicts[:20]:
    print("  CONFLICT (skipped, owner decides):", k)
if conflicts:
    print("conflicts present — nothing applied; resolve with the row owner first")
    raise SystemExit(2)

if a.apply and new:
    with open(a.target, "a") as f:
        for line in new:
            f.write(line + "\n")
    print(f"appended {len(new)} rows to {a.target}")
elif new:
    print("(dry-run; re-run with --apply to append)")
