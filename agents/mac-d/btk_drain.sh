#!/bin/bash
# mac-d — btk gap-cell DRAIN: repatriate, verify, re-render, prove.
#
#   bash agents/mac-d/btk_drain.sh
#
# Encodes the hub's ruling (f699c80a4): re-render at drain with
# --tag final, ff242b78 as the recorded baseline, new hash posted with
# a supersedes line, and **only the caption + the two new columns may
# change** — anything else moving means the render pathway is not what
# we think it is.
#
# That last condition is the reason this is a script and not a habit:
# it is checked here by DIFFING THE PER-T SERIES, not by eyeballing the
# figure. Six guards today reported success while doing nothing; this
# one states what it would look like if it failed (a non-T10 column
# moving prints DRIFT and the script stops before committing).
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"
SCRATCH=/private/tmp/claude-501/-Users-octo-research-projects-agents-mac-d-temp-xc/fda6ec07-55ee-444a-b9f9-223686962949/scratchpad
PRE="$SCRATCH/btk_series_T6done_T10pending.json"   # captured 17:41: T6:n=3 T10:n=1
BASELINE=ff242b784b654673                          # committed deliverable, runpod-2 10:15
POD_IP=31.24.80.41; POD_PORT=12997

echo "=== 1. repatriate btk rows from the pod ==="
ssh -p $POD_PORT -o ConnectTimeout=15 -o BatchMode=yes root@$POD_IP \
  'grep -h "txc_batchtopk_post_btkonly" /workspace/temp_xc/results/leaderboard.jsonl 2>/dev/null | grep "\"n_steps\":25000"' \
  > /tmp/btk_drain.jsonl || { echo "FAIL: could not read pod leaderboard"; exit 1; }

.venv/bin/python - <<'PY' || exit 1
import json, pathlib
lb = pathlib.Path("results/leaderboard.jsonl"); ex = {}
for line in lb.read_text().splitlines():
    if line.strip():
        r = json.loads(line); k = r.get("eval_key")
        if k: ex[k] = line
new, conf = [], 0
for line in pathlib.Path("/tmp/btk_drain.jsonl").read_text().splitlines():
    if not line.strip(): continue
    r = json.loads(line); k = r.get("eval_key")
    if k in ex:
        conf += 0 if ex[k] == line else 1
    else:
        new.append(line); ex[k] = line
if conf:
    print(f"  CONFLICTS {conf} — all-or-nothing, appending NOTHING"); raise SystemExit(1)
if new:
    with lb.open("a") as fh:
        for l in new: fh.write(l + "\n")
print(f"  appended {len(new)} new rows, 0 conflicts")
PY

echo "=== 2. coverage must be uniform 3 seeds at every T ==="
.venv/bin/python - <<'PY' || { echo "  NOT uniform — stopping before render"; exit 1; }
import collections, sys
import experiments.explorations.actmix_rlhf.render_writeup_fig as R
from experiments.explorations.actmix_rlhf.cells import TXC_ARCH, DATASOURCE
s, _ = R.load_points(TXC_ARCH, DATASOURCE)
c = collections.Counter(T for T, _ in s)
print("  coverage:", dict(sorted(c.items())))
sys.exit(0 if all(v == 3 for v in c.values()) else 1)
PY

echo "=== 3. baseline sha (must be the COMMITTED deliverable, not a re-render) ==="
CUR=$(shasum -a 256 figs_writeup/fig_rlhf_shuffle_tsweep.png | cut -c1-16)
echo "  on disk: $CUR   expected: $BASELINE"
[ "$CUR" = "$BASELINE" ] || { echo "  BASELINE MISMATCH — stopping"; exit 1; }

echo "=== 4. render --tag final ==="
.venv/bin/python -m experiments.explorations.actmix_rlhf.render_writeup_fig --tag final || exit 1
NEW=$(shasum -a 256 figs_writeup/fig_rlhf_shuffle_tsweep.png | cut -c1-16)
echo "  sha transition: $BASELINE -> $NEW"

echo "=== 5. PROVE only T10 moved (hub condition) ==="
.venv/bin/python - "$PRE" <<'PY' || { echo "  DRIFT — a column other than T10 moved; DO NOT COMMIT"; exit 1; }
import json, sys
import experiments.explorations.actmix_rlhf.render_writeup_fig as R
from experiments.explorations.actmix_rlhf.cells import TXC_ARCH, DATASOURCE
pre = json.load(open(sys.argv[1]))
s, _ = R.load_points(TXC_ARCH, DATASOURCE)
bad = False
for field in ("ordered", "shuffled"):
    Ts, mu, sd, n = R.mean_sd(s, field)
    for t, m, d, k in zip(Ts, mu, sd, n):
        p = pre[field].get(str(t))
        if p is None:
            print(f"  {field} T{t}: NEW column (unexpected)"); bad = True; continue
        moved = abs(p["mu"] - m) > 1e-12 or p["n"] != k
        if moved and t != 10:
            print(f"  DRIFT {field} T{t}: mu {p['mu']:.6f}->{m:.6f} n {p['n']}->{k}"); bad = True
        elif moved:
            print(f"  expected change {field} T{t}: mu {p['mu']:.6f}->{m:.6f} n {p['n']}->{k}")
sys.exit(1 if bad else 0)
PY
echo "  VERIFIED: only T10 changed; every other column bit-equal"
echo
echo "NEXT (manual): commit fig + rows with the supersedes line, then"
echo "  bash agents/mac-d/teardown_pod.sh tnp7vvew4t80wi btk 6 10"
