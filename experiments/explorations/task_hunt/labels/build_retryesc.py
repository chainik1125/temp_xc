"""``retryesc`` stream build + LABEL-SIDE pre-measure (card
``retryesc/CARD.md``; constants frozen in ``retryesc_lib``).

    .venv/bin/python -m experiments.explorations.task_hunt.labels.build_retryesc

$0, CPU. Builds one document per agent trace from the pinned mirror,
computes the T2 age face (tokens since the last FAILED environment
turn), and runs the trio's label-side instruments. A GPU screen is
bought only if this passes.

Face choice, stated: the menu proposed a T1 **rate** face; the measured
clock kills that (inter-failure gap median ~886 tokens against T ≤ 64,
so the kernel support sits far outside the window — refmark's
reach-limited death, the same reasoning that demoted ``sycgen_rate``).
The **age** face is well-defined at any distance and is carried
instead.

Probe eligibility: AGENT tokens only, with every environment turn
masked out — the failure text itself is never probe-eligible, so the
visible-cue floor cannot read the event directly.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import numpy as np

from . import msdose_r1_lib as mr1
from . import novelty_lib as nl
from . import retryesc_lib as rl
from . import wave3_lib as w3
from .build_wave3_trio import (N_REPS, SEED, _floor_aucs, _spear, _terciles,
                               _triage)
from .gen4c_lib import section_age
from .hunt3_lib import FLOOR_TS
from .lib import doc_split

HERE = Path(__file__).resolve().parent
TOKS = ("gpt2", "gemma2", "llama31")
FROZEN = ("experiments/explorations/task_hunt/labels/retryesc_lib.py",
          "experiments/explorations/task_hunt/labels/build_retryesc.py")

# pre-registered bands (card § 5) — ABSOLUTE ONLY (msdose_r1 lesson)
BAND_UNIGRAM_MAX = 0.60
BAND_DOCMEAN_MAX = 0.88
BAND_POSITION_MAX = 0.95
BAND_QUAL_MIN = 8
BAND_USABLE_MIN = 250_000
BAND_EVENTS_MIN = 300


def _receipt() -> dict:
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE,
                          capture_output=True, text=True,
                          check=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--"] + list(FROZEN),
                           cwd=HERE, capture_output=True, text=True,
                           check=True).stdout.strip()
    assert not dirty, f"frozen logic dirty:\n{dirty}"
    return {"head": head, "frozen_files_clean": True}


def build(key: str, convs: list) -> dict:
    from transformers import AutoTokenizer
    hf = {"gpt2": "gpt2", "gemma2": "google/gemma-2-2b",
          "llama31": "NousResearch/Meta-Llama-3.1-8B"}[key]
    tok = AutoTokenizer.from_pretrained(hf)

    ids_l, agent_l, evfirst_l, evmask_l, off = [], [], [], [], [0]
    n_ev, n_turns, turn_tok = 0, 0, []
    for conv in convs:
        pi, pa, pf, pm = [], [], [], []
        for t in conv:
            role = t.get("role")
            content = str(t.get("content", "") or "")
            enc = np.asarray(tok(content, add_special_tokens=False)
                             ["input_ids"], dtype=np.int32)
            if enc.size == 0:
                continue
            n_turns += 1
            turn_tok.append(enc.size)
            is_agent = int(role == rl.ROLE_AGENT)
            fail = rl.is_failure_turn(role, content)
            f = np.zeros(enc.size, dtype=np.int8)
            m = np.zeros(enc.size, dtype=np.int8)
            if fail:
                f[0] = 1                       # event = first token of turn
                m[:] = 1                       # whole turn masked
                n_ev += 1
            if not is_agent:
                m[:] = 1                       # ALL environment turns masked
            pi.append(enc)
            pa.append(np.full(enc.size, is_agent, dtype=np.int8))
            pf.append(f)
            pm.append(m)
        if not pi:
            continue
        ids_l.append(np.concatenate(pi))
        agent_l.append(np.concatenate(pa))
        evfirst_l.append(np.concatenate(pf))
        evmask_l.append(np.concatenate(pm))
        off.append(off[-1] + len(ids_l[-1]))

    ids = np.concatenate(ids_l)
    agent = np.concatenate(agent_l)
    first = np.concatenate(evfirst_l)
    mask = np.concatenate(evmask_l)
    doc_off = np.array(off, dtype=np.int64)
    n_docs = len(doc_off) - 1
    doc_of = np.repeat(np.arange(n_docs, dtype=np.int32), np.diff(doc_off))
    pos_of = np.concatenate([np.arange(n, dtype=np.int32)
                             for n in np.diff(doc_off)])
    split = doc_split(n_docs, seed=SEED)
    train_rows, test_rows = split[doc_of] == 0, split[doc_of] == 1

    age = np.concatenate([w3.sage_face(first[doc_off[d]:doc_off[d + 1]])
                          for d in range(n_docs)])
    raw = np.concatenate([section_age(first[doc_off[d]:doc_off[d + 1]])
                          for d in range(n_docs)])

    clock = {"n_docs": n_docs, "n_tokens": int(ids.size),
             "tokens_per_doc_mean": float(ids.size / n_docs),
             "tokens_per_turn_mean": float(np.mean(turn_tok)),
             "n_turns": n_turns, "events_total": int(n_ev),
             "failure_turn_frac": float(n_ev / max(n_turns, 1)),
             "agent_token_frac": float(agent.mean())}
    print(f"  [{key}] CLOCK: {clock['tokens_per_doc_mean']:.0f} tok/trace, "
          f"{clock['tokens_per_turn_mean']:.0f} tok/turn, "
          f"{n_ev} events ({clock['failure_turn_frac']*100:.1f}% of turns), "
          f"agent tokens {clock['agent_token_frac']*100:.1f}%", flush=True)

    elig = (mask == 0) & (agent == 1) & (pos_of >= rl.MIN_POS) \
        & np.isfinite(age)
    a = raw[elig]
    claim = {f"T{T}": float((a <= T).mean()) for T in FLOOR_TS}
    print(f"  [{key}] claim zone (event inside window): "
          + " ".join(f"T{T}={v*100:.2f}%" for T, v in
                     zip(FLOOR_TS, claim.values())), flush=True)

    bins, edges = _terciles(age, train_rows, elig)
    st = {"eligible_rows": int(elig.sum()), "tercile_edges": edges,
          "clock_stated_first": clock, "claim_zone_in_window": claim,
          "raw_age": {"mean": float(a.mean()), "median": float(np.median(a))},
          "face_position_spearman": _spear(age, pos_of.astype(float), elig)}
    st.update(_triage("retryesc_age", age, bins, ids, pos_of, doc_of, doc_off,
                      train_rows, test_rows, N_REPS))
    floors = {"censored_age": {T: np.concatenate(
        [w3.sage_floor(first[doc_off[d]:doc_off[d + 1]], T)
         for d in range(n_docs)]) for T in FLOOR_TS},
        "in_window_event_tokens": {T: np.concatenate(
            [w3.dose_window_count(mask[doc_off[d]:doc_off[d + 1]], T)
             for d in range(n_docs)]) for T in FLOOR_TS}}
    st["floors"] = _floor_aucs("retryesc_age", bins, test_rows, floors)
    st["strata_census"] = mr1.strata_census(age, pos_of, elig)
    c = st["strata_census"]
    print(f"    census: {c['n_qualifying']}/{c['n_strata_any']} strata, "
          f"{c['usable_tokens']:,} usable", flush=True)
    st["bands"] = {
        "unigram_le_0.60": (st["unigram_auc"], st["unigram_auc"]
                            <= BAND_UNIGRAM_MAX),
        "doc_mean_le_0.88": (st["doc_mean_only_auc"], st["doc_mean_only_auc"]
                             <= BAND_DOCMEAN_MAX),
        "position_le_0.95": (st["position_auc"], st["position_auc"]
                             <= BAND_POSITION_MAX),
        "qualifying_ge_8": (c["n_qualifying"], c["n_qualifying"]
                            >= BAND_QUAL_MIN),
        "usable_ge_250k": (c["usable_tokens"], c["usable_tokens"]
                           >= BAND_USABLE_MIN),
        "events_ge_300": (n_ev, n_ev >= BAND_EVENTS_MIN),
    }
    st["bands"] = {k: {"value": float(v), "pass": bool(p)}
                   for k, (v, p) in st["bands"].items()}
    st["all_pass"] = all(v["pass"] for v in st["bands"].values())
    print("    bands: " + ", ".join(
        f"{k}={'PASS' if v['pass'] else 'FAIL'}"
        for k, v in st["bands"].items()), flush=True)

    out = HERE / f"retryesc_{key}.npz"
    np.savez_compressed(out, token_ids=ids, doc_off=doc_off,
                        is_agent=agent, event_first=first, event_mask=mask,
                        retryesc_age=age, doc_split=split)
    st["artifact"] = out.name
    return st


def main():
    import pandas as pd
    from huggingface_hub import hf_hub_download
    t0 = time.time()
    receipt = _receipt()
    p = hf_hub_download(rl.DATASET, rl.DATA_FILE, repo_type="dataset",
                        revision=rl.REVISION)
    df = pd.read_parquet(p)
    convs = [list(c) for c in df["conversations"]]
    stats = {"card": "retryesc/CARD.md", "freeze_receipt": receipt,
             "substrate": {"dataset": rl.DATASET, "revision": rl.REVISION,
                           "n_traces": len(convs),
                           "agents": sorted(set(df["agent"])),
                           "models": sorted(set(df["model"])),
                           "n_tasks": int(df["task"].nunique()),
                           "provenance_warning":
                               "THIRD-PARTY MIRROR, not the official "
                               f"{rl.OFFICIAL_SOURCE}; single agent-model "
                               "distribution"},
             "frozen_fail_patterns": list(rl.FAIL_PATTERNS),
             "per_tokenizer": {}}
    for key in TOKS:
        print(f"[retryesc] {key}", flush=True)
        stats["per_tokenizer"][key] = build(key, convs)
    surv = [k for k in TOKS if stats["per_tokenizer"][k]["all_pass"]]
    stats["verdict_input"] = {
        "tokenizers_all_pass": surv,
        "kill_rule": "must pass every band on all 3 tokenizers; "
                     "otherwise label-side KILL, no GPU screen",
        "label_side_killed": bool(len(surv) < 3)}
    out = HERE / "retryesc_premeasure.json"
    out.write_text(json.dumps(stats, indent=1))
    print(f"-> {out} in {time.time()-t0:.0f}s; passing tokenizers: "
          f"{surv or 'NONE'}", flush=True)


if __name__ == "__main__":
    main()
