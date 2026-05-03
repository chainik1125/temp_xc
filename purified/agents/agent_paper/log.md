---
author: agent_paper
date: 2026-05-03
status: active
---

## 2026-05-03 — Day 0 (kick-off)

- Read briefing.md, project_brief.md, and headline research logs for C1–C7.
- Surveyed wasteland: src/architectures/ has 50+ TXC variants (Phase 5+7
  hill-climbing residue); experiments/phase{2,3,5,6,7}_*/ each contain a
  subset of working code.
- Identified three contradictions vs the briefing:
  1. C6 EM result is actually a **negative TXC result** per Dmitry's
     2026-05-03 paper section (SAE arditi beats TXC k=100 at every cell).
  2. C5 steering hill-climb winners (Y/W's Galaxy variants) **lose** at
     probing per `2026-05-02-yw-T8-benchmark.md`.
  3. C7 backtracking is more salvageable than the briefing suggested —
     TXC peak Δgc=+1.574 (~3× next-best) is a real result.
- Asked Han three questions; locked answers:
  - TXC-base = `txc_bare_antidead_t5`; TXC-pro = `phase5b_subseq_h8`.
  - C6 reframed as honest negative.
  - `final` branch from `han-phase7-unification`, push to origin.
- Built `purified/` scaffold:
  - `pyproject.toml`, `.python-version`, `.gitignore`
  - `README.md`, `CLAUDE.md`, `PROTOCOL.md`
  - `src/temp_bench/{architectures,data,training,eval,case_studies,plotting,utils}/`
    skeletons with locked architecture registry + CaseStudy ABC
  - `docs/components/c{1..7}.md` writeup skeletons
  - `docs/paper/{outline,architecture}.md` paper drafts
  - `agents/agent_paper/{decisions,log}.md`
- Next: bootstrap script + experiment dir READMEs, then commit + push.

## TODO (next sessions)

- [ ] First commit + push of `final` to origin.
- [ ] Write `purified/scripts/bootstrap_runpod.sh` (port of Phase 7 script).
- [ ] Implement TXC-base + TXC-pro in `src/temp_bench/architectures/`
      (copy + simplify from wasteland).
- [ ] Implement C1 toy data generator + sweep script (5090-local).
- [ ] First C1 multi-seed run (3 seeds × 12 k values × 4 archs ≈ 6 hr local).
- [ ] Spawn Agent NLP brief for C3+C4 caching (1× H100 RunPod).
- [ ] Spawn Agent EM brief for C6 (1× H100 RunPod).
