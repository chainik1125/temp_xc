# MODAL SPEND LEDGER — shared, append-only (union-merge like the LOG)

**$500 HARD CEILING (Dmitry). $400 soft stop TOTAL. Read the total
before EVERY launch; append after. Caps: mac-a $150, mac-b $100
(raises only by mac-local amendment). Modal dashboard is authority.**

| when (PT) | agent | what | GPU/CPU | est. cost | running total |
|---|---|---|---|---|---|
| 07-26 02:10 | mac-local | smoke attempts + A10 hello | A10 | ~$1 | ~$1 |
| 07-25 18:40 | mac-b | slen screen pipeline: image build + smoke + fineweb-400 caches (gpt2+llama31) + 2× screen (est ≤ 2 A10G-h each, parallel) | A10G | ~$5 | ~$6 |
| 07-25 18:55 | mac-a | tsae top-up bring-up: image build (uv sync) + in-container run.py validate | CPU 4-core | ~$0.5 | ~$6.5 |
| 07-25 19:20 | mac-a | Ward cache rebuild (stream + labels + base/hs13, receipts hard-fail) → Volume | A10G + 8cpu + 48G + 512GiB disk, ~1h | ~$3 | ~$9.5 |
