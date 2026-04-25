# Phase 6.1 — concat_random top-8 label content comparison

Seed=42. Top-8 features per arch (ranked by per-token variance).
Verdict = majority of 2 Haiku judges at temperature=0.

## By arch

### T-SAE (paper-faithful)
*Metrics: 12/32 semantic, coverage 7/7, judge disagreement 0.03*

| rank | feat_idx | peak | verdict | label |
|---|---|---|---|---|
| 1 | 3582 | fw_05 | SYNTACTIC | Quotation marks and punctuation marking |
| 2 | 2982 | fw_03 | SEMANTIC | color and light technical terminology |
| 3 | 2138 | fw_04 | SYNTACTIC | Words introducing lists or enumerations |
| 4 | 2745 | fw_03 | SYNTACTIC | Numerical digits within formatted data strings |
| 5 | 3457 | fw_05 | SEMANTIC | Personal anecdotes about family routines |
| 6 | 1376 | fw_02 | SYNTACTIC | Beginning of new text passages or documents |
| 7 | 1673 | fw_01 | SYNTACTIC | Birth dates and biographical information formatting |
| 8 | 2882 | fw_06 | SYNTACTIC | Punctuation and formatting markers in text |

### TXC+anti-dead (Track 2)
*Metrics: 5/32 semantic, coverage 7/7, judge disagreement 0.22*

| rank | feat_idx | peak | verdict | label |
|---|---|---|---|---|
| 1 | 12211 | fw_04 | SYNTACTIC | Beginning of sequence with capitalized words |
| 2 | 1846 | fw_03 | SYNTACTIC | Beginning of new text passages or documents |
| 3 | 1237 | fw_00 | SYNTACTIC | Beginning of sentence with capitalized word |
| 4 | 1805 | fw_00 | UNKNOWN | Beginning of new text passage or document |
| 5 | 1521 | fw_00 | SYNTACTIC | Beginning of sentence with capitalized word |
| 6 | 6810 | fw_02 | SYNTACTIC | Beginning of sentence with capitalized word token |
| 7 | 14994 | fw_05 | ⚠ SYNTACTIC | Common function words in conditional/descriptive phrases |
| 8 | 8226 | fw_01 | SYNTACTIC | Beginning of sentence with capitalized word |

### T-SAE (naive port)
*Metrics: 3/32 semantic, coverage 6/7, judge disagreement 0.12*

| rank | feat_idx | peak | verdict | label |
|---|---|---|---|---|
| 1 | 8889 | fw_00 | SYNTACTIC | Beginning of sentence markers in web content |
| 2 | 1419 | fw_06 | SYNTACTIC | Beginning of new text sections or documents |
| 3 | 5575 | fw_06 | UNKNOWN | Beginning of new text section or document |
| 4 | 8448 | fw_00 | ⚠ SYNTACTIC | Beginning of sequence token activation |
| 5 | 5217 | fw_00 | SYNTACTIC | Beginning of sequence markers in text |
| 6 | 6813 | fw_06 | SYNTACTIC | Pipe delimiters separating metadata fields |
| 7 | 4053 | fw_04 | SYNTACTIC | Commas and conjunctions in product lists |
| 8 | 1662 | fw_01 | SYNTACTIC | Digits within date values |

### MLC (Phase 5.7)
*Metrics: 2/32 semantic, coverage 4/7, judge disagreement 0.16*

| rank | feat_idx | peak | verdict | label |
|---|---|---|---|---|
| 1 | 2693 | fw_06 | SYNTACTIC | Beginning of new text segment or document |
| 2 | 7338 | fw_00 | SYNTACTIC | Beginning of new text section or document |
| 3 | 4454 | fw_06 | SYNTACTIC | Beginning of new text passage or document |
| 4 | 70 | fw_01 | SYNTACTIC | Individual digits within dates and numbers |
| 5 | 3891 | fw_00 | ⚠ SYNTACTIC | Beginning of sequence token activation |
| 6 | 152 | fw_01 | UNKNOWN | (claude error: RateLimitError) |
| 7 | 6773 | fw_00 | SYNTACTIC | Beginning of new text sections or documents |
| 8 | 4528 | fw_00 | SYNTACTIC | Beginning of new text passage or document |

### TXC+BatchTopK (Cycle F)
*Metrics: 0/32 semantic, coverage 6/7, judge disagreement 0.12*

| rank | feat_idx | peak | verdict | label |
|---|---|---|---|---|
| 1 | 8146 | fw_00 | SYNTACTIC | Punctuation marks followed by beginning of sequence |
| 2 | 1310 | fw_03 | SYNTACTIC | Beginning of new text section or document |
| 3 | 5630 | fw_03 | SYNTACTIC | Beginning of sentence with capitalized word |
| 4 | 17462 | fw_03 | SYNTACTIC | Beginning of sequence markers and boundaries |
| 5 | 11821 | fw_04 | SYNTACTIC | Beginning of sequence markers and formatting tokens |
| 6 | 18185 | fw_05 | SYNTACTIC | Transition between main text and metadata/headers |
| 7 | 4811 | fw_00 | UNKNOWN | Beginning of sentence with quoted word or phrase |
| 8 | 9757 | fw_05 | ⚠ SYNTACTIC | Transition between unrelated text sections |

### TXC (baseline)
*Metrics: 0/32 semantic, coverage 7/7, judge disagreement 0.22*

| rank | feat_idx | peak | verdict | label |
|---|---|---|---|---|
| 1 | 11627 | fw_00 | SYNTACTIC | Beginning of sentence with capitalized word |
| 2 | 12227 | fw_03 | SYNTACTIC | Beginning of sentence with capitalized word token |
| 3 | 7365 | fw_06 | SYNTACTIC | Beginning of sentence with quoted or capitalized word |
| 4 | 3122 | fw_06 | SYNTACTIC | Beginning of new text document or section |
| 5 | 7356 | fw_00 | SYNTACTIC | Beginning of sentence with capitalized word |
| 6 | 3900 | fw_02 | SYNTACTIC | Beginning of sentence capitalized words |
| 7 | 997 | fw_01 | SYNTACTIC | Beginning of new text section or document |
| 8 | 17207 | fw_05 | ⚠ SYNTACTIC | Transition between document sections or metadata |

### TXC+AuxK (Cycle A)
*Metrics: 0/32 semantic, coverage 7/7, judge disagreement 0.22*

| rank | feat_idx | peak | verdict | label |
|---|---|---|---|---|
| 1 | 12539 | fw_00 | SYNTACTIC | Beginning of sentence or document markers |
| 2 | 11193 | fw_00 | SYNTACTIC | Beginning of sentence with capitalized word |
| 3 | 12722 | fw_00 | SYNTACTIC | Beginning of sentence with capitalized word |
| 4 | 6913 | fw_04 | SYNTACTIC | Beginning of sentence or document markers |
| 5 | 16493 | fw_06 | ⚠ SYNTACTIC | Transition between unrelated text segments |
| 6 | 8196 | fw_00 | SYNTACTIC | Beginning of new text section or document |
| 7 | 15021 | fw_00 | ⚠ SYNTACTIC | Transition between document sections or metadata |
| 8 | 7837 | fw_00 | ⚠ SYNTACTIC | Transition between document sections or sources |

### TXC+BatchTopK+AuxK (Cycle H)
*Metrics: 0/32 semantic, coverage 7/7, judge disagreement 0.22*

| rank | feat_idx | peak | verdict | label |
|---|---|---|---|---|
| 1 | 14506 | fw_01 | SYNTACTIC | Beginning of sequence markers and punctuation |
| 2 | 3605 | fw_00 | SYNTACTIC | Beginning of new text sections or documents |
| 3 | 6768 | fw_05 | SYNTACTIC | Beginning of sentence or new text segment |
| 4 | 5308 | fw_00 | SYNTACTIC | Beginning of sentence capitalized words |
| 5 | 536 | fw_01 | UNKNOWN | Beginning of sentence or document markers |
| 6 | 3347 | fw_03 | SYNTACTIC | Beginning of sentence or document markers |
| 7 | 16265 | fw_03 | ⚠ SYNTACTIC | Transition between unrelated text segments |
| 8 | 12541 | fw_01 | SYNTACTIC | Beginning of sentence markers and punctuation |

### TFA
*Metrics: 0/32 semantic, coverage 2/7, judge disagreement 0.03*

| rank | feat_idx | peak | verdict | label |
|---|---|---|---|---|
| 1 | 2212 | fw_06 | SYNTACTIC | Beginning of text with title or heading |
| 2 | 17555 | fw_06 | SYNTACTIC | Beginning of text with title or heading |
| 3 | 1928 | fw_06 | ⚠ SYNTACTIC | Beginning of text with song or venue names |
| 4 | 7724 | fw_06 | SYNTACTIC | Beginning of text with title or heading |
| 5 | 17394 | fw_06 | SYNTACTIC | Beginning of text with song or venue title |
| 6 | 16936 | fw_06 | SYNTACTIC | Beginning of text with title or heading |
| 7 | 8406 | fw_06 | SYNTACTIC | Beginning of text with title or heading |
| 8 | 7160 | fw_06 | UNKNOWN | Beginning of text with title or heading |
