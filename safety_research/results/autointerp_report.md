# Autointerp report — StackedSAE vs TXCDR (T=5, Haiku 4.5)

Pairwise contrast between **StackedSAE (T=5)** and **TXCDR (T=5)** on the same 32-token chains. Both arms share T=5 and the same activation cache; they differ only in the encoder/decoder weight structure (block-diagonal vs full-rank across temporal positions).

Explainer: `claude-haiku-4-5-20251001` (async, concurrency=1, SDK retry-after on 429s). Special tokens render literally in the highlighted window so features that fire on `<bos>` / `<start_of_turn>` / `<end_of_turn>` are no longer mislabeled.

## Qualitative analysis: StackedSAE vs TXCDR

We elicit single-sentence explanations for every active feature in each arm via Claude Haiku 4.5 (8,454 StackedSAE, 5,033 TXCDR features) and observe three consistent qualitative differences. **First**, on per-sentence activation maps over 32-token sequences (see *Sentence-level case studies* below), TXCDR features fire as wide ~T-token diagonal bands that follow the natural span of the underlying concept — a dateline, a discourse-marker phrase, a noun-phrase boundary — while StackedSAE features fire as isolated single-position spikes co-located with the same concept's most informative token: TXCDR's full-rank cross-position weights bind activation to a temporal window, whereas StackedSAE's block-diagonal weights collapse it to a point. **Second**, embedding the explanations with `all-MiniLM-L6-v2` and clustering with HDBSCAN yields *k*=15 well-separated clusters for TXCDR (silhouette +0.01) versus *k*=23 tighter but heavily overlapping clusters for StackedSAE (silhouette −0.20); StackedSAE produces lexically homogeneous groupings around concrete entity types (acronyms, dates, geographic markers, sports headlines), while TXCDR additionally captures span-level discourse structure (first-person narrative openings, news article datelines, contrast/transition markers). **Third**, an LLM-judged temporal-coherence score over the cluster labels favors TXCDR (6.80 vs 4.78), consistent with the case-study geometry; StackedSAE wins on lexical coherence (7.08 vs 6.31), reflecting its tendency to memorize narrow token-level patterns rather than discourse-level abstractions. Safety-tag composition is near-identical between arms (≈99% `NONE`; ≈0.5% `HARMFUL_CONTENT`), so the architectural difference is concentrated in *what* kinds of patterns the dictionary discovers, not in how they are valenced.

## Headline numbers

| arm | n features | mean explanation length |
|-----|-----------:|-----------------------:|
| StackedSAE (T=5) | 8454 | 242 chars |
| TXCDR (T=5) | 5033 | 240 chars |

## Safety-tag distribution

![safety distribution](../figures/autointerp/safety_tag_distribution.png)

| arm | NONE | REFUSAL | DECEPTION | HARMFUL_CONTENT | BIAS | total |
|-----|-----:|--------:|----------:|----------------:|-----:|------:|
| StackedSAE (T=5) | 8390 (99.2%) | 3 (0.0%) | 2 (0.0%) | 46 (0.5%) | 13 (0.2%) | 8454 |
| TXCDR (T=5) | 4991 (99.2%) | 2 (0.0%) | 6 (0.1%) | 27 (0.5%) | 7 (0.1%) | 5033 |

## UMAP cluster meta-autointerp

Per-arm view: each feature's Haiku explanation is embedded with `sentence-transformers/all-MiniLM-L6-v2`, projected to 2D with UMAP, partitioned with HDBSCAN, and labeled lexically by distinctive content tokens.

Source: `safety_research/scripts/umap_meta.py`

### StackedSAE (T=5) — UMAP

`n=8454` features, `k=23` clusters, silhouette `-0.20`, mean cohesion `0.71`, noise frac `0.00%`.

![UMAP tsae](../figures/umap_tsae.png)

| cluster | n_feat | cohesion | safety mix | name | sample explanation |
|--------:|-------:|---------:|------------|------|---------------------|
| 0 | 7 | 0.90 | HARMFUL_CONTENT:7 | sexual · pornographic · adult · crude | This feature activates on sexually explicit pornographic content, particularly text fragments from adult webs… |
| 1 | 17 | 0.78 | NONE:17 | gratitude · thanks · appreciation · thank | This feature activates on gratitude expressions and discount/percentage statements embedded in commercial or … |
| 2 | 36 | 0.69 | NONE:36 | email · domain · addresses · urls | This feature activates on domain names and website URLs, particularly when they appear in contexts describing… |
| 3 | 8 | 0.73 | NONE:8 | take · verb · undertaking · action | This feature tracks the phrase "take" (or "took"/"takes") appearing in contexts where it functions as a pivot… |
| 4 | 13 | 0.77 | NONE:13 | there · existential · discourse · assertions | This feature activates on the opening clause "There is/are/was" constructions that introduce a topic, claim, … |
| 5 | 19 | 0.77 | NONE:19 | empty · cart · commerce · shopping | This feature activates on e-commerce interface text indicating an empty shopping cart or basket, typically ap… |
| 6 | 15 | 0.63 | NONE:15 | religious · divine · christian · spiritual | This feature activates on biblical or religious text passages, particularly those expressing transformative s… |
| 7 | 6 | 0.78 | NONE:6 | seems · looks · perception · subjective | This feature activates on phrases expressing apparent or perceived states using "seems," "looks like," or "se… |
| 8 | 8 | 0.76 | NONE:8 | pandemic · covid · coronavirus · disease | This feature tracks references to COVID-19, pandemics, and coronavirus-related content, activating strongly o… |
| 9 | 37 | 0.74 | NONE:36, HARMFUL_CONTENT:1 | greeting · informal · casual · guys | This feature activates on direct address transitions and conversational shifts where the speaker acknowledges… |

### TXCDR (T=5) — UMAP

`n=5033` features, `k=15` clusters, silhouette `+0.01`, mean cohesion `0.63`, noise frac `0.02%`.

![UMAP txc](../figures/umap_txc.png)

| cluster | n_feat | cohesion | safety mix | name | sample explanation |
|--------:|-------:|---------:|------------|------|---------------------|
| 0 | 17 | 0.63 | NONE:17 | ticker · financial · stock · nyse | This feature detects financial analyst rating statements, specifically phrases indicating positive stock reco… |
| 1 | 117 | 0.67 | NONE:117 | news · location · dateline · article | This feature activates on the beginning of news article headlines and article openings, particularly those wi… |
| 2 | 6 | 0.72 | NONE:6 | death · dead · dying · died | This feature activates on legal definitions related to "dying declarations" in evidence law, particularly the… |
| 3 | 15 | 0.69 | NONE:15 | year · activates · numerical · particularly | This feature activates on age descriptors and demographic identifiers in text, particularly phrases containin… |
| 4 | 46 | 0.66 | NONE:46 | acronyms · acronym · abbreviated · full | This feature detects abbreviations and acronyms followed by periods in business/legal document text, particul… |
| 5 | 56 | 0.57 | NONE:56 | sports · team · particularly · activates | This feature activates on numerical vote counts and their surrounding context in legislative or judicial deci… |
| 6 | 363 | 0.52 | NONE:362, HARMFUL_CONTENT:1 | numerical · numbers · activates · particularly | This feature detects numeric quantities, measurements, specifications, and product/property descriptors that … |
| 7 | 7 | 0.83 | NONE:7 | blog · blogging · blogs · platforms | This feature detects text discussing blog posts, blogging activities, and blog-related metadata (posting freq… |
| 8 | 656 | 0.58 | NONE:656 | date · temporal · time · activates | This feature activates on temporal and date-related information, particularly dates, times, year ranges, and … |
| 9 | 22 | 0.53 | NONE:21, HARMFUL_CONTENT:1 | biblical · verse · chapter · religious | This feature activates on biblical references and citations, particularly when scripture passages (book names… |

### Cross-arm cluster metrics

![cluster metrics](../figures/umap_cluster_metrics.png)

![safety composition](../figures/umap_safety_composition.png)

## Sentence-level case studies

Five 32-token sequences, top-32 features per arm chosen via the **exclusive** selection mode (each token position claims its most-concentrated feature; greedy assignment, no feature reused).

### Top-feature selection procedure

For each token position `p` in the 32-token sequence, pick the feature `j` whose total activation mass is most concentrated at `p`:

```
score[p, j] = acts[p, j]^2 / sum_p(acts[p, j])
```

Squaring forces a real strong activation at `p` (not just rare elsewhere). Greedy assignment: positions sorted in descending order of their best score; each feature gets claimed by at most one position, so the strongest position-feature pairs win first. Result: 32 features, one per token position, each fingerprinted to that position.

Code: `temporal_crosscoders/NLP/sentence.py:307–339` (`select_exclusive_features`). Magnitude-mode alternative (`top-k by sum |activation|`): same file, lines 478–479.

### Chain 12345

![chain 12345](../../temporal_crosscoders/NLP/viz_outputs/sentence_case_studies/sentence_mid_res_k100_T5_chain12345_exclusive.png)

```text
Source: chain12345 | Layer: mid_res | k=100 T=5
Selection: 32 features per-model (one most-exclusive per token position)
All metrics computed LOCALLY on this 32-token sequence.

StackedSAE:
  Local position entropy:    0.3312 +/- 0.1079
  Local feature sparsity:    0.0635
  Local active feats/pos:    431.1 (full D_SAE)
  Local frac nonzero:        0.0635
  Local max activation:      371.83
  Local mean activation (>0):10.73

TXCDR:
  Local position entropy:    0.9702 +/- 0.0521
  Local feature sparsity:    0.4580
  Local active feats/pos:    868.8 (full D_SAE)
  Local frac nonzero:        0.4580
  Local max activation:      834.21
  Local mean activation (>0):29.79
```

### Chain 137

![chain 137](../../temporal_crosscoders/NLP/viz_outputs/sentence_case_studies/sentence_mid_res_k100_T5_chain137_exclusive.png)

```text
Source: chain137 | Layer: mid_res | k=100 T=5
Selection: 32 features per-model (one most-exclusive per token position)
All metrics computed LOCALLY on this 32-token sequence.

StackedSAE:
  Local position entropy:    0.3015 +/- 0.1207
  Local feature sparsity:    0.0566
  Local active feats/pos:    432.3 (full D_SAE)
  Local frac nonzero:        0.0566
  Local max activation:      376.68
  Local mean activation (>0):11.67

TXCDR:
  Local position entropy:    0.9613 +/- 0.0629
  Local feature sparsity:    0.4229
  Local active feats/pos:    900.3 (full D_SAE)
  Local frac nonzero:        0.4229
  Local max activation:      662.78
  Local mean activation (>0):25.25
```

### Chain 16921

![chain 16921](../../temporal_crosscoders/NLP/viz_outputs/sentence_case_studies/sentence_mid_res_k100_T5_chain16921_exclusive.png)

```text
Source: chain16921 | Layer: mid_res | k=100 T=5
Selection: 32 features per-model (one most-exclusive per token position)
All metrics computed LOCALLY on this 32-token sequence.

StackedSAE:
  Local position entropy:    0.2897 +/- 0.1167
  Local feature sparsity:    0.0537
  Local active feats/pos:    432.8 (full D_SAE)
  Local frac nonzero:        0.0537
  Local max activation:      376.68
  Local mean activation (>0):12.30

TXCDR:
  Local position entropy:    0.9853 +/- 0.0275
  Local feature sparsity:    0.5078
  Local active feats/pos:    865.7 (full D_SAE)
  Local frac nonzero:        0.5078
  Local max activation:      746.98
  Local mean activation (>0):26.62
```

### Chain 4242

![chain 4242](../../temporal_crosscoders/NLP/viz_outputs/sentence_case_studies/sentence_mid_res_k100_T5_chain4242_exclusive.png)

```text
Source: chain4242 | Layer: mid_res | k=100 T=5
Selection: 32 features per-model (one most-exclusive per token position)
All metrics computed LOCALLY on this 32-token sequence.

StackedSAE:
  Local position entropy:    0.4064 +/- 0.1104
  Local feature sparsity:    0.0840
  Local active feats/pos:    432.3 (full D_SAE)
  Local frac nonzero:        0.0840
  Local max activation:      376.68
  Local mean activation (>0):9.32

TXCDR:
  Local position entropy:    0.9491 +/- 0.0458
  Local feature sparsity:    0.5957
  Local active feats/pos:    894.9 (full D_SAE)
  Local frac nonzero:        0.5957
  Local max activation:      764.46
  Local mean activation (>0):22.73
```

### Chain 42

![chain 42](../../temporal_crosscoders/NLP/viz_outputs/sentence_case_studies/sentence_mid_res_k100_T5_chain42_exclusive.png)

```text
Source: chain42 | Layer: mid_res | k=100 T=5
Selection: 32 features per-model (one most-exclusive per token position)
All metrics computed LOCALLY on this 32-token sequence.

StackedSAE:
  Local position entropy:    0.3276 +/- 0.1082
  Local feature sparsity:    0.0625
  Local active feats/pos:    432.4 (full D_SAE)
  Local frac nonzero:        0.0625
  Local max activation:      376.68
  Local mean activation (>0):10.71

TXCDR:
  Local position entropy:    0.9641 +/- 0.0467
  Local feature sparsity:    0.5625
  Local active feats/pos:    881.1 (full D_SAE)
  Local frac nonzero:        0.5625
  Local max activation:      837.00
  Local mean activation (>0):25.06
```

## Top-12 most-active features per arm

Features ranked by total activation mass across the 1,500-chain scan. Two example windows shown per feature.

### StackedSAE (T=5) — top-12

| feat | safety | explanation | top windows |
|------|--------|-------------|-------------|
| 7281 | NONE | This feature activates on conversational discourse markers and transitional phrases that establish the beginning of informal blog posts, pe… | `[FOCUS]<bos>Time for another entry[/FOCUS] in Friday Fictioneers challenge, courtesy of R…` <br> `[FOCUS]<bos>another of my tricks[/FOCUS] to pretend that new england winter is not even h…` |
| 14421 | NONE | This feature activates on incomplete or truncated text segments where content is cut off mid-word or mid-phrase, typically at natural break… | `[FOCUS]<bos>Small Ball Acro[/FOCUS]pora 5.5" x 4.` <br> `[FOCUS]<bos>Partner With Alog[/FOCUS]ent Our partnerships make us all more successful—` |
| 12950 | NONE | This feature activates on numerical ranges, dates, and age specifications marked by hyphens or number-dash patterns that segment time perio… | `[FOCUS]<bos>A 31[/FOCUS]-year-old member asked: phimosis` <br> `[FOCUS]<bos>October 3-[/FOCUS]8, 2022 Alumni and` |
| 2973 | NONE | This feature activates on direct address transitions and conversational shifts where the speaker acknowledges or reorients toward an audien… | `[FOCUS]<bos>Tough also noted the[/FOCUS] much of what the company is doing now is becoming` <br> `[FOCUS]<bos>Hej! You have[/FOCUS] found our bottle? Please send a message telling the` |
| 8951 | NONE | This feature activates on text fragments immediately before contextual breaks, metadata boundaries, or topic shifts—detecting positions whe… | `[FOCUS]<bos>Small Ball Acro[/FOCUS]pora 5.5" x 4.` <br> `[FOCUS]<bos>Partner With Alog[/FOCUS]ent Our partnerships make us all more successful—` |
| 1946 | NONE | This feature activates on text segments immediately following the beginning-of-sequence token where content is abruptly truncated or cut mi… | `[FOCUS]<bos>Small Ball Acro[/FOCUS]pora 5.5" x 4.` <br> `[FOCUS]<bos>Partner With Alog[/FOCUS]ent Our partnerships make us all more successful—` |
| 6148 | NONE | This feature activates on abrupt mid-word or mid-phrase text truncations where the token stream cuts off before grammatical completion, typ… | `[FOCUS]<bos>Small Ball Acro[/FOCUS]pora 5.5" x 4.` <br> `[FOCUS]<bos>Partner With Alog[/FOCUS]ent Our partnerships make us all more successful—` |
| 6654 | NONE | This feature detects the beginning of text segments or documents, particularly marking the transition from a beginning-of-sequence token to… | `[FOCUS]<bos>Small Ball Acro[/FOCUS]pora 5.5" x 4.` <br> `[FOCUS]<bos>Partner With Alog[/FOCUS]ent Our partnerships make us all more successful—` |
| 17176 | NONE | This feature activates on the beginning-of-sequence token followed by diverse opening phrases across different document types (product list… | `[FOCUS]<bos>Small Ball Acro[/FOCUS]pora 5.5" x 4.` <br> `[FOCUS]<bos>Partner With Alog[/FOCUS]ent Our partnerships make us all more successful—` |
| 1972 | NONE | This feature detects the onset of specific factual or descriptive content immediately after the beginning-of-sequence token, capturing the … | `[FOCUS]<bos>Spend $50[/FOCUS] more and get free shipping! Your cart is` <br> `[FOCUS]<bos>Amazon Price:$2[/FOCUS]4.99(as of August 2` |
| 17168 | NONE | This feature activates on title-like or headline text patterns that introduce topics, products, or content sections—typically appearing aft… | `[FOCUS]<bos>Summer heat waves in[/FOCUS] Santiago, just like anywhere else, mean one thing` <br> `[FOCUS]<bos>Latest Razer Blade Gets[/FOCUS] Outfitted with More Potent Gaming Hardware, C…` |
| 10085 | NONE | This feature activates on sentence fragments or incomplete phrases that end mid-clause with a capital letter or topic shift following, typi… | `[FOCUS]<bos>Our favourite picks from[/FOCUS] Net-a-porter Everybody’s favourite` <br> `[FOCUS]<bos>Drive economic development through[/FOCUS] high-speed networks An end-to-` |

### TXCDR (T=5) — top-12

| feat | safety | explanation | top windows |
|------|--------|-------------|-------------|
| 1004 | NONE | This feature activates on first-person narrative openings that transition from self-description or personal statements into specific conten… | `[FOCUS]<bos>I have a somewhat[/FOCUS] fancy tv that supports an external wi-fi module` <br> `[FOCUS]<bos>I have to say[/FOCUS] that the speakers at the Science and Society Conference,` |
| 2720 | NONE | This feature activates on temporal and date-related information, particularly dates, times, year ranges, and temporal markers that appear a… | `[FOCUS]<bos>Date: Monday [/FOCUS]11 April, 2016` <br> `[FOCUS]<bos>9 April 2[/FOCUS]019 - "Here I am, send` |
| 13249 | NONE | This feature tracks the beginning of news articles, web content headers, and published text fragments—specifically detecting the opening to… | `[FOCUS]<bos>Sault Ste.[/FOCUS] Marie hot stone spa Jump to. Accessibility Help` <br> `[FOCUS]<bos>The Communications Decency[/FOCUS] Act of 1996 (CDA)` |
| 11361 | NONE | This feature activates on proper nouns and branded names (company names, product titles, band names, game titles) that appear at the beginn… | `[FOCUS]<bos>At LuvBuds[/FOCUS] we strive to be ahead of the curve when buying` <br> `[FOCUS]<bos>Cinch Connectivity Solutions[/FOCUS] (CCS) has been named the 20` |
| 2890 | NONE | This feature activates on search query titles, webpage headers, and content introductions that appear at the beginning of documents—essenti… | `[FOCUS]<bos>Apple cake using apple[/FOCUS] pie filling Recipes / Apple cake using apple p…` <br> `[FOCUS]<bos>Personal Loan In Chennai[/FOCUS] The capital city of Tamil Nadu, Chennai lies` |
| 7347 | NONE | This feature activates on the beginning of news article headlines and article openings, particularly those with location tags, datelines, o… | `[FOCUS]<bos>Personal Growth - Make[/FOCUS] a habit of it ! Ashish Virmani` <br> `[FOCUS]<bos>In Peru, a[/FOCUS] suspecting husband filmed his own wife in bed with` |
| 18396 | NONE | This feature activates on the beginning of straightforward, declarative statements that introduce a topic or subject with a neutral, inform… | `[FOCUS]<bos>Ballet is an[/FOCUS] artistic dance form performed to music, using precise and` <br> `[FOCUS]<bos>Romance novels are known[/FOCUS] for heaving bosoms, but these photos from Pe…` |
| 1511 | NONE | This feature activates on the beginning of sentences that introduce specific named entities or proper nouns (organizations, places, people,… | `[FOCUS]<bos>The Quad Cities area[/FOCUS] is blessed with two local mosques or masajids.` <br> `[FOCUS]<bos>The FiRa Consortium[/FOCUS] has just been established by the ASSA ABLOY` |
| 15506 | NONE | This feature activates on text segments immediately following beginning-of-sequence tokens or natural sentence breaks, capturing the onset … | `[FOCUS]<bos>Your greatest asset in[/FOCUS] life is your Health. Immediate cover when you` <br> `[FOCUS]<bos>There are instances when[/FOCUS] a person gets injured because of the neglige…` |
| 17051 | NONE | This feature detects editorial and meta-textual framing statements that introduce content restrictions, format specifications, source attri… | `[FOCUS]<bos>Letter to the editor[/FOCUS] – vote no on Question 2 I am` <br> `[FOCUS]<bos>The summary should be[/FOCUS] 3 pages long. 5-6 body` |
| 4278 | NONE | This feature activates on the beginning-of-sequence token (<bos>) followed by introductory or framing phrases that establish context, annou… | `[FOCUS]<bos>Just how to Compose[/FOCUS] the Excellent Essay Intro Writing an ideal essay …` <br> `[FOCUS]<bos>You are not logged[/FOCUS] in. (Log in • Create account)` |
| 16845 | NONE | This feature activates on product titles, headings, and document headers that are cut off or truncated mid-word or mid-phrase, typically ap… | `[FOCUS]<bos>Natec Lobster -[/FOCUS] notebook security cable: convenient code operated bar…` <br> `[FOCUS]<bos>Endoscope Repro[/FOCUS]cessing and Infection Control - An endoscope` |

## Random sample (mid-dictionary)

20 features drawn at random from each arm's full Haiku-interpreted set, to spot-check explanation quality outside the head of the ranking.

### StackedSAE (T=5) — 20 random

| feat | safety | explanation | top windows |
|------|--------|-------------|-------------|
| 10703 | NONE | This feature activates on conditional or intentional constructions expressing desire or purpose, particularly patterns like "want to" or "w… | `for The Beginner Network Marketer By Michael Smith[FOCUS] So you want to[/FOCUS] know the…` <br> `[FOCUS]<bos>If you want to[/FOCUS] buy a used Chevrolet Corvette and are looking for one` |
| 16873 | NONE | This feature detects numerical values or measurements that appear in close proximity to descriptive text, often marking quantities like dis… | `May 13, 2010[FOCUS] Photos 36[/FOCUS]5 week 19 Rachel and Philip went` <br> `<bos>A free, weekly[FOCUS], timed 5k[/FOCUS] walk/jog/run 9:30` |
| 3051 | NONE | This feature detects the beginning of diverse text formats and sources—including conversational prompts, email list headers, tweet citation… | `[FOCUS]<bos>Have you been struggling[/FOCUS] with how to talk to your tween about sex?` <br> `[FOCUS]<bos>Have you ever heard[/FOCUS] someone talk about Recovery Month and wondered wh…` |
| 9979 | NONE | This feature detects modifiers describing scale, size, or scope—particularly adjectives like "small," "tiny," "budget-friendly," "nano," an… | `, 4th Edition - n. A[FOCUS] small ball of ground meat[/FOCUS] variously seasoned and cook…` <br> `to join the Apple family is the iPad Mini.[FOCUS] Smaller, yes but not[/FOCUS] in the lea…` |
| 11151 | NONE | This feature detects the boundary pattern of author/byline attribution in web content, specifically the transition from article title or co… | `<bos>Photo: Vivid Images/[FOCUS]Getty Images By Amy[/FOCUS] Osmond Cook When it comes to …` <br> `the following prompt to the staff: "If my[FOCUS] students can __________ by the[/FOCUS] e…` |
| 16286 | NONE | This feature activates on dates and temporal markers (specific dates, day-of-week references, "For Immediate Release") that appear at or ne… | `August 29, 2012[FOCUS] Super talents at Serra[/FOCUS] USC has had quite a run landing pla…` <br> `May 29, 2017[FOCUS] Buy leased building?[/FOCUS] I’ve operated my own small business for` |
| 2164 | NONE | This feature activates on phrases and clauses that establish geographic location or place-specific context, particularly when describing wh… | `<bos[FOCUS]>"Where I live now[/FOCUS]." Top 5 Page for this destination Carson by` <br> `Leeuwarden-Fryslan, one of[FOCUS] the less populated parts of[/FOCUS] the Netherlands, ha…` |
| 3425 | NONE | This feature activates on passages from Christian religious texts, particularly Galatians 2:20 and similar scriptural verses that express t… | `<bos>Cru[FOCUS]cified With Christ [/FOCUS]by Nan Doud, Guest Writer I have` <br> `Christ by Nan Doud, Guest Writer [FOCUS]I have been crucified with[/FOCUS] Christ. It is …` |
| 16980 | NONE | This feature activates on phrases expressing desire, intention, or volition using constructions like "want to," "wanted to," and "you want … | `<bos>Advice for your farm/nursery[FOCUS] Do you want to[/FOCUS] start a farm/nursery? Do …` <br> `such a day of unanticipated and special memories.[FOCUS] Obviously, you want to[/FOCUS] s…` |
| 7712 | NONE | This feature activates on decimal points and numeric separators appearing within larger numbers, particularly in measurement values, coordi… | `6' 4" (193[FOCUS].04 cm)[/FOCUS] Standing at` <br> `<bos>Latitude: 34.2[FOCUS]54700[/FOCUS] * Longitude: -89.872` |
| 11606 | NONE | This feature tracks contrastive or pivoting phrases that transition between two related ideas or statements, often marked by conjunctions l… | `, please choose a 18650[FOCUS] battery option in the drop[/FOCUS] down list above.(detail…` <br> `project quote without entering your home! As an[FOCUS] Essential Business, we are[/FOCUS]…` |
| 887 | NONE | This feature tracks the linguistic pattern of conjunctions and connectors that link two related clauses or ideas, particularly "and" constr… | `you guys! shaper for inbound traffic and[FOCUS] outbound traffic and it works[/FOCUS] so …` <br> `13 – A major field study by the[FOCUS] University of Texas and sponsored[/FOCUS] by the E…` |
| 16066 | NONE | This feature detects contractions and colloquial compressed forms (It's, There's, Let's, won't, that's) that appear at clause or sentence b… | `[FOCUS]<bos>It's giveaway[/FOCUS] time! I've been talking about doing` <br> `[FOCUS]<bos>It's always[/FOCUS] inadvisable to bite the hand that feeds you` |
| 2993 | NONE | This feature activates on transitional phrases and discourse markers that introduce elaboration, contrast, or continuation—typically appear… | `Unified Development for Web, Mobile, and Embedded Applications[FOCUS] WebAssembly is more…` <br> `need to be supported by a expense claim form.[FOCUS] Together with attached invoices[/FOC…` |
| 4548 | NONE | This feature activates on conjunctions and commas that coordinate multiple related concepts, attributes, or items within a list or paired c… | `first year as a mother — was a blur of[FOCUS] wonder, exhaustion and anxiety[/FOCUS] for …` <br> `two-day event celebrating the convergence of online technology[FOCUS], creativity, and em…` |
| 13256 | NONE | This feature activates on text beginnings that introduce or present informational content, particularly opening phrases like "This is," "We… | `[FOCUS]<bos>This is the product[/FOCUS] page for: Black Stud Shoulder Jumper Image carous…` <br> `[FOCUS]<bos>This is the first[/FOCUS] book in the new Urban Fantasy series by Candace B` |
| 3742 | NONE | This feature activates on descriptive phrases that highlight positive, desirable, or aspirational qualities—often used in marketing, instru… | `heavy A Panorama is defined as a picture or[FOCUS] photograph containing a wide view[/FOC…` <br> `-cut Italian microfiber with custom engineered lace for a[FOCUS] high rise, minimal cover…` |
| 15743 | NONE | This feature tracks transitions between named positions/titles and the individuals holding them, particularly in contexts describing person… | `looking to replace Rahm Emanuel as your chief of[FOCUS] staff. I would[/FOCUS] like to hu…` <br> `be looking to replace Rahm Emanuel as your chief[FOCUS] of staff. I[/FOCUS] would like to…` |
| 9121 | NONE | This feature activates on listicle and enumeration patterns, particularly titles or headers that reference numbered collections, rankings, … | `<bos[FOCUS]>The Five Best Concerts in[/FOCUS] L.A. This Weekend Friday, July` <br> `<bos>Hofstede canada vs japan 10[FOCUS] cultural contrasts between us &[/FOCUS] japanese …` |
| 2393 | NONE | This feature activates on proper nouns and named entities (people, organizations, products, places) that appear immediately after discourse… | `<bos>Fans are having[FOCUS] fun keeping up with Kendall[/FOCUS] Jenner's culinary skills.…` <br> `<bos>Believe[FOCUS] it or not, Manfred[/FOCUS] von Richthofen — AKA the Red Baron,` |

### TXCDR (T=5) — 20 random

| feat | safety | explanation | top windows |
|------|--------|-------------|-------------|
| 12195 | NONE | This feature activates on timestamp tokens, specifically time-of-day components (hours and minutes in 24-hour or 12-hour format) that appea… | `4-01-2014 [FOCUS]05:20[/FOCUS] PM\| RCS along with most dwarf shrimp in` <br> `1-17-2011 [FOCUS]01:29[/FOCUS] PM I´m very disappointed at the moment` |
| 8966 | NONE | This feature activates on the phrase "as" or "as...as" used as a comparative conjunction or introductory clause connector, particularly in … | `Register Help In. Meet thousands of local Teme[FOCUS]cula singles, as the[/FOCUS] worlds …` <br> `<bos>We saw lots of other breeds,[FOCUS] common and uncommon as we[/FOCUS] walked around.…` |
| 6093 | NONE | This feature activates on proper nouns (names of people, places, or titles) that appear in news article contexts, particularly when framed … | `2009 with some cash on hand,[FOCUS] and Gov. David Paterson[/FOCUS] said local aid paymen…` <br> `Conference SAN FRANCISCO (KCBS / AP)[FOCUS] — Gov. Jerry Brown[/FOCUS] told a green build…` |
| 1540 | NONE | This feature activates on text segments that appear between content breaks or formatting delimiters, often marking transitions between phra… | `Store Day Guide – a free 40-[FOCUS]page magazine bringing you the[/FOCUS] lowdown` <br> `Live Vinyl comes complete with the official Record Store Day[FOCUS] Guide – a free [/FOCU…` |
| 9738 | NONE | This feature activates on hyphenated numerical descriptors that specify dimensions, durations, ages, or measurements (e.g., "17-year-old," … | `<bos>A [FOCUS]17-year-[/FOCUS]old girl is in police custody following a non-` <br> `<bos>A [FOCUS]31-year-[/FOCUS]old member asked: phimosis and paraphi` |
| 7594 | NONE | This feature detects text passages about educational institutions and academic settings (colleges, law schools, high schools, engineering p… | `<bos>Hi my name is Tiffany and I am[FOCUS] in college. I found[/FOCUS] this site through …` <br> `it is no secret that whilst in our bogart[FOCUS]ing college days, I[/FOCUS] brought my du…` |
| 6494 | NONE | This feature activates on phrases expressing positive personal experiences or fortunate circumstances, typically using formulaic language l… | `Dos for Every Entrepreneur This Thursday, I had[FOCUS] the opportunity to present a[/FOCU…` <br> `<bos>In February I had[FOCUS] the opportunity to spend some[/FOCUS] time with some Milepo…` |
| 2260 | NONE | This feature activates on phrases containing "on" or similar prepositions that bridge two clauses or separate a main statement from a tempo… | `sport where dogs recognize and follow a specific human'[FOCUS]s scent on the ground[/FOCU…` <br> `, John. I had a book illustrated by him[FOCUS] and voiced on a tape[/FOCUS] by Mel Blanc …` |
| 11974 | NONE | This feature activates on date and time ranges, particularly when a hyphen, dash, or "to" separates the start and end points of a temporal … | `time, Security Essen from September 25 to[FOCUS] 28 will take[/FOCUS] place in the modern…` <br> `, July 27, and Sunday, July[FOCUS] 28, so[/FOCUS] it can share the benefits` |
| 9627 | NONE | This feature activates on formal titles, proper nouns, and organizational/institutional names that typically appear at the beginning of new… | `[FOCUS]<bos>Verizon Communications Inc.[/FOCUS] said Tuesday it has committed to a three-…` <br> `[FOCUS]<bos>Pubs and Clubs welcomes[/FOCUS] Entertainment Kapooka - NSW, find places to` |
| 14657 | NONE | This feature detects legal/policy disclosure language, particularly phrases about terms, conditions, and privacy statements that introduce … | `we would like to give you some valuable information concerning[FOCUS] our terms and condi…` <br> `sites. When visiting those sites, your information is[FOCUS] governed by their privacy st…` |
| 16085 | NONE | This feature activates on locations, team names, and institutional identifiers in sports article headers and bylines, particularly at phras… | `Sports Article JOHNSONBURG - The St.[FOCUS] Marys Area Flying Dutch and[/FOCUS] the Elk C…` <br> `Article JOHNSONBURG - The St. Marys[FOCUS] Area Flying Dutch and the[/FOCUS] Elk County` |
| 4894 | NONE | This feature activates on sequences where multiple professional titles, credentials, or descriptive qualifications are listed in apposition… | `Amazon Leading digital marketing consultant, Jason Ciment[FOCUS], a CPA, attorney[/FOCUS]…` <br> `Yi, chief of the Ren Ci hospital and a[FOCUS] prominent religious leader, has[/FOCUS] been` |
| 16128 | NONE | This feature activates on numeric or alphanumeric suffixes and sequential identifiers that appear at boundaries (room numbers, suite number… | `Williams Western Realty 1000 SE Everett[FOCUS] Mall Way Ste 2[/FOCUS]01 Everett, WA 98` <br> `<bos>Violence in Video Games: GameSkinny Round[FOCUS] Table Podcast Ep.1[/FOCUS]2 Welcome…` |
| 7060 | NONE | This feature activates on contrastive or pivotal phrases that mark shifts in narrative or argument, typically introduced by conjunctions or… | `. However, because of their simplicity, traders often[FOCUS] overlook them. By using[/FOC…` <br> `<bos>Sometimes we forget to look around -[FOCUS] smell the flowers - and[/FOCUS] enjoy th…` |
| 17248 | NONE | This feature tracks temporal hedging and qualifications in delivery/processing timelines—specifically language that softens concrete timefr… | `via registered post, however please allow for additional processing[FOCUS] days during bu…` <br> `3 business days via registered post, however please allow[FOCUS] for additional processin…` |
| 6128 | NONE | This feature activates on text passages containing proper nouns, formal titles, institutional affiliations, and named entities (organizatio… | `praise Chinese President Xi Jinping chairs the 1[FOCUS]8th Meeting of the[/FOCUS] Council…` <br> `cientist affiliated with the Nuffield Department of[FOCUS] Clinical Neurosciences at Oxfo…` |
| 15425 | NONE | This feature activates on explanatory phrases that introduce or clarify the primary/foundational definition of a word or concept, particula… | `Principal has several different meanings. It most commonly pertains[FOCUS] to the initial…` <br> `<bos>Principal has several different meanings. It most[FOCUS] commonly pertains to the in…` |
| 15524 | NONE | This feature activates on references to third-party entities, vendors, or external organizations mentioned in contexts involving services, … | `<bos>SteamFirst[FOCUS] uses third-party advertising[/FOCUS] companies to serve ads when y…` <br> `). TD Ameritrade, Inc., and[FOCUS] all third-party companies[/FOCUS]` |
| 3128 | NONE | This feature activates on sentence fragments or truncated phrases where text has been cut off mid-word or mid-clause, particularly at natur… | `<bos[FOCUS]>8. VMware Adds Nic[/FOCUS]ira For $1.2 Billion VMware` <br> `<bos[FOCUS]>I need you change few[/FOCUS] task in Gold Coders script also template .` |

## Topic vocabulary diff (Haiku explanations)

Top distinctive content words per arm — coarse summary of what concepts each arm's dictionary tends to label.

| rank | StackedSAE (T=5) | count | TXCDR (T=5) | count |
|-----:|------------------|------:|-------------|------:|
| 1 | phrases | 3451 | phrases | 1980 |
| 2 | where | 2229 | where | 1211 |
| 3 | between | 1914 | between | 1015 |
| 4 | like | 1570 | like | 874 |
| 5 | boundaries | 1371 | descriptive | 769 |
| 6 | descriptive | 1341 | boundaries | 754 |
| 7 | product | 1332 | when | 746 |
| 8 | contexts | 1310 | names | 737 |
| 9 | names | 1299 | contexts | 718 |
| 10 | when | 1245 | product | 690 |
| 11 | markers | 1149 | markers | 623 |
| 12 | transitions | 1033 | within | 555 |
| 13 | within | 991 | temporal | 553 |
| 14 | temporal | 938 | introduce | 529 |
| 15 | introduce | 861 | transitions | 490 |
| 16 | information | 841 | information | 465 |
| 17 | language | 774 | specifically | 462 |
| 18 | appearing | 759 | news | 456 |
| 19 | clauses | 711 | language | 445 |
| 20 | narrative | 702 | statements | 440 |

## Explanation verbosity

![explanation length](../figures/autointerp/explanation_length.png)
