---
author: OpenAI
date: 2026-04-18
tags:
  - proposal
  - experiment
  - python-code
  - temporal-crosscoder
  - mlc
  - benchmark
---

# Python code benchmark plan — TempXC vs MLC

## TL;DR

Run two matched experiments on **Gemma-2B-it activations**:

1. **Broad Python corpus.** Train SAE, MLC, and TempXC on a large Python code corpus, then inspect them on both broad held-out code and a small AST-curated motif set.
2. **Curated function-definition corpus.** Train the same architectures on Python functions selected to make short-horizon temporal structure salient, then evaluate on AST-defined code tasks, reconstruction loss, and autointerp.

The key hypothesis is deliberately narrow:

> **If TempXC has a real niche, it should show up on short, local, phase-structured code motifs, not on generic code understanding.**

So the benchmark on which TempXC should win is **not** a broad semantic benchmark like docstring generation or generic code/topic probing. It is an **AST-curated local motif benchmark** built from patterns such as:

- `with ... as x:` followed by early uses of `x`
- `except E as e:` followed by uses of `e` in the handler
- `for x in xs:` followed by early uses of `x`
- `import foo as bar` followed by nearby `bar.` calls or attribute access

These are the Python cases where one latent cause plausibly appears in multiple nearby roles across a short token window.

---

## 1. Core question

Can a temporal crosscoder trained on Python activations recover **short-range temporal motifs** that are harder to isolate with:

- a **single-token SAE**, and
- a **multi-layer crosscoder (MLC)** that reads same-token structure across layers?

This is intentionally narrower than “does temporal information help in general?”

The specific scientific question is:

> **Does crosscoding across sequence positions help when the object of interest is a short, reusable, phase-structured code motif?**

Examples include “bind symbol → use symbol,” “enter scoped block → operate on bound handle,” or “bind alias → call through alias.”

If the answer is no even in this setting, then TempXC probably does not have a clear niche relative to MLC.

---

## 2. Why Python is the right code setting

Python is a good first target because it gives us:

- **clean AST structure** for `FunctionDef`, `With`, `Try`, `ExceptHandler`, `For`, `Import`, and `ImportFrom`
- **obvious scoped binding/use motifs** that happen within short windows
- a mix of **header/body phase structure** and **symbol reuse**
- easy automatic curation of positive and negative examples

That makes Python an attractive “middle step” between synthetic data and fully unconstrained natural text.

The most promising motifs are not plain `def` or plain `import` by themselves. They are **binding/use motifs** inside a local scope, where the same symbol appears in different roles over nearby positions.

---

## 3. Architectures to compare

Use the same three architectures as in the existing repo work:

### 3.1 SAE
- Single-token TopK SAE
- Input: one token activation at one layer
- Baseline for “does time help at all?”

### 3.2 MLC
- Multi-layer crosscoder over a 5-layer window around the anchor layer
- Same token, multiple layers
- Baseline for “same weak model, but reading the stronger model’s compressed cross-layer representation”

### 3.3 TempXC
- Temporal crosscoder over a short token window at a single layer
- Same layer, multiple sequence positions
- Architecture under test

### 3.4 Matching protocol

Start with the **same comparison scaffold as the current SAEBench experiment** so the new results are directly comparable:

- same subject model: **Gemma-2B-it**
- same anchor layer: **layer 12**
- same MLC layer window: **layers 10–14**
- same `d_sae`, optimizer, scheduler, and token budget
- same TopK family unless there is a strong reason to deviate

Primary comparison should be at **TempXC T=5**, since that is the cleanest apples-to-apples comparison with the existing code. Only run larger `T` once the motif benchmark shows some actual signal.

---

## 4. Dataset plan

We want two data tracks.

## 4A. Broad Python corpus

### Goal
Train all architectures on a realistic Python distribution, then inspect whether TempXC learns anything distinct before moving to a more targeted benchmark.

### Primary training corpus
**The Stack, Python subset**.

Why:
- broad and diverse Python code
- large enough for stable feature learning
- direct file content is available
- easier first pass than The Stack v2

### Practical note
Do **not** use The Stack v2 as the default first choice for activation harvesting. The current v2 release primarily exposes **Software Heritage IDs** and download workflow rather than directly serving file contents, which makes it less convenient for a fast first experiment.

### Optional fallback
If loading The Stack Python becomes annoying, use a directly loadable Python slice from the BigCode ecosystem as a fallback. The point of this track is breadth, not a particular brand name.

### Preprocessing
Filter the raw Python files to keep training sane:

- require `ast.parse` success
- drop files that are too short to contain useful structure
- drop extremely long files for the first pass
- split by repository when possible to reduce train/eval leakage
- chunk into fixed token windows with overlap

Reasonable first-pass defaults:

- 128–512 model tokens per chunk
- stride 64–256
- hold out a repo-level validation slice

### What this track is for
This is the **broad distribution** track. It tells us:

- whether TempXC learns anything obviously distinct on code at scale
- whether code looks more favorable than web text in the shuffle / motif sense
- whether MLC still dominates once the domain is shifted from generic text to code

It is **not** the track on which I would expect TempXC to clearly beat MLC.

---

## 4B. Curated function-definition corpus

### Goal
Construct a function-level dataset where local temporal motifs are common and automatically labelable.

### Primary source pool
Use one of these as the source pool for functions:

1. **CodeSearchNet Python** (preferred when docstrings/metadata help autointerp)
2. **`bigcode/python-stack-v1-functions-filtered`** (preferred when we want a pure, large function-only corpus)

### Why these are useful
They give us function-shaped training examples instead of arbitrary file chunks, which makes it much easier to:

- align windows to headers and early body statements
- stratify by AST motif type
- use docstrings and function names as weak semantic metadata
- inspect features at the function level rather than over arbitrary source slices

### Two versions of the function track

#### Track B1: broad function-def training
Train on a large random sample of parseable Python functions.

Purpose:
- learn generic function-level features
- ask whether TempXC improves when the data distribution is narrowed from “all Python files” to “Python functions”

#### Track B2: motif-enriched function training
Train on a function corpus enriched for local scoped-binding motifs.

For example, oversample functions containing at least one of:

- `with ... as`
- `except ... as`
- `for <name> in ...`
- `import ... as ...`
- `from ... import ... as ...`

Purpose:
- maximize the chance that TempXC sees the kind of short trajectory atom it is supposed to model
- test a matched regime rather than hoping the broad corpus gives it one by accident

### Suggested curation rules
Keep only functions that are:

- parseable by Python `ast`
- not tiny one-liners unless they contain a target motif
- not enormous monolithic functions for the first pass
- optionally top-level only in v1 of the experiment

A reasonable v1 filter is:

- 32–512 model tokens
- top-level `FunctionDef` / `AsyncFunctionDef`
- body spans at least two statements
- keep separate buckets for functions with and without target motifs

---

## 5. Held-out evaluation sets

Training corpora and evaluation sets should be separated.

### 5.1 Broad held-out Python slice
A random held-out set from the same source family as Track A.

Used for:
- reconstruction loss
- broad-code sparse probing
- feature activation inspection

### 5.2 AST-curated motif holdout (**primary TempXC benchmark**)
This is the most important evaluation set.

Construct fixed-length windows centered on local motifs such as:

- `with ... as x:` + first body use of `x`
- `except E as e:` + first use of `e` in handler
- `for x in xs:` + first body use of `x`
- `import foo as bar` + first nearby `bar.` call

Each positive motif family should have matched negatives, for example:

- same keyword but no alias/bound variable
- alias bound but not used in the window
- body uses a different symbol than the bound one
- same bag of local tokens but phase-scrambled across positions

This gives us a benchmark where the object of interest is explicitly **local and phase-structured**.

### 5.3 MBPP sanitized
Use as a small curated Python evaluation slice.

Role:
- qualitative inspection
- reconstruction loss on clean task-like code
- autointerp examples
- secondary sparse probing

### 5.4 HumanEval
Use as another small curated slice.

Role:
- qualitative inspection
- clean held-out function examples with docstrings and tests
- secondary autointerp and motif analysis

MBPP and HumanEval should be treated as **curated evaluation sets**, not as the main training corpus.

---

## 6. What we should actually evaluate against MLC

The evaluation bundle should match the meeting’s emphasis on **reconstruction loss, sparse probing, confusion matrices, and feature geometry**, but adapted to code. The goal is to make the MLC comparison crisp.

## 6.1 Reconstruction loss (overall and slice-specific)

Report reconstruction loss for all three architectures on:

- broad held-out Python
- held-out function-def data
- the AST motif holdout

This is not the main scientific metric, but it is still load-bearing. If TempXC loses badly even on the motif slice, that is very bad news.

Also report **slice-specific reconstruction loss** for motif windows vs non-motif windows. That is more informative than one global average.

---

## 6.2 Broad-code sparse probing

Train sparse probes on encoded features for generic Python labels.

Examples of broad labels:

- inside function header vs body
- inside `class` definition
- inside `import` statement
- inside docstring
- async vs non-async function
- decorated vs undecorated function
- function contains `with`
- function contains exception handling

This benchmark asks whether one architecture produces generally more useful code features.

### Expectation
**MLC should probably win or tie here.**

Reason: these are relatively semantic or syntactic labels that the model may already compress into same-token representations across layers.

If TempXC wins here too, great. But this is not where I would put the probability mass.

---

## 6.3 AST motif sparse probing (**primary benchmark**)

This is the benchmark on which TempXC should have its best chance.

### Unit of evaluation
Use **window-level examples** rather than isolated tokens.

For each motif family, build balanced binary tasks such as:

- `WITH_BIND_USE`: window contains a `with ... as x` binding and an early body use of `x`
- `EXCEPT_BIND_USE`: window contains `except E as e` and an early use of `e`
- `FOR_BIND_USE`: window contains loop-variable binding and an early use in loop body
- `IMPORT_ALIAS_USE`: window contains alias binding and nearby alias usage

Also consider a **multi-class** task over motif identity.

### Probe protocol
Run the same sparse probing protocol across SAE, MLC, and TempXC.

For fair comparison:

- use the same held-out windows
- use the same probe budgets, e.g. `k in {1, 5, 20}`
- use the same train/val/test split
- use the same feature aggregation policy for any architecture that outputs per-position features

For SAE and MLC, aggregate per-token encodings across the same window. For TempXC, use the native window representation.

### Primary metric
Report:

- AUROC
- accuracy / F1 for balanced tasks
- probe-budget sweep (`k=1,5,20`)
- per-task breakdown

### Most important comparison
The most important result is the **low probe-budget regime**.

If TempXC has a real advantage, I expect it to show up most clearly when the probe only gets a few features. That is the regime where one latent “trajectory atom” matters.

### Main hypothesis
> **TempXC should beat SAE, and should have its best shot at beating MLC, on low-budget sparse probing over AST-curated local binding/use motifs.**

I would not pre-register a giant margin, but I would explicitly predict:

- **TempXC > SAE** on these motif tasks
- **TempXC ≳ MLC** at low probe budget on at least some motif families
- any TempXC advantage should be **largest on phase-structured motifs**, not on broad semantic labels

---

## 6.4 Phase randomization / shuffle controls (**TempXC-specific diagnostic**) 

The strongest control is not generic sentence shuffling, but **motif-local phase randomization**.

### Example control
Start from a positive motif window like:

```python
with open(path) as f:
    data = f.read()
```

Then build matched controls that preserve much of the token inventory but disrupt the phase structure, e.g.:

- swap body lines so the first use disappears from the local window
- replace `f.read()` with a use of a different variable
- keep the same bag of symbols but randomize local order within a constrained span
- keep the header but move the use outside the window

### Metrics
For each architecture, compare motif benchmark performance on:

- original windows
- phase-randomized matched controls

### Hypothesis
If TempXC is really using local temporal structure, then its feature usefulness should drop more sharply than MLC’s when the motif’s phase structure is destroyed.

This is a much more architecture-targeted test than generic shuffling.

---

## 6.5 Confusion matrices / overlap analysis

For each motif task, inspect where architectures differ:

- examples MLC gets right and TempXC gets wrong
- examples TempXC gets right and MLC gets wrong
- examples only SAE gets
- examples all three get

This was one of the most useful proposed next steps from the meeting, and it matters even more here because the whole point is to find a **regime difference**, not just a leaderboard.

If TempXC’s “wins” are mostly the same examples MLC already gets, then the scientific value is much smaller.

---

## 6.6 Autointerp

Run autointerp on top features from all three architectures, but make it explicitly secondary.

The point is not to win on vibes. The point is to ask whether the features that fire on motif windows are actually cleaner.

Useful outputs:

- top-activating examples for candidate motif features
- proportion of top activations that stay within one AST motif family
- qualitative comparison of a TempXC motif feature vs nearest MLC / SAE feature

### Hypothesis
On the motif slice, TempXC should have a chance to produce features with **higher activation purity** for local binding/use structure, even if MLC remains better on generic semantics.

---

## 6.7 Feature geometry / clustering

UMAP or other decoder-geometry summaries are useful as exploratory diagnostics, not as the main claim.

Use them to answer:

- do motif-enriched TempXC features cluster separately from broad semantic MLC features?
- do shuffle or phase-randomization controls change the TempXC geometry more strongly?

These plots are supportive, not decisive.

---

## 7. Exact benchmark hypothesis

Here is the most honest hypothesis split.

## 7.1 Where MLC should win

I expect **MLC** to win or tie on:

- broad reconstruction loss
- broad-code sparse probing
- docstring- or function-semantics-heavy tasks
- tasks where the stronger model likely already compresses the relevant information into same-token cross-layer structure

Examples:

- “is this function async?”
- “does this function have a docstring?”
- “is this a class method vs top-level function?”
- broad code-topic or code-style labels

## 7.2 Where TempXC should win

I expect **TempXC** to have its best chance on:

- **low-budget sparse probing over AST-defined local binding/use motifs**
- **phase-randomization controls** where local motif structure is destroyed
- any benchmark that rewards one sparse latent representing a short multi-position code event rather than a token-level semantic state

The cleanest pre-registered target is:

> **Primary TempXC win condition:** TempXC beats MLC on at least one AST-curated local motif family under low probe budget (`k <= 5`) and that advantage shrinks or vanishes on broad-code tasks.

That pattern would show a genuine niche rather than a generic superiority claim.

## 7.3 What would count as a failure for TempXC

Any of the following should update us strongly against TempXC:

- MLC beats TempXC on the AST motif benchmark too
- phase-randomization hurts MLC and TempXC equally little
- TempXC only learns smeared local token indicators rather than true binding/use features
- TempXC only looks distinct in UMAP/autointerp but shows no quantitative win anywhere

If that happens, the honest conclusion is that code was a fair matched test and TempXC still failed to show a niche.

---

## 8. Recommended experiment sequence

Do not run everything at once.

### Phase 1: Broad Python pilot
Train SAE / MLC / TempXC on a manageable broad Python subset.

Evaluate:
- reconstruction loss
- broad-code sparse probing
- motif holdout as a small targeted eval

Purpose:
- see whether code is even more favorable than web text
- get initial feature inspection

### Phase 2: Function-def training
Train the same three architectures on a function-level corpus.

Evaluate:
- reconstruction loss
- broad function-level sparse probing
- AST motif sparse probing
- confusion matrices
- autointerp

Purpose:
- remove the “files are too messy” confound

### Phase 3: Motif-enriched function training
If Phase 2 is promising, oversample or directly train on motif-heavy functions.

Evaluate:
- motif sparse probing
- phase-randomization controls
- reconstruction loss on motif vs non-motif slices

Purpose:
- test TempXC in the most matched regime

### Phase 4: only then run T-sweeps or architecture tweaks
Only do larger `T` or extra regularization once there is evidence that the primary motif benchmark is sensitive enough to detect a TempXC niche.

Otherwise the project risks becoming another architecture treadmill without a matched benchmark.

---

## 9. Minimal concrete version to run first

If we want the simplest strong first pass, run this:

### Train
- **SAE / MLC / TempXC** on **CodeSearchNet Python** or **`python-stack-v1-functions-filtered`**
- same Gemma-2B-it layer setup as current repo
- TempXC at **T=5** only

### Evaluate
1. **Reconstruction loss** on held-out functions
2. **Broad sparse probing** on generic function labels
3. **Primary motif sparse probing** on four AST tasks:
   - `WITH_BIND_USE`
   - `EXCEPT_BIND_USE`
   - `FOR_BIND_USE`
   - `IMPORT_ALIAS_USE`
4. **Confusion matrix** between MLC and TempXC on those four tasks
5. **Autointerp** on top features for each motif family

### Pre-registered expectation
- **MLC >= TempXC** on broad function-level tasks
- **TempXC > SAE** on motif tasks
- **TempXC has its best shot at matching or beating MLC** on the motif tasks, especially at low probe budget

That gives a clean answer quickly.

---

## 10. Dataset choices in one sentence each

- **The Stack Python**: best first broad training corpus
- **CodeSearchNet Python**: best first curated function corpus when docstrings/metadata help inspection
- **`python-stack-v1-functions-filtered`**: best large pure-function alternative when we want function-only training at scale
- **MBPP sanitized**: small curated Python eval for inspection
- **HumanEval**: small high-quality eval for inspection

---

## 11. Bottom line

The right claim is not “TempXC should beat MLC on code.”

The right claim is:

> **If TempXC has a real niche, it should appear on short AST-defined binding/use motifs in Python, under low sparse-probe budget, with phase-randomization controls showing selective degradation.**

That is the benchmark family on which TempXC should win.

If it does not win there, then code was a fair matched test and the case for TempXC becomes much weaker.

---

## References / implementation pointers

- The Stack (Python subset): `bigcode/the-stack`  
  <https://huggingface.co/datasets/bigcode/the-stack>
- Function-only Python corpus: `bigcode/python-stack-v1-functions-filtered`  
  <https://huggingface.co/datasets/bigcode/python-stack-v1-functions-filtered>
- CodeSearchNet / CodeXGLUE Python function data  
  <https://huggingface.co/datasets/google/code_x_glue_ct_code_to_text>
- MBPP sanitized  
  <https://huggingface.co/datasets/Muennighoff/mbpp>
- HumanEval  
  <https://huggingface.co/datasets/openai/openai_humaneval>

Suggested practical implementation:

1. build AST indexers for `With`, `ExceptHandler`, `For`, `Import`, `ImportFrom`, `FunctionDef`
2. emit fixed-size positive / matched-negative windows
3. cache Gemma-2B-it activations on those windows
4. train SAE / MLC / TempXC with matched budgets
5. run reconstruction loss, sparse probing, confusion matrices, and autointerp
