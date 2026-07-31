## Persona drift as a temporal prediction problem

This experiment tests whether a causal activation history contains information
about *future* movement along the released Assistant Axis beyond:

- the current Assistant Axis projection;
- the latest user-message embedding; and
- sparse features of the current assistant turn.

It deliberately stops after the predictive Step 7 gate. No steering result is
claimed until the temporal prediction gate has been evaluated.

## Outcome of the initial 10k-step run

The predeclared Step-7 gate **did not pass**. In the primary \(W=8\),
four-turn-horizon cell:

- raw history minus raw current activation had
  \(\Delta R^2=+0.0018\), with 95% CI \([-0.0131,+0.0167]\);
- SAE plus TXC minus SAE had
  \(\Delta R^2=-0.0773\), with 95% CI \([-0.1320,-0.0360]\).

All four representation models reached their best fixed-validation loss before
4,000 steps and regressed by 10,000 steps. The 10k checkpoints were retained
for the frozen comparison requested by the protocol. Shorter exploratory
horizons contain a small raw-history signal, but the final TXC latents do not
capture it beyond an SAE.

See [RESULTS.md](RESULTS.md) for the full interpretation,
[training diagnostics](results/training/training_diagnostics.png) for the
loss/dead-latent plot, and the
[Step-7 comparison](results/future_drift_probe.png) for the predictive result.

## Primary hypothesis

For Assistant Axis projection \(a_t\), the temporal hypothesis is:

\[
p(a_{t+k}\mid h_{t-W+1:t}, u_t)
\quad\text{outperforms}\quad
p(a_{t+k}\mid h_t, u_t),
\]

where \(u_t\) is the latest user message. The primary continuous target is the
minimum Assistant Axis projection over the following \(k\) assistant turns.

## Reference machinery

The implementation pins
[`safety-research/assistant-axis`](https://github.com/safety-research/assistant-axis)
at commit `a98961956072224eaf244eb289d6c01700b63795` and delegates:

- the one-conversation agreement smoke to
  `assistant_axis.generate_response`;
- full-corpus batched generation to the reference
  `assistant_axis.VLLMGenerator`;
- chat templating and response-span identification to
  `ConversationEncoder`;
- post-MLP residual extraction to `ActivationExtractor`; and
- response-token pooling to `SpanMapper`.

The released Qwen3-32B Assistant Axis is loaded from
`lu-christina/assistant-axis-vectors`. The monitor is the paper's middle layer
32. A vectorized projection is checked against the reference `project`
function for every extracted turn.

The Qwen subject model, Qwen embedding model, and Assistant-Axis vector dataset
are pinned to resolved Hugging Face revisions in `config.json`. The pipeline
also fails closed unless the reference interpreter has Transformers 4.57.5,
PyTorch 2.9.0, and (for generation) vLLM 0.13.0.

Before collecting the new corpus, the pipeline projects the released Qwen
case-study transcripts and records capped versus unsteered trajectories.

## Semi-synthetic conversations

The complete multi-turn corpus from the Assistant Axis paper is not released.
Only example transcripts are public. This experiment therefore uses a new,
fixed user-side corpus that preserves the paper's four domains:

- coding;
- writing;
- therapy-like emotional support; and
- philosophical discussion about AI.

There are five user personas and twenty topics per domain, giving 100
conversations per domain and 400 total. Every script contains fifteen user
turns. The user sequence is fixed and does not depend on exact assistant
wording, so later controller comparisons can face identical disturbances.

Splits are made by persona within every domain:

- persona indices 0–2: train;
- persona index 3: validation;
- persona index 4: test.

Consequently, no turn from a held-out conversational style enters dictionary
training or probe fitting. The initial test set contains one held-out style in
each of four domains; conversation-bootstrap intervals therefore quantify
generalization across its topics, not across a large population of personas.

## Representations

All representations are trained on layer-32 assistant-turn means normalized by
the training-set mean and one global RMS scalar:

- BatchTopK SAE on the current turn;
- paper T-SAE on consecutive turn pairs;
- post-squash BatchTopK TXC with \(W=4\); and
- post-squash BatchTopK TXC with \(W=8\).

The initial run uses one representation seed (`42`), 8,192 latents, nominal
\(L_0=20\), and **10,000 optimizer steps per model**. T-SAE uses the paper's
contrastive coefficient 0.1. The dead-feature threshold is one million tracked
positions rather than the architecture's ten-million default, because this run
contains only 5.12 million positions; otherwise AuxK could never activate.
Atomic resume checkpoints are refreshed every 2,000 steps and removed after a
model completes.

Training health artifacts include:

- the complete sampled training-objective curve;
- a comparable fixed-validation reconstruction-NMSE curve used for the plateau
  diagnostic (rather than T-SAE's two-level cumulative objective);
- comparable reconstruction NMSE reported separately on train, validation, and
  persona-held-out test conversations;
- realized inference \(L_0\);
- firing-rate distribution and features that never fire, both reported
  separately by split as well as over the complete activation corpus;
- features starved according to multiple tokens-since-fired thresholds; and
- a tail loss slope and explicit plateau diagnostic.

Only final weights are retained; optimizer-state checkpoints are not saved.

## Step-7 predictors

For horizons 1, 2, and 4, validation-selected ridge probes compare:

- current axis only;
- current axis plus turn position and domain;
- current axis, turn/domain, and latest user embedding;
- raw current activation;
- raw activation history;
- SAE latents;
- T-SAE latents;
- TXC latents;
- SAE plus TXC latents; and
- T-SAE plus TXC latents.

All nontrivial models include current axis, normalized turn position, domain
indicators, and latest-user covariates. This prevents a temporal representation
from winning merely by recovering the scheduled script position. The critical
nested comparison is `(SAE + TXC) - SAE`; the latent-free information gate is
`raw history - raw local`. Confidence intervals use a conversation-level
bootstrap stratified by domain. The predeclared primary cell is \(W=8\),
horizon 4; other cells are exploratory.

At the primary cell, additional controls include the complete causal history of
user-message embeddings and a deliberately noncausal oracle containing the
future user messages. These distinguish model-state history from predictable
structure in the fixed user scripts. AUPRC is reported only for the compatible
future-minimum target; the future-final-delta target is evaluated with
continuous metrics only.

## Commands

Generate and freeze the user scripts locally:

```bash
uv run python -m \
  experiments.power_spectrum.persona_drift_txc.generate_user_scripts \
  --output experiments/power_spectrum/persona_drift_txc/data/user_scripts.jsonl \
  --usage-output \
  experiments/power_spectrum/persona_drift_txc/data/script_generation_usage.json
```

On a GPU machine, run a one-conversation end-to-end collection smoke:

```bash
bash experiments/power_spectrum/persona_drift_txc/run_pipeline.sh collect-smoke
```

Run the full pipeline through Step 7:

```bash
export PERSONA_DRIFT_REFERENCE_PYTHON=/path/to/assistant-axis/.venv/bin/python
export PERSONA_DRIFT_VLLM_PYTHON="$PERSONA_DRIFT_REFERENCE_PYTHON"
bash experiments/power_spectrum/persona_drift_txc/run_pipeline.sh all
```

The pinned Assistant-Axis environment is kept separate from the experiment's
`uv` environment on RunPod. It is created from the reference repository's
committed `uv.lock` and supplies both activation extraction and vLLM
generation. This prevents the repository's newer Transformers version from
silently changing chat-template or span behavior. Representation training
continues to use this repository's normal `uv` environment.

Large activations and checkpoints stay under this capsule's ignored
`artifacts/` directory. Small tables, plots, and the Step-7 gate are written to
`results/`.

Run local protocol and training smoke tests:

```bash
uv run pytest -q \
  experiments/power_spectrum/persona_drift_txc/test_persona_drift_txc.py
uv run ruff check experiments/power_spectrum/persona_drift_txc
```

## Decision rule

Proceed to steering only if at least one of the following has a
conversation-bootstrap \(\Delta R^2\) 95% lower bound above zero in the
predeclared \(W=8\), horizon-4 cell:

- raw history improves future-drift prediction over raw current activation; or
- TXC latents improve future-drift prediction beyond current SAE latents and
  the latest user message.

If raw history adds no reliable information, the experiment is not a temporal
benchmark. If raw history helps but TXC latents do not, the negative result is
about the learned representation rather than the underlying temporal signal.
