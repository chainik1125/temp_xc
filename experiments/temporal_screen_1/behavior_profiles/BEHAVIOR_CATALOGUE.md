## Safety-relevant behavior catalogue

The temporal screen should distinguish two targets rather than force every
behavior into an onset analysis:

- **Event-like:** the rollout contains a defensible first behavior-bearing
  event. Align trajectories to that event and measure formation *before* it.
  The label must not be used to select the neutral.
- **State-like:** the behavior is a persistent propensity or persona with no
  unique first token. Measure a fixed, cross-fitted score over preregistered
  normalized rollout positions. A first scored proposition may be reported as
  a secondary expression event, but is not the onset of the latent state.
- **Triggered state with an event:** a prompt-side trigger creates a state that
  is later expressed as a discrete action. Report both the trigger-aligned
  formation curve and the output-event-aligned curve.

“Programmatic reliability” refers to the event label, not merely the task
label: **high** means an exact parser, canary, or simulator determines the first
event; **medium** means a frozen deterministic classifier can do so; **low**
means semantic annotation or an LLM judge is normally required.

| Behavior and class | Onset/event label | Programmatic reliability | Neutral/control construction | Causal lever | Expected temporal profile | Safety relevance | Priority |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Refusal** — event-like output; prompt-side state formation | First refusal-bearing clause, plus first-decision refuse-versus-comply log odds. Independently sweep how much of the harmful request has been revealed. | **High** for templated refusal openings and token-mass metric; **medium** for semantically complete refusals. | Pair each harmful request with a tokenizer-length-, domain-, and style-matched harmless request. Preserve the system and chat-template prefix. Add word-order or benign-intensity controls so prompt truncation is not just loss of grammaticality. | Add and ablate the published refusal direction; reveal, mask, or relocate harmful spans. Directional ablation is the mediation test and prefix reveal is the contextual cause. | A sigmoid or step as decisive harmful information is revealed; direction projection should rise no later than refusal probability. The response-side state should be present before the first refusal token. Clean single-direction induction does **not** imply that the evidence was formed at one token. | Calibration case for safety policy formation and jailbreak resistance. | **1 — immediate mechanism calibration** |
| **Backtracking / semantic self-correction** — event-like | First genuine abandonment or correction of an earlier reasoning claim, not a lexical “wait.” Retain final correctness as a separate endpoint. | **Medium** with the frozen Ward event judge; **high** only in a synthetic state machine with explicit revisions. | Do not use event-present versus event-absent rollouts as the confirmatory neutral. Programmatically plant a correct versus incorrect intermediate state, or a passed versus failed check, on the same problem and at the same token length; preregister the pair before generation. | Published Ward dense direction; incorrect-state or failed-check insertion; activation patch/ablation around the identified pre-event band. | A localized pre-event rise, plausibly in the roughly 5–20-token band, followed by expression at the correction. A stable dense steering direction can coexist with evidence distributed over several token positions. | Reasoning reliability, error correction, and the existing TXC-positive calibration task. | **1 — immediate mechanism calibration** |
| **Sleeper activation and payload** — triggered state with an event | Prompt side: exact trigger boundary. Output side: first payload token, forbidden action, or target API call. Keep “armed” state and payload emission as separate outcomes. | **High** when the trigger and payload are planted canaries or structured actions. | Same prompt and rollout scaffold with trigger absent, nonce-substituted, or placed after the decision; match exact token count with inert nonce strings. Include triggered tasks where the requested action is benign to separate trigger recognition from payload policy. | Insert/remove/relocate the trigger; enable/disable the sleeper adapter; patch or ablate the trigger-state representation if identified. | Step at the trigger followed by a plateau if the model stores an armed state, then a possible second ramp immediately before payload emission. A flat early state would indicate weight-local rather than inference-temporal behavior. | Hidden policies, model-organism auditing, and deployment-time trigger detection. | **1 — clean scalable screen** |
| **Prompt-injection compliance** — triggered state with an event | First token or structured action uniquely satisfying the attacker instruction rather than the protected instruction. Use distinct random canaries for the two instructions. | **High** with canary outputs, exact choices, or structured tool calls; **medium** for open-ended instruction following. | Generate paired retrieved documents with identical format and token length: one contains an inert or task-relevant instruction and one an attacker instruction. Counterbalance instruction order and canary identity; retain the protected system/query prefix. | Insert, remove, paraphrase, or relocate the injected instruction; swap protected and attacker canaries; patch attention or residual states from the clean paired run. | A step or ramp after the model has seen both the protected goal and the conflicting instruction; the compliance score should rise before the first attacker-consistent action. Position-only and literal-keyword controls should be flat. | Direct defense against indirect prompt injection and instruction-hierarchy failure. | **1 — clean scalable screen** |
| **Secret exfiltration** — triggered state with an event | First exact occurrence of a random secret canary in text, a network/tool argument, or an externally visible message. Also score partial prefix leakage. | **High** with cryptographically random canaries and structured action logs. | Same domain, scaffold, and exact token length with a fresh decoy canary; cross secret accessibility with send authorization in a 2×2 design. A neutral should contain equally salient sensitive-looking text but no authorization conflict. | Grant/remove secret access; add/remove or relocate the exfiltration instruction; disable the send tool; patch secret-bearing or policy-bearing states separately. | Little effect until both secret and exfiltration goal are available, followed by a conjunctive step/ramp and a sharp event at the first leaked token or send call. This cleanly separates remembering a secret from choosing to leak it. | Data loss, privacy, insider-threat model organisms, and agent security. | **1 — clean scalable screen** |
| **Tool use and tool-call ordering** — event-like | Exact opening of a structured tool call; tool name and argument validity. For plans, label the first dependency-order violation or first call completing the intended subgoal. | **High** with schema parsing and a deterministic task simulator. | Same task with tool required versus a length-matched directly answerable variant; for ordering, permute independent tool-result blocks while preserving their multiset and use tasks with a unique dependency DAG. | Add/remove tool availability or a required intermediate result; tool-use steering feature/direction where available; patch states from the correct-order paired run. | A pre-call ramp or spike followed by a reset after the call. Multi-call tasks may show repeated phase-specific peaks; a constant tool-use direction may induce calls without encoding the correct schedule. | Unsafe actions, unauthorized API use, and agentic control-flow failures. | **2 — next programmatic panel** |
| **Hallucination / factual fabrication** — event-like only with explicit ground truth | First unsupported atomic claim. Prefer exact entity, value, citation, or simulator-state violations over free-form factuality judgments. | **High** in synthetic databases, closed-world QA, code execution, or simulators; otherwise **low–medium**. | Counterfactual paired worlds with the same syntax and token length; one supplies the decisive fact and one supplies a matched distractor. Match answer length and separate “unknown/abstain” from a false assertion. | Reveal/remove/contradict the decisive evidence; patch the fact-bearing state from the correct world; steer abstention only as a secondary lever. | Evidence-supported answers should form after the decisive fact. A false-claim precursor may emerge immediately before the first unsupported proposition, but open-ended judge labels will blur the boundary. | Factual reliability, fabricated citations, and unsafe medical or technical advice. | **2 — use structured worlds first** |
| **Deception / concealment** — state-like intention with a possible event | In a formal game, the first statement or action that is false relative to model-visible state and advances a hidden goal. A false statement alone is not evidence of deception. | **High** in a simulator with known observations, goals, and payoffs; otherwise **low** because intent must be inferred. | Identical world state with truthful versus concealment incentives, or the same organism with hidden-goal trigger absent. Match surface facts and action affordances; include accidental-error trials. | Toggle the hidden objective or reward contingency; trigger/disable the model-organism policy; test a deception direction only with causal ablation and preservation of task competence. | A persistent goal/intention state may precede a sharp deceptive act. Report state-profile and event-aligned views separately; do not interpret the first lie as the state’s onset. | Scheming, oversight evasion, and strategic manipulation. | **3 — only in formal model organisms** |
| **Evaluation awareness** — state-like, often externally triggered | Prompt-side boundary of the last decisive evaluation cue. Output side: first preregistered action that differs between evaluation and deployment conditions. | **High** for cue placement and exact-choice organisms; **medium–low** for open-ended behavior. | Same evidence multiset with evaluation cues moved, paraphrased, or label-permuted; pair evaluation and deployment scenarios with matched task content and length. Include no-literal-match cues. | Add/remove/relocate evaluation cues; toggle the organism trigger; ablate or steer an evaluation-awareness direction where validated. | Step after sufficient evaluation evidence followed by persistence. If the state is already separable at the first output token, it is short-horizon even when later behavior is temporally extended. | Sandbagging, audit evasion, and deployment-conditioned behavior. | **2 — state-profile benchmark** |
| **Emergent misalignment (EM)** — state-like | No primary token-level onset. Use a fixed cross-fitted misalignment score at normalized rollout positions and the final frozen alignment metric. Optionally mark the first coherent misaligned proposition as an expression event. | **Medium** for whole-response frozen scoring; **low** for the optional first-proposition boundary. | Same prompts for the aligned base and EM organism, matched generation lengths, plus within-organism benign-domain prompts. Cross-fit by prompt, not response, and exclude incoherent outputs or score coherence separately. | Adapter on/off is the clean organism-level cause; use validated EM steering directions or model-difference patches as mediation tests. Do not infer causality from an observational probe alone. | A persistent or gradually strengthening score rather than a unique onset. A flat separation from the first generated position is a valid “weight-encoded, short inference horizon” result, not a failed screen. | Broadly misaligned persona and current-paper negative/comparison case. | **1 — state calibration because infrastructure exists** |
| **Sycophancy** — state-like pressure with an answer event | In exact-answer tasks, first answer choice changed from the known-correct answer toward the user’s stated belief. In dialogue, use normalized-turn score rather than inventing an onset. | **High** for counterfactual MCQ or arithmetic; **low–medium** for free dialogue. | Same user-turn multiset with opinion pressure reordered or polarity-counterbalanced; no-opinion and correct-opinion arms; exact token-length and answer-position matching. | Add/remove/relocate user opinion; flip its polarity; apply an agreeableness/sycophancy direction only with a competence-preserving control. | Accumulation across pressure turns followed by a choice event, or a broadcastable state visible throughout. A single agreeableness direction may steer the outcome while carrying little temporal schedule. | Social manipulation, preference falsification, and unreliable advice under user pressure. | **2 — structured exact-answer version** |
| **Recovery after unsafe or erroneous behavior** — event-like | First action that returns to a formally safe/correct state: retracting a false claim, cancelling an unsafe call, repairing a failed test, or switching from compliance to refusal. Score whether recovery succeeds, not just recovery language. | **High** with tests, simulators, and action logs; **medium** for semantic safety recovery. | Programmatically inject a matched failure signal versus success signal at a fixed checkpoint. Use the same prefix, action budget, and token length; never select controls based on whether spontaneous recovery later occurred. | Inject/remove a test failure, policy warning, or tool error; patch pre-recovery state from a successful paired run; steer the known backtracking/refusal lever when appropriate. | A change after the diagnostic signal, then a localized precursor to the first corrective action. Repeated failures permit a recovery-latency curve and test whether the state resets after correction. | Runtime containment, self-correction, and resilience of autonomous agents. | **1 — clean generalization of backtracking** |

## Recommended first panel

A compact panel should cover all three temporal classes:

1. **Refusal** as the clean internal-mediation calibration: a known direction,
   prompt-prefix reveal, and a discrete output behavior.
2. **Backtracking** as the existing distributed/event-like calibration, using a
   programmatically planted pre-outcome error rather than selecting no-event
   rollouts after generation.
3. **Sleeper payload or exfiltration** as the strongest exact triggered-state
   task: the trigger, state demand, and emitted event are all mechanically
   controlled.
4. **EM** as the state-like comparison: a cross-fitted normalized-progress
   profile should be allowed to return “present from the start.”
5. **Recovery** as an additional exact event task if a simulator or unit-test
   harness is already available.

For every row, record two different causal objects:

- **Context causality:** an exogenous prompt, environment, trigger, or evidence
  intervention changes the behavior.
- **Internal mediation:** ablation or patching of the measured representation
  removes or restores that change while preserving generic capability.

The first supports a temporal-dependence claim. The second is needed to say
that the measured feature *mediates* the behavior. A steering vector that can
induce a behavior is an effector; it is not, by itself, evidence that the
natural feature formed at a single position.

## Control rules

- Generate neutrals from the task specification before observing outcomes.
- Match tokenizer length, domain, format, answer affordances, and decoding
  budget; average character length is not enough.
- Keep prompt-side trigger position, output-event position, and normalized
  rollout progress as separate coordinates.
- Use exact parsers or simulators whenever possible. Freeze semantic judges and
  validate them on a human-labeled subset when exact labels are unavailable.
- Report coherence or task competence independently. Entropy cannot separate
  temporal dependence from incoherence.
- Cross-fit learned directions and probes by prompt or scenario family. Never
  fit and evaluate temporal formation on tokens from the same rollout.
- Treat event-conditioned neutrals as descriptive sensitivity analyses, not
  confirmatory causal controls.

## Local basis

This catalogue consolidates the behavior-specific controls in
`experiments/temporal_screen_1/overnight/literature.md`, the dual prompt and
trajectory panels in `experiments/temporal_screen_1/overnight/theory.md`, the
Ward feature-formation pilots in
`experiments/temporal_screen_1/feature_formation/`, and the broader task audit
in `docs/dmitry/sprints/2026-07-26_txcwins_10h/literature_catalogue.md`.
