# Backtracking-like task candidates for the NeurIPS reviewer stage

_Literature screen begun 2026-07-22; continuously updated through 2026-07-24. These are ex ante task proposals, not TXC results._

## Bottom line

The leading independent task is now **human writing-revision destination**.
Immediately before a writer begins deleting text at the leading edge, predict
how many subject-model tokens the writer will erase. The exact-token cohort has
6,224 events from 2,510 writers. On layer-10 subject-model activations, ordered
history improves from 1.423 log loss at \(T=1\) to 1.239 at \(T=6\), versus
1.423 for the endpoint, 1.492 for an order-invariant summary, and 1.632 for a
probe retrained on shuffled histories. All equal-writer intervals exclude zero.
This is a passed raw-representation gate; frozen TXC-versus-SAE evaluation is
still pending.

The cleanest literal temporal-offset backup remains **speech-repair
destination distance**: immediately before the first repair word, predict
which of the preceding five positions began the reparandum. This fixes the
failed onset formulation, whose label is already readable from the current
repair word. On 9,030 simple Switchboard repairs, ordered five-word history
beats the endpoint and order-invariant controls, loses its advantage under
reversal, and improves monotonically from \(T=1\) through \(T=5\). It is the
next activation experiment if the deletion dictionary gate fails.

The strongest immediate exact-model reasoning experiment remains **strict pre-onset prediction of self-checking**. The strongest independent comprehension task is **human reading-time spillover** in Natural Stories. Its local opportunity gate passes before any activation or dictionary result is examined: ordered current-plus-history lexical and surprisal covariates beat an endpoint, an explicit first difference, and parameter-matched order-invariant histories in held-out stories, and reversing the historical positions removes the gain. The lag-identity result independently replicates on A-Maze responses to the same ten stories.

The strongest mechanism-focused bridge is now **garden-path reanalysis**. Hanna and Mueller already use per-token SAEs to identify competing parses, while their public stimuli expose the ambiguous cue, a variable-duration intervening region, and the disambiguating word. This gives a direct test of the proposed scale story: a short TXC should help only when useful information lies in the ordered local transition from maintained parse commitment to conflicting evidence, not merely because the distant cue and current state are correlated. It also reuses our existing Gemma-2-2B-IT layer-13 dictionaries and directly distinguishes this paper from the closest same-title SAE work. The dataset is small, so the first result must be a paired raw gate rather than another trained-benchmark headline.

The exact-model **early reasoning-instability prediction of final mathematical correctness** test is now closed. It used mechanical answer labels and fixed 32/64/128-token prefixes, but ordered \(T=6\) windows did not beat endpoint or invariant summaries and did not degrade under reversal or shuffling. This supports the prediction that an early rollout prefix is a weak or global label, not a localized transition task.

The most elegant spectral candidate is **reasoning-loop onset**, but public-corpus audits make it a generation project rather than a rebuttal shortcut. The strict 30-gram-at-least-20-times detector fires in only 53 of 29,856 public 2K-token exact-model traces and 39 of 22,037 public 32K-cap exact-model traces. Word Salad Chopper supplies broader semantic repetition labels, but only 89 of 1,000 released traces cross its sustained word-salad boundary. Pipis et al. obtain abundant loops by generating low-temperature AIME traces up to 30K tokens; reproducing that regime remains scientifically attractive but compute-heavy.

**First reasoning-error localization** is now secondary. ProcessBench has attractive human boundaries, but none of its traces come from the exact subject model, the median erroneous step is 325 characters, and the published dynamics result uses whole-step or whole-trace information. It is a useful raw longer-window gate, not the best immediate \(T\leq6\) test.

**Counterfactually localized deceptive commitment** from Merrill and Srivastava's _The Point of No Return_ remains the cleanest solution to the emergent-misalignment label problem, but a 100-file audit makes it a no-go as the immediate backtracking analogue. Commitment is labeled after a causative sentence whose median length is 33 tokens, current \(T=6\) windows cover none of those sentences fully, and the paper's Llama-8B temporal baselines already lose to its raw endpoint. It is retained only as a conditional raw gate and future long-window task.

| Rank | Candidate | Observable boundary | Strongest reason to try it | Main threat | Operational status |
|---|---|---|---|---|---|
| Lead | Human writing-revision destination | Strictly before a leading-edge deletion burst, identify how many subject-model tokens will be erased | 6,224 exact-token events; ordered layer-10 activation history wins at \(T=6\) against endpoint, invariant, difference, and retrained-shuffle controls | A frozen TXC may still fail to recover the raw ordered signal | Exact-token and raw-activation gates complete; frozen TXC-versus-SAE next |
| 1 | Speech-repair destination distance | Strictly before the first repair word, identify which prior model-token position began the reparandum | 7,071 exact-token events after global window deduplication; ordered history beats endpoint/invariant controls with a monotone \(T=1\ldots5\) curve | Raw activation and frozen-dictionary gates remain | Best literal temporal-offset backup |
| 2 | Self-checking or plan-generation onset | First sentence entering a tagged reasoning mode | 435K public sentence labels on the exact subject model; cheap strict-pre-onset construction | Close to the existing MATH/backtracking domain; labels are GPT-derived | Cheapest immediate reasoning pilot |
| 3 | Human reading-time spillover | Word-level latency after a short stimulus history | 10,198 self-paced targets plus 9,777 A-Maze replication targets; ordered lexical gate beats invariant and reversal controls in both paradigms; exact Gemma dictionary match | Gates use word lags rather than model-token activations; participant-level nuisance control is required | Best independent comprehension task; activation gate next |
| 4 | Syntactic belief update / garden-path reanalysis | Word that sharply changes the incremental parse distribution | Formal continuous transition label; released incremental parser; paired human garden-path validation; exact Gemma dictionary match | Stock labels leak remaining sentence length and require a fixed-horizon correction; \(T>2\) must beat an explicit first difference | Strong independent theory bridge |
| Closed | Incremental speech-repair onset | First word that repairs an earlier reparandum | 2,014 matched public onsets; literal six-token transition; independent of math CoT | Ordered raw AP .943 equals endpoint AP .943 | Replaced by destination-distance task |
| 5 | Reasoning-loop onset | First unit entering a sustained cyclic trajectory | Exact model, mechanical boundary, and an intrinsically periodic phenomenon | Released normal-temperature corpora have too few strict loops | Low-temperature generation required |
| 6 | First reasoning-error localization | End of first human-annotated incorrect step | Human labels and direct precedent for velocity/acceleration-style hidden-state features | Existing evidence is at whole-step scale; released traces are from other models | Secondary raw and long-window gate |
| 7 | Forced-answer belief update | Sharp change in answer distribution between successive prefixes | Behavioral, model-intrinsic event label with no LLM judge | Additional inference; current state or explicit endpoint difference may suffice | Principled localized pilot |
| 8 | Deceptive commitment juncture | Sentence where counterfactual deception probability jumps | Exact subject model, local causal label, five-domain transfer, matched honest/deceptive scenarios | Strong endpoint; current \(T=6\) misses the sentence-scale transition | Conditional long-window/raw task |
| 9 | Confidence-region entry | Change point from exploratory to converged answer entropy | Explicit sequential change-point theory and model-intrinsic labels | The entropy scalar and final state may already be sufficient | Principled raw-gate candidate |
| 10 | RAG hallucination onset | First annotated unsupported token | Public word-level labels and natural local on/off spans | Exact \(T=6\) text gate does not robustly beat the invariant bag; subject checkpoint is incompletely versioned | Data-to-text-only raw backup |
| Closed | Early reasoning instability | Bag of local windows from the first 32, 64, or 128 CoT tokens | Exact model and mechanical final-correctness labels | Ordered raw windows lose to invariant summaries | Raw gate failed; no dictionaries |
| 11 | Error recognition before self-correction | Post-answer newline before verification | Strong probe and activation-patching evidence | Published task is designed around a sufficient single token | Citation or later adaptation |
| Control | Emergent response planning | Whole-response property | Tests the predicted null for global persistent labels | No localized transition | Negative-control task |

## What counts as backtracking-like

The motivating backtracking result has a three-stage structure: the model commits to a partial course, an internal state associated with recognizing a problem develops, and an overt correction begins later. A useful second task should therefore have most of the following properties:

- an onset that can be aligned at a token or sentence boundary;
- published or preliminary evidence that the relevant state exists before the overt marker;
- labels that already exist or can be generated cheaply;
- a reason that several nearby positions, and ideally their order, should matter;
- a detection metric and, if possible, a causal intervention;
- enough natural variation to resist lexical and length shortcuts.

This criterion is stricter than “a task involving sequences.” The scientific claim is that TXCs should help when evidence for a latent transition is distributed across nearby positions, not whenever the input happens to be text.

## 0A. Counterfactually localized deceptive commitment

### Why it is the cleanest answer to the EM-label problem

[Merrill and Srivastava (2026)](https://arxiv.org/abs/2605.17113) construct five strategic environments—bluffing, maze guidance, investment advice, car sales, and offer negotiation—in which deception is never requested and the final behavioral label follows mechanically from environment state. For a reasoning trace split into sentences \(s_1,\ldots,s_K\), they repeatedly sample continuations after a prefix and estimate

\[
p_k = \Pr(\text{deceptive completion}\mid s_{\leq k}).
\]

A deceptive commitment juncture is a boundary where \(\Delta p_k=p_k-p_{k-1}\) exceeds a calibrated threshold. This is the temporal ground truth we wanted for emergent misalignment: a rollout is not treated as uniformly misaligned, and locally neutral windows are not mislabeled positive merely because deception appears somewhere later.

The [released corpus](https://huggingface.co/datasets/anonymous-neurips-2026-ED/deception-localization) contains 12,500 deceptive and 12,500 honest Llama-8B trajectories, 2,500 of each class in each environment. A trace has 69.6 reasoning sentences on average and 14.94 adaptively localized boundaries; each localized boundary is scored with 50 counterfactual continuations. A deceptive juncture occurs in 40.6% of deceptive Llama traces and an honest juncture in 34.1% of honest traces. The paper's text-only features transfer at chance, while internal features transfer above chance, so the task is not obviously solved by stable lexical cues.

### The two questions must remain separate

1. **Contemporaneous detection:** use a \(T\)-token window ending at the last token of sentence \(s_k\) to predict whether \(\Delta p_k\) is a commitment jump. This asks whether the ordered process by which the commitment sentence forms carries information beyond its endpoint.
2. **Strict anticipation:** use a window ending before the first token of \(s_k\) to predict whether the next sentence will cause a commitment jump. This is closer to the forbidden-word and backtracking precursor story, but the source paper explicitly finds the pre-commitment prefix ambiguous, so failure here would not discredit contemporaneous event detection.

Calling the first task “pre-onset” would be wrong: it sees the commitment sentence. Calling the second the headline without first establishing signal would risk choosing a target that is counterfactually defined to be difficult before onset.

### 100-file audit and decision

A deterministic audit of 100 Llama-8B files, 20 per environment, found 1,395 evaluated history transitions, of which 951 were truly adjacent sentence boundaries. Forty adjacent transitions had \(|\Delta p|>0.3\), and 35 remained semantic commitment events after removing `</think>` artifacts. Their event sentences had median length 33 model tokens, interquartile range 21–40, and maximum 50. Full event-sentence coverage was:

| Window | Events fully covered |
|---:|---:|
| \(T=6\) | 0 / 35 |
| \(T=12\) | 3 / 35 |
| \(T=24\) | 11 / 35 |
| \(T=32\) | 17 / 35 |
| \(T=50\) | 35 / 35 |

The paper's exact Llama-8B results also give a strong prior against a short-window win: raw final-token activation reaches 0.705 OOD AUROC, while final-minus-previous activation reaches 0.648, final-minus-mean-of-four reaches 0.675, all attention reaches 0.641, and attention plus PCA reaches 0.701. Human prediction rises from 20.7% before the commitment sentence to 94.9% after it, which suggests strict anticipation may be genuinely underdetermined rather than merely hard.

This demotes deceptive commitment from the immediate headline experiment. Current \(T=6\) sees only the tail of a sentence-scale event, while the existing endpoint is already strong. A positive \(T=32\) or \(T=50\) raw result would motivate future long-window dictionaries but would not validate the present rebuttal configuration.

### Cheap raw-activation gate

Sample 1,000 Llama-8B files, 200 per environment, stratified across trajectory class and scenarios. The frozen deterministic manifest totals 1.740 GiB compressed and has sorted-path SHA-256 `26b428b19623637fdcd9b56d2b40d3807ba4b35ee73849c0e061fb697aa1c41b`. Discard the 50 generated continuations stored at every boundary and retain only the original trace, sentence spans, boundary indices, \(p_k\), confidence intervals, environment, and scenario identifier. The full exact-model slice is roughly 47 GB compressed because it embeds all continuations. In a direct 2026-07-24 check, one 646 KB source file became 82 KB as uncompressed thin JSON and 4.8 KB when recompressed, a roughly \(133\times\) reduction.

Teacher-force the released trace through the exact model and cache residual activations around each evaluated boundary. Before touching TXC dictionaries, compare:

- the final token \(h_{t_k}\);
- the best individual offset within the \(T\)-token window;
- mean, max, and learned order-invariant pooling;
- ordered concatenation or a small position-specific linear model;
- \(h_{t_k}-h_{t_{k-1}}\), matching the paper's sentence-boundary difference baseline;
- shuffled and reversed token order;
- the paper's lexical last-sentence baseline.

Use scenario-grouped folds and make leave-one-environment-out transfer the eventual headline. The raw opportunity gate passes only if the ordered window beats the endpoint and invariant controls, and if reversing or shuffling the window removes a meaningful fraction of the gain. If only the sentence-boundary difference works, the evidence supports longer-range change-point features but not a \(T\leq6\) adjacent-token TXC.

### Go/no-go interpretation

A TXC advantage here would connect the theory to a real latent state path: commitment is a localized change in a conditional future-behavior probability, and the useful feature is an ordered pattern at the transition rather than a global deception label. A null against the endpoint would also be informative and consistent with the paper's Llama results, but it would make this a poor rebuttal experiment.

Run four cached views: the existing Ward-style \(-13{:}-8\) pre-onset window, an immediate \(-6{:}-1\) pre-onset window, the last six tokens of the commitment sentence, and raw-only intra-sentence windows \(T\in\{12,24,32,50\}\). Only a pre-onset ordered PR-AUC advantage above 0.02 over the frozen strongest static comparator, with a positive prompt-clustered 95% CI, authorizes describing this as backtracking-like or evaluating current dictionaries. An intra-sentence result supports the weaker claim of commitment detection; a gain only at \(T\geq32\) is a future long-window result.

**Recommendation:** no-go as the immediate task; retain one bounded raw gate after the higher-priority candidates.

## 0B. Exact-model reasoning-mode onset

### Why it is the fastest high-powered test

The public [DeepSeek-R1-Distill-Llama-8B MATH labeled-sentence corpus](https://huggingface.co/datasets/jrosseruk/DeepSeek-R1-Distill-Llama-8B-MATH-labeled-sentences) contains 435,525 sentences from 4,413 traces, with function tags including `plan_generation`, `self_checking`, `active_computation`, and `uncertainty_management`. Because the traces were generated by the same model used for our existing dictionaries, this avoids the teacher-forced foreign-model caveat of RAGTruth. It is also small enough to prepare locally before GPU capacity opens.

For tag \(z\), define an onset at sentence \(k\) by \(y_k=1[z\in s_k \land z\notin s_{k-1}]\). The strict test uses only the last \(T\) token activations before the first token of \(s_k\). `self_checking` is the best primary target because a verification state could build across the preceding computation; `plan_generation` is a secondary target that tests transition into a new plan without reusing the backtracking label.

Negatives are sampled from within the same trace and matched one-to-one on the exact previous-sentence tag set and relative trace position within a 0.05 caliper. This automatically matches correctness, MATH category, and difficulty without conditioning on the future target sentence's length. All traces for a problem stay in one fold. Evaluate \(T\in\{1,2,3,4,5,6\}\) and the same endpoint, best-offset, invariant, shuffle, reverse, and boundary-jitter controls used for backtracking.

The task passes only if an ordered representation beats the best single offset and invariant pooling before the new sentence is visible. A gain only after the tagged sentence begins is ordinary mode recognition; a gain explained by the preceding function tag or trace position is label leakage. Because these labels were assigned by GPT-4o-mini, a positive result should be manually audited on a blinded sample before it enters the paper.

A deterministic cohort build on 2026-07-24 sequentially aligned all 435,525 labeled sentences to all 4,413 raw responses after whitespace normalization, with exact raw-span hash round trips. It found 7,486 raw self-checking onsets and 14,150 plan-generation onsets. The frozen within-trace matcher retains 5,621 and 9,858 positive-control pairs, respectively, for 30,958 rows across 282 problem groups. Every row records the exact raw boundary and asserts that the activation prefix ends before the target sentence.

A preliminary five-fold problem-grouped TF-IDF probe over the strict raw prefix achieved:

| Target | Last 128 characters AUROC | Last 512 characters AUROC |
|---|---:|---:|
| `self_checking` onset | .563 | .533 |
| `plan_generation` onset | .607 | .570 |

An exact-token \(T=6\) view gives a more adverse order control:

| Target | Endpoint token AUROC | Invariant six-token bag | Ordered token-position view |
|---|---:|---:|---:|
| `self_checking` onset | .549 | **.575** | .557 |
| `plan_generation` onset | .599 | **.623** | .611 |

The task therefore has modest local surface predictability, but its visible-token signal is set-like rather than order-specific. This lowers the prior for a TXC advantage without making the hidden-state gate redundant: the remaining question is whether an ordered internal trajectory exists beyond the token identities, last hidden state, invariant activation summaries, and explicit first differences.

The extraction boundary is now audited for the complete cohort. The exact fast-tokenizer backend safely maps all 30,958 rows from 1,785 traces, with zero failures. In 14,510 cases, the first target-sentence character lies inside a leading-space model token that begins before the raw character boundary, so the extractor excludes that entire overlapping token. The longest prompt plus response is 15,594 tokens.

**Recommendation:** keep this as the first exact-model CoT activation screen, but run the Natural Stories gate first if only one independent task fits the rebuttal budget. Keep `self_checking` primary; `plan_generation` is the prespecified replication rather than a target selected after seeing activation results.

## 0C. Human reading-time spillover

### Why this task has a principled temporal target

[Natural Stories](https://pmc.ncbi.nlm.nih.gov/articles/PMC8549930/) contains 10,245 words in ten English narratives, with word-by-word self-paced reading times from 181 participants. The stories were edited to retain naturalistic discourse while over-representing rare syntactic constructions. This is useful for TXCs because human processing cost is known to be temporally diffuse: difficulty induced by one word can slow responses to later words. [Shain and Schuler (2018)](https://aclanthology.org/D18-1288/) formalize this as an impulse-response problem and show that models which recover the temporal response kernel improve prediction on Natural Stories and other reading corpora.

[Tsipidi et al. (2026)](https://aclanthology.org/2026.acl-long.575/) provide the closest direct precedent: regularized probes on current-word hidden representations predict Provo and MECO eye-tracking measures, with early-layer states often beating surprisal for first-fixation and gaze duration. Their unit representation mean-pools the current word's subtokens and does not test a lagged ordered trajectory or sparse dictionaries. The result raises the prior that raw states contain psychometric signal while also defining a mandatory current-word-mean baseline and a clear novelty boundary for the TXC experiment.

[Barenholtz (2026)](https://arxiv.org/abs/2606.05346) is an even closer temporal baseline. At each word, the paper fits a linear trajectory through the preceding hidden states and measures the Euclidean error of the one-step extrapolation. A three-word linear fit predicts Natural Stories reading time beyond lexical controls and surprisal, becomes stronger at garden-path disambiguation, and replicates from GPT-2 to Pythia. The paper's direction-preservation analysis also separates middle and final layers: middle-layer directional continuity largely dies after one word, whereas final-layer direction persists over several words. This strongly motivates an ordered activation gate, but it occupies the generic claim that local hidden-state trajectory deviation predicts reading time. Our distinct question is whether a frozen sparse TXC recovers this behaviorally relevant direction more usefully or interpretably than raw extrapolation error, endpoint displacement, and explicit finite differences.

Let \(x_t\) be a model-derived processing impulse at word \(t\), and let \(y_t\) be log reading time. A finite-lag approximation is

\[
y_t = b(z_t) + \sum_{\ell=0}^{L} k_\ell^\top x_{t-\ell} + \epsilon_t,
\]

where \(z_t\) contains lexical and display controls. The endpoint baseline assumes \(k_\ell=0\) for \(\ell>0\). An order-invariant history can retain the multiset \(\{x_{t-\ell}\}\) but cannot identify which value occurred at which lag. A TXC can instead learn position-specific filters

\[
f_j(t)=\sigma\!\left(\sum_{\ell=0}^{T-1}
w_{j,\ell}^{\top}h_{t-\ell}\right),
\]

which are matched to an oriented spillover waveform. In the spectral view, \(k\) has transfer function \(K(\omega)\), while the slow power-law activation background determines the input spectrum. This gives frequency analysis a concrete role: it asks which temporal bands of the model trajectory explain a measured human response, rather than treating power-law autocorrelation itself as evidence that TXCs must help.

### Frozen lexical opportunity gate

Before extracting activations, the corrected Natural Stories release at git revision `4700daad696e942f5aba23c957a7423d0de66612` was aligned to its released GPT-3 token log probabilities and item-level geometric-mean reading times. Leave-one-story-out ridge models predicted aggregate log reading time from word length, unigram frequency, GPT-3 surprisal, punctuation, and story position. The ordered view exposed the current word and five preceding words. The strongest parameter-matched invariant controls either lexicographically canonicalize the intact historical word vectors or summarize the same set by moments; neither invents covariate combinations or retains lag identity. Regularization was selected only inside training stories. A post-hoc audit caught that the original punctuation regex was all zero because punctuation remains attached to released words; every result below was recomputed with current and lagged punctuation encoded correctly.

| View, five-word history | Held-out MSE | Held-out Pearson \(r\) |
|---|---:|---:|
| Endpoint | .006088 | .557 |
| Explicit first difference | .005518 | .612 |
| Intact-vector invariant history | .005567 | .608 |
| Set-summary invariant history | .005596 | .605 |
| Ordered history | **.005262** | **.635** |
| Ordered fit, shuffled history at test | .005912 | — |
| Ordered fit, reversed history at test | .006202 | — |

The ordered MSE improvement over the intact-vector invariant view is .000304 with a whole-story bootstrap 95% interval [.000240, .000364]. It beats the explicit first difference by .000263 [.000125, .000399]. Shuffling and reversing the held-out historical positions increase MSE by .000652 [.000571, .000739] and .000940 [.000846, .001035], respectively. Ordered history beats the intact-vector invariant view in all ten stories. The advantage first appears with two prior words and grows through six.

This is evidence that the **label itself** contains an ordered local kernel; it is not yet evidence that hidden activations or TXC features recover that kernel.

### Activation and dictionary gate

Use `google/gemma-2-2b-it` layer 13 so the task can reuse the existing \(T=5\) dictionaries. Reconstruct every story causally and align each displayed word to model subtokens. The primary model-token window ends at the last subtoken of the current word. Run \(T=1,\ldots,6\) raw views first: final-subtoken endpoint, Tsipidi-style current-word subtoken mean, every offset, explicit first and second differences, one-step displacement, Barenholtz-style linear trajectory extrapolation error, parameter-matched invariant mean/std and canonical-set summaries, ordered flattening, reverse, within-window shuffle, and boundary jitter. Split by whole story and retain the standard current and lagged word-length, frequency, punctuation, position, and subject-model surprisal controls.

The raw gate passes only if the ordered activation view improves held-out reading-time prediction beyond the strongest lexical-plus-surprisal baseline and the strongest raw endpoint/invariant/difference view, with a whole-story interval above zero, and loses the gain under reversal or shuffle. Only then encode the same windows with the frozen TXC and SAE dictionaries. The confirmatory analysis should use participant-level log reading times with subject intercepts rather than the aggregate screening target. Replication on Natural Stories Maze or an eye-tracking corpus would distinguish cognitive spillover from button-press dynamics.

### Eye-movement backtracking as a sharper replication

The first independent replication is already complete on [Natural Stories A-Maze](https://doi.org/10.5070/G6011190), which collected a separate 95-participant forced-choice reading experiment on the same ten stories. Mirroring the released cleaning code leaves 62 attentive readers, 54,523 trials, and 9,777 aggregate word targets. At five prior words, ordered history obtains MSE .036524 and \(r=.605\), compared with .036984 and .599 for the intact-vector invariant control. The whole-story ordered gain is .000466 [.000074, .000858]. Reversal raises MSE to .038166, a penalty of .001636 [.001367, .001888], and shuffle raises it to .037715. Ordered and explicit first-difference performance are statistically indistinguishable in A-Maze, so this replication supports **lag identity** without claiming that a general nonlinear trajectory is required.

![Natural Stories ordered-spillover replication](../../../results/task_screening/naturalstories_spr_maze_replication.png)

The stimuli contain one documented same-length Story 2 difference, `peaked` in the self-paced release versus `peeked` in A-Maze. It is preserved explicitly rather than silently sharing the wrong subject-model trajectory. The two behavioral paradigms can otherwise share the same extraction and split infrastructure.

The [Provo Corpus](https://doi.org/10.3758/s13428-017-0908-4) supplies raw fixation sequences from 84 readers over 55 short English passages, so it supports a still more literal backtracking target: whether the next fixation moves to an earlier word. [Rego et al. (2026)](https://doi.org/10.1016/j.cognition.2026.106535) show that language-model surprisal helps explain both where regressions begin and, together with gradient saliency, where they land; they also explicitly test previous-word spillover. [Madureira et al. (2023)](https://aclanthology.org/2023.conll-1.22/) independently connect human regressions to revision events in incremental NLP systems. Most directly, [Duan et al. (2026)](https://aclanthology.org/2026.scil-main.44/) report that phrase-level attention predicts regressive saccades better than word-level attention and survives a shuffled phrase-boundary control. These results make multi-token internal structure a motivated hypothesis rather than a generic sequence-model baseline.

The primary screen predicts the **source hazard** \(P(R_t=1)\), where \(R_t\) records whether a reader's next valid inter-word saccade from word \(t\) is backward. It compares endpoint, current-word subtoken mean, explicit differences, intact-vector invariant history, and ordered history under passage-held-out evaluation, first on per-word regression probability and then on participant-level events with reader effects. First-fixation and line-wrap corrections must follow the published cleaning rules. A sharper secondary screen conditions on a backward saccade whose destination lies within the previous five words and predicts its destination offset. That categorical target is intrinsically oriented: an invariant set knows which representations occurred but not which candidate occupied offset \(-1,\ldots,-5\).

A strict complete-covariate destination screen now retains 20,091 regression events from 73 readers, 2,209 distinct source words, and all 55 passages. The five classes are imbalanced—13,715 events land one word back and only 254 land five words back—so multiclass log loss, not raw accuracy, is the primary metric. Ten-fold passage-held-out logistic models with nested regularization give:

| Five-word source history | Held-out log loss | Balanced accuracy |
|---|---:|---:|
| Empirical distance prior | .898 | .200 |
| Source-word endpoint | .893 | .200 |
| Endpoint plus first difference | .852 | .206 |
| Intact-vector invariant history | .889 | .200 |
| Ordered history | **.840** | **.210** |
| Ordered fit, reversed history at test | .930 | .205 |
| Ordered fit, shuffled history at test | .954 | .202 |

The ordered log-loss gain over the explicit first difference is .0119 with passage-clustered 95% interval [.0076, .0158], reader-clustered interval [.0087, .0135], and source-word-clustered interval [.0106, .0187]. Against the intact-vector invariant view, the corresponding estimates are .0508 [.0385, .0614], .0526 [.0463, .0590], and .0381 [.0293, .0466]. The effect is calibration-sensitive rather than a large top-one classification gain, but it survives every clustering unit and becomes worse under fixed-fit reversal and shuffle.

![Provo regression-destination text gate](../../../results/task_screening/provo_regression_destination_text_gate.png)

This passes the task-side opportunity gate: the destination task genuinely requires lag identity beyond the empirical distance prior, source endpoint, explicit first difference, and two invariant summaries. It is not yet evidence that raw activations or a TXC recover the information. The activation gate must teacher-force each passage, represent the five candidate destination words and source with exact word-to-subtoken alignment, and retain reader-, passage-, and source-clustered inference. Reversing the activation window changes the claimed destination while preserving every activation vector, making it an especially diagnostic control.

**Recommendation:** this is the highest-priority independent task once a replacement GPU is available. It has enough observations for a real benchmark, a separately established temporal mechanism, a passed ordered-label gate, and an exact match to existing Gemma dictionaries.

## 0D. Early reasoning instability and final correctness

### Why a rollout-level label can be legitimate here

[Chrabąszcz et al. (2026)](https://arxiv.org/abs/2605.18549) evaluate the exact DeepSeek-R1-Distill-Llama-8B subject model on real GSM8K and MATH generations. They apply a concept probe cumulatively across generated tokens and show that eventual mathematical correctness is better separated by the **shape of the probe trajectory** than by a static probe. Their reported gains reach 17 AUROC points on GSM8K, and important math features include first-difference variance, second-difference variance, slope, and mean-crossing rate. Their “first 5%” analysis is not admissible for our online claim because its cutoff is computed from completed response length; it is evidence for trajectory statistics, not clean evidence that a fixed early prefix predicts correctness.

[Sun et al. (2026)](https://arxiv.org/abs/2604.05655) provide direct adverse evidence. Across Llama-3.1-8B variants including DeepSeek-R1-Distill-Llama-8B, correct and incorrect trajectories follow similar early paths and diverge late. In their fixed-step GSM8K experiment, early Step-1/2 features obtain only about 0.61--0.63 AUROC, while late-transition features average 0.83 and peak near 0.87; the final answer-marker activation alone averages 0.81. Their [code](https://github.com/slhleosun/reasoning-trajectory) also makes clear that the strong signal uses step-boundary representations, not six adjacent tokens. This lowers the prior for our early-prefix pilot and raises the endpoint and reasoning-to-answer-transition baselines that any late variant must beat.

This is the principled exception to the rule that TXC targets should be local. It would still be wrong to assign the final-correctness label to every individual window and pretend that all windows are erroneous. Instead, let a rollout contain non-overlapping local windows \(W_1,\ldots,W_m\), compute representation-specific window codes \(a(W_i)\), and train one **bag-level** classifier

\[
\widehat y
=g\!\left(
\operatorname{Agg}_{i\leq m} a(W_i)
\right),
\]

where \(y\) is mechanical final-answer correctness and `Agg` is held fixed across TXC, T-SAE, and SAE. The claim is then that a failed rollout contains a different distribution of local temporal events—oscillation, abrupt reversal, or instability—not that every local window is itself mislabeled.

The current exact-model [MATH trace corpus](https://huggingface.co/datasets/jrosseruk/DeepSeek-R1-Distill-Llama-8B-MATH-traces-balanced) already supplies 4,492 traces over 500 problems, with 2,500 correct and 1,992 incorrect final answers. The 230 problems containing both outcomes supply 2,322 traces. Of these traces, 204 lack `</think>` and all 204 are incorrect, so excluding them based on that future marker would create severe outcome-conditioned selection. We instead crop at `</think>` when present, treat a missing marker as a still-open or truncated CoT, require the requested fixed prefix to exist, and use problem-and-label-balanced training weights rather than outcome downsampling. No new generation is required for the first gate, although the released exact-match labels should be audited with semantic math verification because formatting errors are outcome-label noise.

### Connection to TXC, the power law, and frequency

Write a TXC feature at endpoint \(t\) as

\[
a_{j,t}
=\phi\!\left(
\sum_{\tau=0}^{T-1}d_{j,\tau}^{\top}h_{t-\tau}-b_j
\right).
\]

Its position profile \(d_{j,\tau}\) is a learned finite-impulse-response filter. If ordinary residual dynamics have a long-memory background \(S_h(\omega)\propto |\omega|^{\alpha-1}\), a profile with small DC response,
\(\sum_{\tau}d_{j,\tau}\approx0\), suppresses that low-frequency background while responding to localized changes. Incorrect reasoning can then be modeled as an added transient process with greater derivative or band-limited energy. This predicts that error-relevant TXC features should have more high-pass or band-pass spectral mass, activate in sparse bursts, and improve a bag-level max/top-\(k\) classifier over a static SAE evaluated at the same endpoints.

The paper's cumulative max probe is itself an envelope detector: once a salient event occurs, max pooling preserves it instead of diluting it across neutral tokens. That gives a natural outer aggregation for TXC features, but it must be compared against the same max pooling over SAE and T-SAE features. Any advantage from simply seeing more positions belongs to the pooling baseline, not to TXC.

### Minimal exact-model pilot

Use problem-grouped folds so traces from the same MATH problem never cross train and test. Restrict every representation to the same first-\(q\) prefix for \(q\in\{32,64,128\}\) generated tokens and require only that those \(q\) reasoning tokens exist. Do not require extra future length or match on completed length: both are post-prefix variables that can erase or induce the target association. Run each \(q\) on its own eligible cohort, with a common “survives to 128” cohort only as a paired-prefix sensitivity. Form windows for \(T=1,\ldots,6\) at every possible phase, or demonstrate robustness across phases. For every representation, evaluate:

- the final endpoint alone;
- maximum and top-\(k\) mean across window scores;
- mean and variance across windows;
- explicit first- and second-difference statistics;
- a capacity-matched bag-level logistic probe;
- shuffled-within-window and reversed-window TXC codes;
- shuffled window order for any outer sequential statistic.

The pre-registered primary comparison is \(q=64,T=6\): TXC against identically pooled SAE \(T=1\), T-SAE, the best individual offset, raw endpoint residuals, raw invariant pooling, explicit raw velocity/acceleration, and early-text TF-IDF. Pooling and probe regularization are selected only in nested training folds. The primary metric is correct-versus-incorrect ranking within the same held-out problem, averaged across problems, with confidence intervals from bootstrapping whole problems. The pilot passes only if TXC reaches at least 0.65 AUROC, improves by at least 0.03 over the strongest locked baseline with a problem-clustered 95% confidence interval above zero, and loses at least 0.02 when token order is destroyed within each \(T\)-window without a material reconstruction penalty. The advantage must have the same sign at \(q=32\) or \(q=128\). A result that depends only on the number of windows or on a random-forest trajectory classifier would reproduce the prior paper's global dynamics claim without establishing a TXC advantage.

### Text-only opportunity gate

A reproducible 2026-07-24 CPU gate crops at `</think>` when present, treats a missing marker as open/truncated reasoning, requires only the requested fixed prefix, and assigns every rollout for a problem to one of five folds. Its primary fit gives every training problem equal mass and splits that mass equally across its two outcomes. The score is the correct-versus-incorrect pairwise AUROC within each held-out problem, averaged equally over problems; the interval bootstraps whole problems. With seed 42, a whitespace-word/bigram TF-IDF probe obtains:

| Fixed prefix | Traces / problems | Global OOF AUROC | Macro within-problem AUROC |
|---:|---:|---:|---:|
| 32 words | 2,311 / 229 | 0.558 | 0.548 [0.511, 0.583] |
| 64 words | 2,260 / 228 | 0.556 | 0.557 [0.519, 0.593] |
| 128 words | 1,744 / 189 | 0.572 | 0.538 [0.498, 0.580] |

Class-balanced-fit sensitivity gives within-problem AUROC 0.562, 0.546, and 0.535 respectively. Early surface text therefore carries a small signal at 32–64 words but does not solve the task. The final comparator must repeat this baseline with exact subject-tokenizer prefixes so its evidence budget matches the activation models.

Percentage-of-completed-rollout prefixes remain inadmissible: their cutoff uses future response length, and incorrect traces are much longer in this corpus (median 3,143 versus 2,485 characters; mean 11,522 versus 5,159). Completed-length matching is also inappropriate for the primary online task because it conditions on a downstream variable. These controls make the raw hidden-state gate worthwhile, but they do not themselves establish an activation-level temporal advantage.

This task also resolves the EM-label discussion cleanly: one label is attached to one bag, not copied to many locally neutral examples. If it works, the same multiple-instance protocol can be applied to emergent misalignment, with sparse max/top-\(k\) pooling allowed to select the few genuinely misaligned windows.

**Recommendation:** run one bounded raw gate because the data and implementation are ready, but do not train or reuse dictionaries unless it clears the frozen threshold. Existing backtracking dictionaries come from Llama-3.1-8B-base and are invalid for these DeepSeek activations despite matching dimensionality; any learned comparison needs fresh same-model dictionaries trained without outcome-conditioned sampling.

## 0E. Reasoning-loop onset

### Why this is unusually well matched to a temporal representation

[Pipis et al. (2026)](https://arxiv.org/abs/2512.12895) provide the strongest exact-model mechanism. On AIME 2024/2025, they call a trace a loop when any 30-token \(n\)-gram occurs at least 20 times. DeepSeek-R1-Distill-Llama-8B loops on 54% of greedy traces under that definition, robustly 41--56% across nearby detector settings. They explain the failure through two coupled effects: probability mass for a difficult progress action diffuses across many alternatives while the easier cyclic action retains concentrated mass, and Transformer dynamics correlate errors across time so a repeated choice reinforces itself at low temperature.

[Duan et al. (2026)](https://arxiv.org/abs/2601.05693) independently characterize a loop as an entropy collapse into a low-dimensional periodic orbit. Their sentence-level residual-state CUSUM detector on the exact Llama distill reports a mean lead of 46.2 sentences, but it uses a five-sentence statistic and therefore supports a transition-state hypothesis rather than a six-adjacent-token TXC result. [Xie et al. (2025)](https://arxiv.org/abs/2511.00536) release [Word Salad Chopper](https://github.com/wenyaxie023/WordSaladChopper) traces and show that one trailing delimiter state already detects broad semantic repetition well. That is important adverse evidence: contemporaneous loop detection may be an endpoint task unless strict pre-onset windows expose something earlier.

Model the residual path as a long-memory background plus an emerging orbit,

\[
h_t=b_t+A(t)u\cos(\omega_0t+\phi)+\epsilon_t.
\]

The orbit adds a narrow-band component to the slow multiscale background. A TXC can act as a finite impulse-response detector for its local phase only if the relevant period or transition waveform is visible inside \(T\); a distant periodicity cannot be inferred from the marginal power law. This is a principled bridge from spectral structure to a task, but it makes explicit autocorrelation, entropy, endpoint, and repeated-token controls mandatory.

The event has two distinct boundaries. The **retrospective onset** is the first occurrence of the pattern that eventually repeats 20 times; it uses future information only to define the event, while the evaluated pre-onset window remains future-free. The **online confirmation** is the end of the twentieth occurrence. The latter is operationally observable but is likely trivial from textual repetition; the former is scientifically stronger but may not be locally predictable. They must not be reported as the same task.

The raw baselines must include the last state, best offset, invariant mean/max, explicit autocorrelation at every lag up to \(T-1\), token identity, repeated-subtoken flags, entropy, and raw first/second differences. Reverse and shuffle positions within the window, jitter \(t^*\), and compare against ordinary repeated phrases that do not lead to a sustained loop. A TXC win over SAE but not over explicit autocorrelation would establish a temporal phenomenon without establishing a useful learned representation.

### Public-corpus prevalence audits

Three exact-model audits show why this is not a free rebuttal benchmark:

| Corpus | Traces | Strict 30-gram \(\times20\) loops | Rate |
|---|---:|---:|---:|
| Strategic-TTC AIME, 2K cap | 29,856 | 53 | 0.178% |
| OpenThoughts math, 32K cap | 22,037 | 39 | 0.177% |
| Word Salad Chopper release | 1,000 | 38 | 3.8% |

The Word Salad Chopper semantic recipe is broader: 834/1,000 traces contain at least one semantic duplicate chunk, but only 89/1,000 cross the released sustained word-salad boundary. Its boundary is retrospective because ten consecutive duplicate-like chunks are required, and its published detector uses a single delimiter state. In the strict mechanical audits, the median selected loop period is seven tokens in the 2K corpus, 124 tokens in OpenThoughts, and 52 tokens in Word Salad Chopper. The latter two periods are far outside \(T=6\), so only an internal onset transition—not direct cycle coverage—could rescue the current architecture.

The viable experiment is therefore low-temperature regeneration of Pipis et al.'s public AIME prompts: 20 traces per problem at temperature 0.2 or 0.4, a long cap, frozen 30-gram detector, and problem-grouped evaluation. Advance only if at least 200 independent traces contain a clean event. Match positives to earlier windows in the same trace on absolute/relative position and local entropy, then add non-loop traces from the same problem. The raw baselines must include the last state, best offset, invariant mean/max, explicit autocorrelation at every lag up to \(T-1\), token identity, repeated-subtoken flags, entropy, and first/second differences. Reverse and shuffle positions, jitter the onset, and compare against ordinary repetition that does not become a sustained loop.

**Recommendation:** retain as the cleanest mechanism experiment, but do not spend activation compute on the available public corpora. It requires deliberate low-temperature generation and is therefore behind data-ready speech repair and self-check onset for the reviewer deadline.

## 0F. Incremental speech-repair onset

### Why it fits \(T=6\) rather than merely involving a sequence

[Hough and Schlangen (2017)](https://aclanthology.org/E17-1031/) formulate incremental speech repair as a three-state path: reparandum, optional interregnum, and repair. Their gold tag at the first repair word records how many words back the reparandum begins (`rpS-1` through `rpS-8`). The released [Deep Disfluency repository](https://github.com/dsg-bielefeld/deep_disfluency) includes token-level Switchboard annotations and fixed train/held-out/test divisions. This is real conversational language, independent of math reasoning, and the label itself describes a directed local transition rather than a persistent sentence or rollout property.

A deterministic counts-only audit of the public non-partial-word files found 742,496 tokens from 597 conversations and 24,586 repair onsets. The reparandum start is at most six words before the repair onset in 98.9% of events. Under the DeepSeek-R1-Distill-Llama-8B tokenizer, the entire substring from reparandum start through repair onset fits inside six model tokens for 95.1% of events; its median length is two tokens and 75th percentile is four. This is the strongest literal receptive-field match found in the search.

Surface shortcuts are substantial but controllable. Of repair onsets with six preceding words, 70.3% repeat a recent word and 43.9% follow an edit term, compared with 5.6% and 23.4% for fluent tokens. The primary cohort therefore matches each onset to fluent endpoints in the same conversation with the same current word, current-word repetition flag, recent-edit-term flag, and within-utterance position bucket. After deduplicating exact windows and forbidding reuse of a fluent negative across positive events, this leaves 2,325 positive onsets, 5,824 total examples, and 537 conversation groups. An even harder subset with neither a recent repeated word nor an edit term contains 1,509 positive events before exact-word matching.

### Text gate and temporal prediction

On the exact-current-word matched cohort, five-fold conversation-grouped text probes obtain:

| Text view | OOF AUROC | OOF average precision |
|---|---:|---:|
| Current word only | 0.526 | 0.420 |
| Six-word bag of unigrams | 0.596 | 0.470 |
| Ordered word 1--3 grams | 0.730 | 0.663 |
| Ordered character 3--5 grams | 0.751 | 0.681 |

The 19.3-point AP gap from the invariant word bag to ordered word \(n\)-grams is direct evidence that this matched task depends on local order. It does not establish an activation or TXC advantage, and the strong character baseline must remain in the final comparison.

The primary activation task is **contemporaneous repair-onset detection**: a six-token window ends at the final subtoken of the first repair word. Calling the preceding reparandum “predictive” would be wrong; the incremental-disfluency literature explicitly notes that a reparandum often becomes identifiable only when repair begins. A strict window ending before the repair word is therefore a negative-control anticipation task, not the headline.

Teacher-force the public transcript through the subject model and split all examples from one Switchboard conversation together. Compare raw endpoint, best offset, invariant mean/std and learned pooling, explicit token-repetition and edit-term flags, ordered concatenation, reverse/shuffle controls, TXC, and an identically exposed SAE. The candidate advances only if ordered raw activations beat the endpoint and strongest invariant view by at least 0.02 AP with a conversation-clustered interval above zero, and if reverse or shuffle removes at least 0.02 AP. Any learned TXC claim must additionally beat the ordered text baseline or be framed narrowly as a sparse representation result.

The completed exact-model raw gate retained 2,014 positive onsets and 2,976
fluent controls across 510 conversation-grouped folds after exact model-window
deduplication. Ordered \(T=6\) AP is .943, exactly equal to the endpoint;
explicit differences reach .945, invariant mean/std .737, and the strict
pre-word endpoint and ordered views both reach .756. Reversal, shuffle, and a
one-token-earlier anchor destroy the fitted ordered classifier, but this only
confirms that it relies on the correctly positioned endpoint.

**Recommendation:** no-go. The event is real, localized, and highly decodable,
but it does not require a temporal representation under the frozen gate.

### Destination-distance pivot

The failed onset task asked whether the current word begins a repair, but the
current-word hidden state was sufficient. The sharper task uses only the prefix
strictly before that word and predicts the annotated offset from the upcoming
repair onset (`rps`) back to its reparandum start (`rms`):

\[
y_t=t-\operatorname{rms}(t)\in\{1,\ldots,5\}.
\]

Nested and interleaved repair IDs are excluded. The primary surface screen
contains 9,030 repairs from all 597 conversations, with class counts
\(5{,}179, 2{,}022, 1{,}003, 551, 275\). Conversation-grouped probes at
\(T=5\) obtain:

| Strict-pre-onset view | Held-out log loss | Balanced accuracy |
|---|---:|---:|
| Distance prior | 1.175 | .200 |
| Last word only | .980 | .323 |
| Order-invariant word bag | 1.070 | .233 |
| Parameter-matched canonical multiset | 1.083 | .219 |
| Ordered five-word history | **.907** | **.330** |
| Ordered fit, reversed at test | 1.195 | .204 |

Equal-conversation bootstrap contrasts are .0723 [.0683, .0765] for
endpoint-minus-ordered log loss and .1586 [.1508, .1664] for bag-minus-ordered;
562/597 and 570/597 conversation-level contrasts are positive. Removing every
repair containing an edit term leaves 7,136 examples and preserves the ordered
gain: .779 log loss versus .823 for the endpoint, .835 for the bag, and .855
after reversal.

On one fixed length-eligible surface cohort, the ordered curve is .959, .915,
.903, .898, .896, and .895 for \(T=1,\ldots,6\), while the identical endpoint
remains .959. The improvement saturates at the five-position label horizon,
which is the scale-specific behavior the TXC theory predicts rather than an
indefinite benefit from adding context.

The exact DeepSeek tokenizer audit then redefines the label as the model-token
offset from the strict pre-repair anchor to the first reparandum subtoken. Of
19,342 simple annotated boundaries, 7,188 fit within \(T=5\). Removing
same-label duplicate token windows and every conflicting-label duplicate leaves
7,071 unique windows from all 597 conversations, with class counts 3,904,
1,476, 809, 528, and 354. The categorical model-token gate is:

| Exact-token view, \(T=5\) | Held-out log loss | Balanced accuracy |
|---|---:|---:|
| Distance prior | 1.247 | .200 |
| Endpoint token | 1.076 | .291 |
| Order-invariant token bag | 1.119 | .240 |
| Parameter-matched canonical multiset | 1.156 | .221 |
| Ordered token history | **1.000** | **.307** |
| Ordered fit, reversed at test | 1.240 | .202 |

Endpoint-minus-ordered equal-conversation log loss is .0762 [.0715, .0810],
positive in 569/597 conversations. Removing all edit-term repairs retains
5,669 events and an ordered log loss of .879 versus .937 for the endpoint and
.928 for the invariant bag. Ordered loss decreases monotonically from 1.076
at \(T=1\) to 1.035, 1.018, 1.007, and 1.000 at \(T=2,\ldots,5\); the endpoint
is unchanged.

![Exact-token speech-repair destination sweep](../../../results/task_screening/switchboard_repair_destination_model_token_sweep.png)

The replacement-GPU version can now consume the frozen token-ID-only Parquet
without rebuilding or storing transcript text. It must compare every offset,
endpoint displacement, first/second differences, trajectory residual,
invariant pooling, ordered raw states, reversal, shuffle, TXC, and identically
exposed SAE features under conversation-grouped folds. The task advances to a
TXC claim only if the raw ordered view first beats the best single offset and
invariant/difference controls.

## 0G. Human writing-revision destination

### Why this is a second literal destination task

[Tian, Crossley, and Van Waes (2025)](https://doi.org/10.17239/jowr-2025.17.01.02)
release KLiCKe, keystroke logs for 4,992 argumentative essays. Each row records
the operation, its time, cursor position, exact text change, and an activity
label such as input, removal, paste, or replacement. This makes the behavioral
boundary mechanical: immediately before the first operation in a consecutive
deletion burst at the leading edge, use the current local text to predict where
the writer will stop deleting.

If the final word before deletion is \(w_t\) and the burst removes \(D_t\)
complete lexical words, define

\[
y_t=\min(D_t,5),\qquad D_t\geq2,
\]

so \(y_t\in\{2,3,4,5+\}\). For \(D_t\leq5\), the label identifies the candidate
boundary \(w_{t-D_t+1}\) within the five-word pre-deletion window. Unlike a
rollout-level outcome, each class therefore corresponds to a different
oriented location in the same local history. This is the structural condition
under which a TXC has a reason to beat a per-token SAE.

### Conservative corpus reconstruction

The extraction replays simple inputs, pastes, and removals from the public CSV
members. A session is truncated at its first replacement, text-move, unknown
activity, or cursor/text mismatch rather than guessing how a complex edit
changed the document. Candidate bursts must delete a suffix of the current
document, contain at least two complete words, and leave a clean lexical
boundary. No essay text is written to the result artifact.

Across 11,692,552 logged operations, all 4,992 files parse successfully.
Complex operations conservatively truncate 2,148 sessions, and cursor or
removal mismatches truncate another 225. The remaining prefixes yield 22,071
eligible events. Global exact-window deduplication removes 74 same-label
duplicates and all 54 rows belonging to conflicting-label duplicates, leaving
21,943 events from 3,923 writers. Class counts for two, three, four, and five
or more deleted words are 10,986, 4,330, 2,284, and 4,343.

Writer-grouped five-fold probes obtain:

| Strict pre-deletion view, \(T=5\) | Held-out log loss | Balanced accuracy |
|---|---:|---:|
| Destination prior | 1.223 | .250 |
| Pause/position metadata | 1.208 | .258 |
| Last word only | 1.214 | .250 |
| Order-invariant word bag | 1.206 | .250 |
| Parameter-matched canonical multiset | 1.212 | .250 |
| Ordered five-word history | **1.154** | **.254** |
| Ordered fit, reversed at test | 1.226 | .250 |

Because the natural classes are imbalanced, calibrated multiclass log loss is
primary. Equal-writer endpoint-minus-ordered log loss is .0611 with a 95%
bootstrap interval [.0572, .0649], positive for 2,934/3,923 writers.
Bag-minus-ordered is .0537 [.0496, .0574], and reversing the ordered input
costs .0749 [.0703, .0796]. Restricting to the 21,467 bursts made entirely of
single-character Backspace operations preserves the result: 1.146 ordered
versus 1.206 endpoint, 1.198 bag, and 1.217 reversed.

| Surface-word \(T\) | 1 | 2 | 3 | 4 | 5 |
|---:|---:|---:|---:|---:|---:|
| Ordered log loss | 1.214 | 1.192 | 1.170 | 1.160 | 1.154 |
| Endpoint log loss | 1.214 | 1.214 | 1.214 | 1.214 | 1.214 |
| Order-invariant bag | 1.214 | 1.193 | 1.194 | 1.201 | 1.206 |

![KLiCKe writing-revision destination sweep](../../../results/task_screening/klicke_trailing_deletion_destination_window_sweep.png)

The monotone ordered curve and degradation under reversal are the exact
task-side pattern predicted by an oriented finite-window representation.
There is also a plausible frequency interpretation: normal text supplies a
slow, correlated background, while a locally ill-formed or abandoned suffix
creates a boundary-specific residual that a learned position profile can
localize. The result does not show that a language model encodes the human
writer's decision, and the human decision can depend on cognitive state absent
from the text. It establishes only that the public task itself contains
ordered local linguistic signal unavailable to endpoint and invariant
controls.

### Exact-token and activation gates

Before using a GPU, reconstruct the same frozen events with the exact subject
tokenizer. Tokenize both the complete pre-deletion prefix and the retained
prefix, require exact prefix-tokenization stability, define the destination by
their token-length difference, deduplicate exact token-ID windows globally,
and repeat \(T=1,\ldots,6\). This prevents a five-word result from being
misreported as a five-token TXC result.

Only if that audit passes should the raw gate teacher-force the pre-deletion
prefix and compare every offset, the endpoint, invariant mean and dispersion,
first and second differences, an extrapolation residual, ordered
concatenation, reversal, and deterministic shuffling under writer-grouped
folds. Frozen TXC and SAE features run only if the ordered raw activation
window beats the strongest static, invariant, and difference control.

### Exact-token and raw-activation result

The exact-token audit and raw gate now pass. Requiring the post-deletion token
sequence to be an exact prefix of the pre-deletion sequence, globally
deduplicating the final ten-token history, and dropping conflicting targets
leaves 6,224 events from 2,510 writers. The primary label is the number of
subject-model tokens deleted, capped at six; its class counts for
\(2,3,4,5,6+\) are 1,218, 1,634, 1,058, 692, and 1,622.

We teacher-force each retained prefix through Llama-3.1-8B and cache the final
ten layer-10 residual states. Extraction uses unpadded singleton inference:
repeated forwards of both the shortest and longest audited prefixes agree
bit-for-bit. The probe uses writer-grouped five-fold evaluation and selects 64
hidden coordinates using only each outer training fold, applying the same
coordinates at every temporal position.

For the primary token-distance target, ordered history improves through
\(T=6\) and then degrades:

| Token \(T\) | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Ordered log loss | 1.423 | 1.392 | 1.333 | 1.281 | 1.243 | **1.239** | 1.250 | 1.268 | 1.287 | 1.304 |
| Endpoint log loss | 1.423 | 1.423 | 1.423 | 1.423 | 1.423 | 1.423 | 1.423 | 1.423 | 1.423 | 1.423 |
| Invariant mean/std/max | 1.426 | 1.445 | 1.454 | 1.461 | 1.476 | 1.492 | 1.511 | 1.531 | 1.541 | 1.549 |

At \(T=6\), the ordered view obtains .475 balanced accuracy versus .326 for
the endpoint. Equal-writer endpoint-minus-ordered log loss is .2012 with a 95%
bootstrap interval [.1774, .2255]. Invariant-minus-ordered is .2891
[.2627, .3148], explicit second-difference-minus-ordered is .1047
[.0834, .1250], and a probe retrained on deterministically shuffled histories
is worse by .4159 [.3859, .4469]. Fixed-fit reversal and shuffle are more
destructive still, with log losses 2.148 and 1.994 versus 1.239 ordered.

![KLiCKe exact-token activation sweep](../../../results/neurips_rebuttal/writing_revision_destination/publication_singleton_v1/token_distance.png)

The original lexical 2/3/4/5+ label is a weaker secondary target: its best
ordered result occurs at \(T=4\), 1.175 versus 1.229 for the endpoint and
1.205 for the invariant view, but its advantage over the train-selected best
single offset has an interval crossing zero. The exact-token target is
therefore the defensible benchmark.

**Recommendation:** promote this to the leading independent semi-synthetic
task. It is mechanically labeled, writer-grouped, and shows a finite useful
history scale plus decisive order-destruction controls in real model
activations. This is still a raw-representation gate rather than a TXC result;
the next experiment is the preregistered frozen TXC-versus-SAE comparison on
the same cohort.

## 0H. First reasoning-error localization

### Why it directly tests temporal dynamics

[ProcessBench](https://arxiv.org/abs/2412.06559) contains 3,400 mathematical reasoning traces whose earliest incorrect step was annotated by human experts. The [public data](https://huggingface.co/datasets/Qwen/ProcessBench) cover GSM8K, MATH, OlympiadBench, and Omni-MATH. Of the released traces, 2,221 have a localized first error and 1,179 are fully correct; 620 were generated by Llama-3/3.1 8B-family models, although none by our exact DeepSeek-R1-Distill-Llama-8B subject.

[Alvarez and Baheri (2026)](https://arxiv.org/abs/2605.13772) explicitly model a first error as a hidden-trajectory excursion. For step representation \(z_i\), their useful features include velocity \(v_i=z_i-z_{i-1}\), acceleration \(a_i=v_i-v_{i-1}\), local energy, and directional persistence. They report large gains over a static linear probe on ProcessBench and related datasets. This is direct precedent for the claim that a reasoning error can be a **transition property** rather than a static state label.

The precedent needs to be interpreted cautiously. Their \(z_i\) mean-pools a whole reasoning step, their label-conditioned teacher is not deployable, and their BiLSTM student sees the complete trace rather than an online local window. The paper has no order-shuffle, invariant-pooling, or matched-capacity sequence ablation. Its deployable student reaches 75.0 AUROC versus 67.8 for the static probe on ProcessBench, but its first-error localization accuracy is only 34.4, below entropy at 46.3 and attention at 43.8; cross-model student AUROC falls to 33.4–58.5. The result therefore motivates an ordered raw gate but does not establish that a six-adjacent-token TXC should win.

### Connection to the power-law and spectral story

If ordinary reasoning has long-memory covariance \(C(\Delta)\propto\Delta^{-\alpha}\), then its low-frequency background spectrum scales approximately as \(S(\omega)\propto|\omega|^{\alpha-1}\) for \(0<\alpha<1\). A localized first-error excursion adds comparatively high-frequency energy. The discrete first- and second-difference operators used above have frequency responses

\[
\widehat v(\omega)=(1-e^{-i\omega})\widehat z(\omega),
\qquad
\widehat a(\omega)=(1-e^{-i\omega})^2\widehat z(\omega),
\]

so they suppress the slowly varying background and amplify localized transitions. A TXC feature with an oriented position profile can act like a learned finite-impulse-response filter over a short window; a spectral penalty can favor filters according to where their response lies relative to the empirical background spectrum. This gives a concrete bridge from the observed power law, through frequency-selective features, to a task where high-frequency transition energy should matter.

### Proposed raw gate

The primary task is **contemporaneous first-error detection**, not prediction before the error exists. Let \(e\) be the first human-annotated incorrect step and \(t_e\) its final token. Predict \(y=1\) from windows ending at \(t_e\); negatives should be the immediately preceding correct transition and step-index-matched transitions from fully correct traces. There are 1,949 error traces with a preceding correct step, so clean within-trace pairs are available; 272 step-zero errors must be excluded. Keep all traces for one problem in one fold, stratify by source benchmark and generator, and report leave-one-generator-family-out transfer. The source solutions were resegmented into steps by Qwen2.5-72B before human labeling, another reason to avoid treating the endpoint as an exact error-token boundary.

Compare the endpoint, best single offset, mean/max pooling, ordered concatenation, reverse/shuffle controls, the previous-to-current endpoint difference, full-step mean, and a GeoReason-style velocity/acceleration baseline. Sweep \(T=1,\ldots,6\) first, then use raw activations at \(T\in\{12,24,48\}\) to test the granularity mismatch before paying to train longer-window dictionaries. The median annotated error step is 325 characters long, so \(T=6\) covers only a small tail of the event.

A separate strict pre-error task can end before the first token of step \(e\), but it should not be mixed into the primary result: ProcessBench labels the first **actually incorrect** step, not a latent precursor, and a negative anticipatory result would be unsurprising.

Use an ordered-gain gate of at least 0.02 average precision over the strongest static or invariant comparator, with a problem-clustered 95% confidence interval above zero and at least a 0.02 loss under order destruction. If only \(T=12,24,50\) or whole-step velocity wins, the result supports a longer-window or step-scale story, not the current TXC.

**Recommendation:** demote behind exact-model self-checking and early reasoning-instability. Retain as a bounded secondary raw gate because the labels are high quality, but do not spend rebuttal time training dictionaries unless \(T\leq6\) passes the strict ordered gate.

## 0I. Forced-answer belief-update events

[Boppana et al. (2026)](https://arxiv.org/abs/2603.05488) track a model's answer belief through reasoning by training attention probes over prefixes and by forcing an answer at intermediate boundaries. Their strongest attention probe aggregates the complete prefix: on MMLU it reaches 87.98% macro accuracy while a single-token linear probe reaches 31.85%. They also find that verbalized backtracks, realizations, and reconsiderations do not align reliably with local confidence shifts across models and datasets. This is negative evidence for treating words such as “Wait” as the event label, but positive evidence that a separately measured belief trajectory contains real change points.

For our exact subject model, generate multiple-choice MMLU-Redux traces and force the model to answer at successive sentence or paragraph boundaries. Let \(q_k\in\Delta^3\) be the four-choice distribution at boundary \(k\). Define a behavioral update event using either

\[
y_k
=\mathbf 1\!\left[
\arg\max q_k\neq \arg\max q_{k-1}
\right]
\quad\text{or}\quad
y_k
=\mathbf 1\!\left[
\operatorname{JSD}(q_k,q_{k-1})>\delta
\right].
\]

Probe the original, unmodified trace with a local window ending at boundary \(k\). Match negative boundaries on the current answer, current confidence, step index, trace length, and question; otherwise an endpoint probe can win simply by decoding which answer is currently favored. Group all traces of a question in one fold.

The essential baselines are \(h_k\), \(h_{k-1}\), \(h_k-h_{k-1}\), the best individual offset, invariant pooling, the full-prefix attention probe, and the scalar forced-answer quantities \(q_k,q_{k-1}\). Reverse and shuffle the local activation window. A TXC win over the endpoint but not over \(h_k-h_{k-1}\) would show that change matters without showing that the current dictionary is the right representation; a win only for the full-prefix attention probe would indicate that the required scale is much longer than \(T=6\).

This construction is more expensive than final-correctness prediction because each boundary requires an additional model query, but its ground truth is mechanical and local rather than supplied by an LLM judge. A sentence-boundary pilot on 300–500 questions should determine event prevalence before collecting token-dense trajectories.

**Recommendation:** retain as the cleanest localized follow-up to the bag-level correctness task. Do not use the attention probe's own predictions as labels, because that would train a representation to mimic another readout of the same activations; forced-answer distributions provide the behaviorally grounded target.

## 0J. Confidence-region change points

[Xu et al. (2026)](https://arxiv.org/abs/2606.02020) query an intermediate answer after each reasoning step, track its predictive entropy, and model the transition from exploratory high entropy to stable low entropy as a two-regime change-point problem. Their CUSUM detector provides an explicit sequential-statistics theory and works across math, science, and coding reasoning.

This can be reproduced on a few hundred of our exact-model MATH traces without new annotation: append an answer-inducing suffix after each sentence prefix, estimate the answer entropy sequence, identify a confidence-entry change point, then ask whether ordered residual windows at the original sentence boundary predict that event. It is more expensive than using the released function tags because every boundary requires an additional answer query.

The main threat is target leakage: the change point is defined from a scalar model output that may already be recoverable from the current endpoint state. The necessary baselines are the entropy itself, the endpoint residual, entropy slope, CUSUM over the scalar sequence, invariant pooling, and order destruction. This is theoretically clean only if an ordered hidden window improves on those sufficient-looking statistics.

**Recommendation:** retain as a second-stage theory experiment, not the first new benchmark.

## 1. RAG hallucination onset

### Why it fits

[RAGTruth](https://aclanthology.org/2024.acl-long.585/) contains 17,790 naturally generated RAG responses with manual response- and word-level hallucination annotations across question answering, data-to-text generation, and summarization. Its character-offset spans can be mapped to model tokens, making the first unsupported token a clean event boundary. A single response can move from grounded text into an unsupported span and back again, so the label is genuinely local rather than a property assigned to an entire rollout.

Two later results make this more than a generic hallucination benchmark:

- [Lookback Lens](https://aclanthology.org/2024.emnlp-main.84/) labels an eight-token sliding chunk as hallucinated if it overlaps a hallucination span, then detects it using attention-derived features averaged across that chunk. This establishes that window-level hallucination detection and guided decoding are operationally feasible, although averaging is permutation-invariant and the detector observes the hallucinated chunk rather than a precursor.
- [RAGLens](https://arxiv.org/abs/2512.08892) identifies an SAE feature that sometimes activates immediately before unsupported numeric or temporal details. In selected prefix interventions, increasing that feature changed invented numbers or dates into nonspecific, context-faithful continuations. This is useful precursor and causal evidence, but it is a small case study: other reported features activate during or after the hallucination, and the paper does not measure lead time systematically. Its main detector max-pools features over the complete response, so its strong results do not establish temporal-order sensitivity.

### Proposed TXC task

Let \(t^*\) be the first token in an annotated hallucination span. Build separate examples from windows strictly before \(t^*\), windows crossing \(t^*\), and windows after \(t^*\). The headline test should be prediction from strictly pre-onset windows; a result that appears only once the unsupported words are present is useful monitoring, but it is not analogous to the backtracking precursor result.

Compare TXC, T-SAE, and conventional SAE features using the same sparse probe and grouped folds. The essential baselines are the best single offset and exact order-invariant max, mean, and top-\(k\) pooling. Shuffle, reverse, and onset-jitter controls distinguish ordered transition information from extra context or loose event alignment. All responses derived from one RAGTruth source must remain in the same fold, and unsupported numbers should be matched against grounded numbers so the probe cannot learn prefixes such as “at the age of.” Report conflict and baseless-information spans separately, along with the three source tasks.

A causal extension would intervene on selected features at several offsets before \(t^*\) and measure whether fresh generations remain grounded. That should follow a positive lead-time result rather than be part of the first pilot.

### Feasibility and main risk

A retrospective pilot is moderate-cost: teacher-force a few thousand released responses through the model used by our dictionaries, cache event-aligned activations, and run the probes on one A40. Label preparation is cheap; activation extraction is the main compute cost. Because many released responses were generated by other models, this version tests whether our model represents the transition but cannot support a strong claim about its own generation mechanism. The clean causal version requires fresh generations from the dictionary's subject model and new span annotations.

A counts-only audit restricted to the 2,911 good
`mistral-7B-instruct` responses found 3,486 model-token-aligned onsets. After
matching within the official split on task, position decile, exact preceding
model token, and punctuation, and removing repeated six-token windows, 2,182
positive onsets and 6,341 grounded controls remain. The paper records only the
generic model name; using `mistralai/Mistral-7B-Instruct-v0.1` is a
date-based inference rather than a confirmed checkpoint identity.

On the untouched official test split, the exact \(T=6\) token-identity gate
obtains AP .331 for ordered 1--3 grams, .327 for the train-selected best
single offset, and .320 for an invariant unigram bag. Source-clustered 95%
intervals for both ordered advantages cross zero. A sensitivity analysis
that matches once using a common \(T=24\) cohort gives a more favorable
\(T=6\) result: AP .380 versus .322 for the best offset
(\(\Delta=.059\), 95% CI [.014, .100]) and .352 for the bag
(\(\Delta=.028\), 95% CI [-.003, .054]). At \(T=12\), ordered AP rises to
.406, but the invariant bag reaches .391 and again closes the order-specific
gap. Data-to-text consistently supplies nearly all of the multi-token
advantage.

This establishes distributed local lexical context, especially before
structured-data hallucinations, but not yet a robust order-specific
opportunity. The common-cohort result also conditions on having a 24-token
history, so it is a sensitivity analysis rather than the primary six-token
estimate.

**Recommendation:** do not train a Mistral TXC for this dataset. Retain a
bounded raw data-to-text gate only if spare compute remains; the stronger
follow-up is fresh exact-subject-model data-to-text generation with
mechanically checkable unsupported numbers or entities.

## 2. Garden-path reanalysis

### Why it fits

[Hanna and Mueller (NAACL 2025)](https://aclanthology.org/2025.naacl-long.164/) study how autoregressive LMs handle temporarily ambiguous garden-path sentences. Their released set contains 72 items across NP/Z, NP/S, and MV/RR constructions, with ambiguous and lexically controlled unambiguous variants. The event boundary \(t^*\) is the first word that rules out the initially preferred parse. For example, the same critical word can follow a garden-path prefix (“After the politician signed the bill **received** ...”) or a comma-disambiguated control (“After the politician signed, the bill **received** ...”).

Using Pythia-70M and a Gemma-2-2B replication, the paper finds SAE features supporting both parses before resolution and shows that clamping subject-, object-, and clause-boundary features causally changes continuation preferences. This is strong evidence for a latent state distributed over an ambiguous prefix. The paper does not, however, trace a feature switching on densely across \(t^*\), nor does it show that an ordered window beats token-local or invariant representations.

The companion [Garden-Path Traversal dataset](https://github.com/wjurayj/garden-path-gpt2) makes the temporal scale explicit. It contains 43 NP/Z, 20 NP/S, and 20 MV/RR templates, each factored into an ambiguity-inducing cue, an optional extension that lengthens the ambiguous region, a disambiguator, and a matched negated or unambiguous form. The associated hidden-state paper reports that latent-state distances can reveal an ambiguity before the disambiguating word even when next-token surprisal does not. This is unusually close to the desired phenomenon: a state is induced, persists for a controlled lag, then must be revised.

[Zhou, Stanojević, and Hale (2026)](https://arxiv.org/abs/2606.27206) supply the missing formal event variable. Their syntactic belief update at word \(w_i\) is the Rényi divergence between an incremental parser's post-word and pre-word distributions over dependency trees,

\[
q_i(Y)=p(Y\mid w_{1:i-1}),\qquad
p_i(Y)=p(Y\mid w_{1:i}),
\]

\[
\operatorname{SBU}_{i,\alpha}
=D_\alpha(p_i\Vert q_i)
=\frac{1}{\alpha-1}
\log\sum_Y p_i(Y)
\left(\frac{p_i(Y)}{q_i(Y)}\right)^{\alpha-1}.
\]

This is a directed, word-level measure of how much the latent parse belief changes, rather than a construction label attached to a whole sentence. On controlled human data, SBU uniquely recovers the aggregate difficulty ordering NP/S \(<\) NP/Z \(<\) MV/RR, and larger \(\alpha\) better matches the human magnitude hierarchy by emphasizing changes among high-probability parses. It does not explain item-level variation within a construction as well as all lexical/supertag controls, so the target should be treated as a syntactic transition variable rather than a complete model of reading time.

The authors' [incremental-parser release](https://github.com/atzhou8/discriminative_incremental_parsing) makes this scalable beyond the 72 paired stimuli. It accepts arbitrary one-sentence-per-line text and emits word-level backward KL and Rényi-divergence columns; the public parser checkpoint is about 4.3 GB. The stock inference is not strictly online, however: at word \(i\), it supplies the parser with the known number of future positions as mask tokens, reflecting the dashes shown to participants in self-paced reading. A causal LM state does not know the eventual sentence length. The primary natural-text label must therefore use a fixed future horizon \(H\) at every word, comparing prefix \(w_{1:i-1}\) followed by \(H\) masks against prefix \(w_{1:i}\) followed by \(H-1\) masks. We can then label high-SBU state changes in a large natural-text corpus, match them to low-SBU words on lexical identity, frequency, POS, lexical surprisal, sentence position, and prefix length, and reserve SAP's controlled garden paths and human reading-time effects for external validation.

### Proposed TXC task

The powered primary task should use the fixed-horizon continuous SBU label rather than train on the small garden-path set. For each candidate word \(i\), teacher-force the prefix through the subject model and cache windows ending before, at, and after \(w_i\). Compare prediction of continuous \(\operatorname{SBU}_{i,\alpha}\), plus a preregistered high-versus-low matched classification, using the endpoint, every offset, mean/std and learned invariant pooling, the explicit first difference \(h_i-h_{i-1}\), the explicit second difference, ordered flattening, reverse/shuffle, and boundary jitter. Split by source sentence or document before event sampling so nearby words never cross folds.

The controlled validation defines the subject-model garden-path effect for item \(i\) as

\[
g_i =
  -\log p(w_i^*\mid x_i^{\mathrm{amb}})
  +\log p(w_i^*\mid x_i^{\mathrm{ctrl}}),
\]

where \(w_i^*\) is the same disambiguating word in an ambiguous prefix and its matched negated or unambiguous control. This behavioral target prevents the benchmark from assuming that every human garden path produces the same model state. At the final token before and at \(w_i^*\), cache the last \(T\in\{1,\ldots,6\}\) residual states and ask whether a raw ordered view predicts \(g_i\), or a preregistered high-versus-low \(g_i\) split, better than the endpoint and invariant views. A feature learned on the natural-text SBU task should then be evaluated without refitting on the paired garden paths.

The validation analysis is paired and grouped by underlying item, then repeated across zero, short, and long optional extensions. The predicted interaction is more important than an unconditional classifier score: if TXC captures a local reanalysis trajectory, order sensitivity should peak in windows crossing \(t^*\), while a stable parse commitment should be endpoint- or low-frequency-dominated before the transition. A gain that survives only when the ambiguity cue itself lies inside \(T\) is a local lexical construction result, not evidence about persistent temporal state.

Only after the raw gate passes should we encode the same windows with the existing Gemma-2-2B-IT layer-13 \(T=5\) dictionaries. The strongest learned result would be a sparse TXC readout that beats an identically exposed SAE and loses performance under order destruction, followed by a causal intervention on the selected parse-transition feature. This would extend Hanna and Mueller's per-token SAE result rather than duplicate it: their features represent competing parses, while the TXC feature would represent the ordered transition between commitment and reanalysis.

### Connection to the power-law and spectral claims

Let \(z_t\) denote a scalar parse-commitment coordinate. During the ambiguous region, \(z_t\) should be slowly varying and therefore concentrate at low temporal frequencies. At disambiguation, a successful reanalysis produces a signed local change \(\Delta z_t=z_t-z_{t-1}\), whose transfer function \(|1-e^{-i\omega}|\) suppresses the slow background and emphasizes the transition. SBU gives an external measurement of the magnitude of this update, while varying the extension length changes the lag over which commitment must persist without changing the local disambiguating event.

This yields a falsifiable bridge rather than an inference from the empirical activation autocorrelation alone. A power-law background may explain multiscale persistence, but it does not imply a preferred \(T\). The task-specific prediction is that a short ordered filter helps at the transition because it approximates a derivative or directed motif; the endpoint should win during stable persistence, and distant covariance beyond the window should not matter once the last \(T\) covariance is held fixed. The strongest falsifier is the explicit first-difference baseline: if \(h_i-h_{i-1}\) matches the ordered view, the task supports temporal differencing but does not motivate a six-token TXC.

### Feasibility and main risk

The [official repository](https://github.com/hannamw/GP-mechanisms) includes stimuli and intervention code, while [SAP Benchmark](https://github.com/caplabnyu/sapbenchmark) supplies exact critical-word positions and matched ambiguous/unambiguous sentences. The SBU repository supplies the dense labeler, but downloading and running its 4.3 GB RoBERTa-large parser is a larger setup cost than the paired forward passes. Its implementation runs two full masked-sentence parser passes per word, so large-corpus cost grows roughly with sentence length times the parser's full-sentence attention cost. The 72 core items and 83 factored templates make validation trivial on an A40 and need no generation or judge. The sample is small, punctuation or construction-specific words can become shortcuts, the stock SBU label leaks future length, and the Gemma replication used the base rather than instruction-tuned checkpoint; fixed-horizon labels, paired behavioral targets, held-out items, same-critical-word comparisons, and explicit token-identity controls are mandatory.

**Recommendation:** first verify the fixed-horizon SBU modification on the paired stimuli, then start a dense natural-text cohort while using the paired garden paths only as external validation. This is the cleanest theory-to-task bridge found so far, but it advances to TXC training only if ordered raw states beat both the endpoint and explicit first difference.

## 3. Error recognition before self-correction

### Why it fits

[Kumaran et al. (2026)](https://arxiv.org/abs/2604.22271) use a verify-then-correct protocol on TriviaQA and MNLI. After an initial answer, they read activations at the first post-answer newline (PANL), before the verification instruction. That state predicts whether the model will declare its answer wrong, revise it, and successfully correct it. Restoring clean PANL activations after corrupting answer information causally rescues error-detection behavior, and the findings replicate in Gemma-3-27B and Qwen-2.5-7B.

This is almost the same conceptual sequence as backtracking: commitment, latent evaluation, then overt correction. The problem for TXC is that the published readout is one token that can already attend over the entire answer, and the factual answers are usually very short. The paper therefore establishes a strong single-state baseline rather than a reason that an external ordered window should win.

### Proposed TXC adaptation

Move the protocol to long reasoning traces and insert or identify intermediate commitment boundaries. At each boundary, predict later error recognition, answer revision, and successful correction from the preceding ordered window. Compare against PANL alone, the last answer token, the best individual offset, and exact invariant pooling. This could also motivate a finer analysis of our existing backtracking traces, but that would strengthen the current task rather than add an independent benchmark.

No author-linked code or processed activation dataset was found, so reproducing the published protocol and then extending it to long traces is materially more work than the first two candidates.

**Recommendation:** cite it as direct mechanistic support for the backtracking story; implement only if the reviews specifically demand a second self-correction task or if there is time beyond the RAG pilot.

## 4. Emergent response planning: relevant control, weak candidate

[Dong et al. (2025)](https://arxiv.org/abs/2502.06258) probe the final prompt-token representation to predict global properties of a future response: length, reasoning-step count, later character choice, final multiple-choice answer, correctness, and factual consistency. They also report a high-low-high trajectory when probing single hidden states at coarse positions during generation. The [official code](https://github.com/niconi19/Emergent-Response-Planning-in-LLMs) reconstructs generations, labels, activations, and probes from public datasets.

This paper is relevant evidence that future behavior can be represented before it is verbalized, but its labels describe whole-response outcomes and it has no localized commitment-to-reversal boundary. Every reported probe reads one position, and causal intervention is explicitly left to future work. A successful TXC result here could therefore reflect extra capacity or prompt information rather than a temporal transition.

The clean use is as a negative control: compare TXC with a final-token probe and invariant pooling on one planning label. Little or no TXC gain would support the paper's proposed distinction between global, persistent attributes and localized, ordered transition signals.

**Recommendation:** do not present this as the second backtracking-like real-world task.

## Closed local screens

Two additional public-corpus ideas were screened locally and should not be
retried for the reviewer deadline:

- **MaiChat message revision and pause onset:** only 304 complete-boundary
  multiword deletion events survive at \(T=5\). Ordered log loss is 1.1316
  versus 1.1328 for the endpoint, too small to justify activation extraction.
  A separately matched 1,738-event long-pause task is at chance: ordered AUROC
  .504 versus .525 for the endpoint and .528 after reversal.
- **MultiWOZ dialogue-state overwrite:** at \(T=6\), overwrite-versus-initial
  AUROC is .858 ordered versus .865 for the invariant bag, while simple turn
  metadata reaches .999. For the harder overwrite-versus-retention contrast,
  ordered AUROC is .797 versus .816 for the bag and .742 for metadata. The
  label contains history, but order adds nothing beyond extra lexical content
  and the construction has severe positional leakage.

These nulls sharpen the selection rule. A task involving editing, delay, or
state change is insufficient; the ordered window must beat an invariant view
before any activation or dictionary experiment is authorized.

## Minimum credible protocol across candidates

Whichever task is chosen, the result should answer whether temporal order helps rather than whether a larger input helps:

1. Use event-aligned windows and report a lead-time curve from pre-onset through post-onset positions.
2. Compare against the best single offset and exact order-invariant pooling under the same probe budget.
3. Shuffle and reverse positions, and jitter the annotated boundary.
4. Group splits by the underlying prompt/source/item, not by windows.
5. Match obvious lexical, length, and token-type confounders.
6. Treat a TXC win as convincing only if it is strongest before or across the transition and weakens when order or alignment is destroyed.

## References

- Futrell, Richard, Edward Gibson, Harry J. Tily, Idan Blank, Anastasia Vishnevetsky, Steven T. Piantadosi, and Evelina Fedorenko. 2021. [“The Natural Stories Corpus: A Reading-Time Corpus of English Texts Containing Rare Syntactic Constructions.”](https://pmc.ncbi.nlm.nih.gov/articles/PMC8549930/) _Language Resources and Evaluation_ 55:63–77. [Data](https://github.com/languageMIT/naturalstories).
- Shain, Cory, and William Schuler. 2018. [“Deconvolutional Time Series Regression: A Technique for Modeling Temporally Diffuse Effects.”](https://aclanthology.org/D18-1288/) EMNLP 2018.
- Tsipidi, Eleftheria, Samuel Kiegeland, Francesco Ignazio Re, Tianyang Xu, Mario Giulianelli, Karolina Stanczak, and Ryan Cotterell. 2026. [“Probing for Reading Times.”](https://aclanthology.org/2026.acl-long.575/) ACL 2026. [Code](https://github.com/rycolab/llm-representations-rt).
- Barenholtz, Elan. 2026. [“Trajectory Dynamics in Language Model Hidden States Predict Human Processing Costs Beyond Surprisal.”](https://arxiv.org/abs/2606.05346) arXiv:2606.05346.
- Chrabąszcz, Maciej, Aleksander Szymczyk, Marcin Sendera, Tomasz Trzciński, and Sebastian Cygert. 2026. [“Monitoring the Internal Monologue: Probe Trajectories Reveal Reasoning Dynamics.”](https://arxiv.org/abs/2605.18549) arXiv:2605.18549.
- Sun, Lihao, Hang Dong, Bo Qiao, Qingwei Lin, Dongmei Zhang, and Saravan Rajmohan. 2026. [“LLM Reasoning as Trajectories: Step-Specific Representation Geometry and Correctness Signals.”](https://arxiv.org/abs/2604.05655) ACL 2026. [Code](https://github.com/slhleosun/reasoning-trajectory).
- Pipis, Charilaos, Shivam Garg, Vasilis Kontonis, Vaishnavi Shrivastava, Akshay Krishnamurthy, and Dimitris Papailiopoulos. 2026. [“Wait, Wait, Wait... Why Do Reasoning Models Loop?”](https://arxiv.org/abs/2512.12895) ICML 2026.
- Duan et al. 2026. [“Circular Reasoning: Understanding Self-Reinforcing Loops in Large Reasoning Models.”](https://arxiv.org/abs/2601.05693) arXiv:2601.05693.
- Xie et al. 2025. [“Word Salad Chopper: Reasoning Models Waste A Ton Of Decoding Budget On Useless Repetitions, Self-Knowingly.”](https://arxiv.org/abs/2511.00536) EMNLP 2025. [Code and data](https://github.com/wenyaxie023/WordSaladChopper).
- Hough, Julian, and David Schlangen. 2017. [“Joint, Incremental Disfluency Detection and Utterance Segmentation from Speech.”](https://aclanthology.org/E17-1031/) EACL 2017. [Code and annotations](https://github.com/dsg-bielefeld/deep_disfluency).
- Tian, Yu, Scott A. Crossley, and Luuk Van Waes. 2025. [“The KLiCKe Corpus: Keystroke Logging in Compositions for Knowledge Evaluation.”](https://doi.org/10.17239/jowr-2025.17.01.02) _Journal of Writing Research_ 17(1):23–60. [Data repository](https://github.com/terryyutian/KLiCKe-Corpus).
- Yang, Shu, Junchao Wu, Xin Chen, Yunze Xiao, Xinyi Yang, Derek F. Wong, and Di Wang. 2025. [“Understanding Aha Moments: from External Observations to Internal Mechanisms.”](https://arxiv.org/abs/2504.02956) arXiv:2504.02956.
- Boppana, Siddharth, Annabel Ma, Max Loeffler, Raphael Sarfati, Eric Bigelow, Atticus Geiger, Owen Lewis, and Jack Merullo. 2026. [“Reasoning Theater: Disentangling Model Beliefs from Chain-of-Thought.”](https://arxiv.org/abs/2603.05488) arXiv:2603.05488. [Code](https://github.com/AskSid/disentangling-computation-from-cot).
- Zheng, Chujie, Zhenru Zhang, Beichen Zhang, Runji Lin, Keming Lu, Bowen Yu, Dayiheng Liu, Jingren Zhou, and Junyang Lin. 2025. [“ProcessBench: Identifying Process Errors in Mathematical Reasoning.”](https://arxiv.org/abs/2412.06559) ACL 2025. [Data](https://huggingface.co/datasets/Qwen/ProcessBench).
- Alvarez, Argenis, and Ali Baheri. 2026. [“Where Does Reasoning Break? A Geometric Probe of Hidden-State Dynamics in Large Language Models.”](https://arxiv.org/abs/2605.13772) arXiv:2605.13772.
- Xiong, Guangzhi, Zhenghao He, Bohan Liu, Sanchit Sinha, and Aidong Zhang. 2026. [“Toward Faithful Retrieval-Augmented Generation with Sparse Autoencoders.”](https://arxiv.org/abs/2512.08892) ICLR 2026. [Code](https://github.com/gzxiong/RAGLens).
- Niu, Cheng, Yuanhao Wu, Juno Zhu, Siliang Xu, KaShun Shum, Randy Zhong, Juntong Song, and Tong Zhang. 2024. [“RAGTruth: A Hallucination Corpus for Developing Trustworthy Retrieval-Augmented Language Models.”](https://aclanthology.org/2024.acl-long.585/) ACL 2024. [Data and code](https://github.com/ParticleMedia/RAGTruth).
- Chuang, Yung-Sung, Linlu Qiu, Cheng-Yu Hsieh, Ranjay Krishna, Yoon Kim, and James R. Glass. 2024. [“Lookback Lens: Detecting and Mitigating Contextual Hallucinations in Large Language Models Using Only Attention Maps.”](https://aclanthology.org/2024.emnlp-main.84/) EMNLP 2024. [Code](https://github.com/voidism/Lookback-Lens).
- Hanna, Michael, and Aaron Mueller. 2025. [“Incremental Sentence Processing Mechanisms in Autoregressive Transformer Language Models.”](https://aclanthology.org/2025.naacl-long.164/) NAACL 2025. [Code and data](https://github.com/hannamw/GP-mechanisms).
- Kumaran, Dharshan, Viorica Patraucean, Simon Osindero, Petar Veličković, and Nathaniel Daw. 2026. [“How LLMs Detect and Correct Their Own Errors: The Role of Internal Confidence Signals.”](https://arxiv.org/abs/2604.22271) arXiv:2604.22271.
- Dong, Zhichen, Zhanhui Zhou, Zhixuan Liu, Chao Yang, and Chaochao Lu. 2025. [“Emergent Response Planning in LLMs.”](https://arxiv.org/abs/2502.06258) arXiv:2502.06258. [Code](https://github.com/niconi19/Emergent-Response-Planning-in-LLMs).
