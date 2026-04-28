# Temporal Crosscoders and Contrastive Losses

## Summary

For temporal crosscoders that stack activations from different sequence positions at the same layer, a contrastive loss similar to the one used in the temporal SAE paper is worth trying, but it should not be applied naively to all latents.

The most natural generalization is:

\[
\boxed{\text{contrast high-level state features across masked or source-split temporal views}}
\]

not:

\[
\boxed{\text{contrast all latents across adjacent overlapping windows.}}
\]

The core intuition is:

\[
\text{semantic/context state changes slowly, while syntax/token details change quickly.}
\]

The paper's T-SAE setup exploits this by partitioning SAE latents into high-level and low-level chunks, then applying a temporal contrastive loss only to the high-level chunk. The low-level chunk is left free to reconstruct residual, token-specific information.

For temporal crosscoders, the same principle applies, but the feature ontology should be richer. A good initial decomposition is:

\[
z_t = [h_t, r_t, \ell_t]
\]

where:

- \(h_t\): smooth state/context/semantic features;
- \(r_t\): fast temporal-relational features, such as copy, induction, delimiter transitions, answer-option phase, or binding events;
- \(\ell_t\): position-local/token-local features, such as POS, token identity, punctuation, and formatting.

The contrastive loss should apply to \(h_t\), not to \(r_t\) or \(\ell_t\).

---

## 1. Basic Temporal Crosscoder Setup

A temporal crosscoder takes a stack of same-layer activations from several sequence positions. For example:

\[
X_t = [x_{t-3}, x_{t-2}, x_{t-1}, x_t]
\]

More generally:

\[
X_t = [x_{t+\tau}]_{\tau \in \Omega}
\]

where \(\Omega\) is a set of temporal offsets.

The encoder produces sparse latents:

\[
z_t = E(X_t)
\]

and the decoder reconstructs the stacked activation window:

\[
\hat{X}_t = D z_t + b.
\]

For a decomposed temporal crosscoder, use:

\[
z_t = [h_t, r_t, \ell_t]
\]

and:

\[
\hat{X}_t = D_H h_t + D_R r_t + D_L \ell_t + b.
\]

The reconstruction loss is then:

\[
\mathcal{L}_{\text{recon}}
= \|X_t - \hat{X}_t\|_2^2.
\]

The full objective can be:

\[
\mathcal{L}
=
\mathcal{L}_{\text{recon}}
+
\lambda_{\text{sparse}}\mathcal{L}_{\text{sparse}}
+
\lambda_{\text{dead}}\mathcal{L}_{\text{dead}}
+
\lambda_c \mathcal{L}_{\text{state-contrast}}.
\]

---

## 2. Why Naive Temporal Contrast Is Risky

A naive adaptation would add a contrastive loss between adjacent full crosscoder codes:

\[
\operatorname{InfoNCE}(z_t, z_{t+1})
\]

or a direct smoothness penalty:

\[
\|z_t - z_{t+1}\|^2.
\]

This is usually the wrong thing to do.

### Problem 1: adjacent windows overlap heavily

For a 4-position temporal window:

\[
X_t = [x_{t-3},x_{t-2},x_{t-1},x_t]
\]

and:

\[
X_{t+1} = [x_{t-2},x_{t-1},x_t,x_{t+1}],
\]

three of the four positions are shared.

A contrastive loss between \(z_t\) and \(z_{t+1}\) can therefore be satisfied by encoding the overlapping local phrase rather than a true semantic state.

The model may learn something like:

\[
\text{``contains this overlapping trigram''}
\]

instead of:

\[
\text{``this passage is about photosynthesis''}.
\]

### Problem 2: many temporal-crosscoder features should be sharp

Temporal crosscoders are useful precisely because they can discover features that are not smooth over positions. Examples include:

- induction/copy features;
- delimiter-opening or delimiter-closing features;
- beginning-of-answer features;
- name-binding features;
- quote/parenthesis/code-block phase transitions;
- features that fire only when a source token and destination token stand in a particular relation.

A feature such as:

\[
r_t = \text{``copy answer associated with repeated key''}
\]

may correctly fire only at one position:

\[
r_{t-1}=0, \qquad r_t=1, \qquad r_{t+1}=0.
\]

A smoothness or contrastive objective on the full latent vector would punish this sharpness. That would make the crosscoder worse.

So the contrastive loss should be restricted to a designated smooth-state bank \(h_t\).

---

## 3. Recommended Architecture

Use three latent banks:

\[
z_t = [h_t, r_t, \ell_t].
\]

### 3.1 High-level state bank: \(h_t\)

This bank should encode slowly varying features such as:

- topic;
- entity or speaker context;
- document type;
- task setting;
- discourse mode;
- broad semantic frame;
- local world-state variables that persist across several tokens.

This is the only bank that receives the temporal/state contrastive loss.

### 3.2 Temporal-relational bank: \(r_t\)

This bank should encode sharp relational features across positions, such as:

- copy/induction events;
- subject-object binding;
- source-destination relations;
- matching quotation marks or parentheses;
- local algorithmic transitions;
- answer-option or list-item phase.

This bank should be sparse and reconstruction-useful, but should not be contrastively smoothed.

### 3.3 Local bank: \(\ell_t\)

This bank should encode local token-level or position-level details, such as:

- token identity;
- POS;
- punctuation;
- whitespace or formatting;
- local syntactic category;
- nearby phrase shape.

This bank should also not receive the contrastive loss.

---

## 4. Activation Function Choices

The best activation function depends on the sparsity mechanism, but from first principles the key desiderata are:

1. **Nonnegative latents** are usually preferable for interpretability.
2. **Hard or approximate top-k sparsity** is attractive because it gives an explicit feature budget.
3. **Magnitude information** should remain meaningful, because temporal features may vary in strength.
4. **Dead-feature recovery** matters, especially when introducing auxiliary objectives.

Good initial choices:

### Option A: BatchTopK or TopK-style sparse activation

This is a strong default. It makes the feature budget explicit and avoids tuning an L1 coefficient as the primary sparsity control.

For an encoder preactivation \(a_t\):

\[
z_t = \operatorname{TopK}(\operatorname{ReLU}(a_t)).
\]

For temporal crosscoders, this can be done globally across banks or separately per bank.

I would prefer separate budgets at first:

\[
K = K_H + K_R + K_L.
\]

For example:

\[
K_H = 4, \qquad K_R = 8, \qquad K_L = 8.
\]

This prevents the high-level bank from being starved by local reconstruction features.

### Option B: JumpReLU or gated sparse activation

A JumpReLU-like activation can work if you want learned thresholds:

\[
z_i = a_i \cdot \mathbf{1}[a_i > \theta_i].
\]

This may be useful if feature densities differ dramatically across \(h\), \(r\), and \(\ell\). The risk is additional training complexity.

### Option C: ReLU + L1

This is simpler but less controlled:

\[
z_t = \operatorname{ReLU}(a_t)
\]

with:

\[
\lambda_1 \|z_t\|_1.
\]

It is reasonable for prototypes, but I would not make it the primary final architecture unless it empirically outperforms TopK-style sparsity.

---

## 5. Regularization Choices

### 5.1 Sparse regularization

Use either an explicit top-k budget or an L1 penalty. I would start with top-k because it gives cleaner control over how much information each bank may carry.

Possible per-bank sparsity:

\[
\|h_t\|_0 \le K_H,
\qquad
\|r_t\|_0 \le K_R,
\qquad
\|\ell_t\|_0 \le K_L.
\]

This avoids a degenerate solution where the state bank becomes dense and carries everything.

### 5.2 Decoder norm constraints

Normalize decoder columns to prevent arbitrary scale exchange between activations and dictionary vectors:

\[
\|D_j\|_2 = 1.
\]

For temporal crosscoders, each decoder feature has a temporal footprint across offsets:

\[
D_j = [d_{j,\tau}]_{\tau \in \Omega}.
\]

Normalize the whole feature footprint:

\[
\sum_{\tau \in \Omega} \|d_{j,\tau}\|_2^2 = 1.
\]

### 5.3 Bank-specific footprint regularization

A useful crosscoder-specific regularizer is to shape temporal footprints differently by bank.

For high-level features, encourage broad or smooth temporal footprints:

\[
\mathcal{L}_{\text{smooth-footprint}}
= \sum_{j \in H}\sum_{\tau}
\|d_{j,\tau+1} - d_{j,\tau}\|_2^2.
\]

For local features, optionally encourage narrow footprints:

\[
\mathcal{L}_{\text{local-footprint}}
= \sum_{j \in L}
\sum_{\tau \ne \tau^*(j)} \|d_{j,\tau}\|_2^2.
\]

For relational features, allow multi-position structure and do not impose excessive smoothness.

### 5.4 Anti-collapse / entropy monitoring

The high-level contrastive bank can collapse into dense document-ID or sequence-ID features. Monitor:

\[
\text{feature firing rate},
\]

\[
\text{activation entropy},
\]

\[
\text{number of live features},
\]

and whether max-activating examples are semantically coherent rather than merely from the same document template.

---

## 6. Auxiliary Losses

### 6.1 Dead-feature auxiliary loss

A dead-feature auxiliary loss is still useful. TopK-like sparse models often need a mechanism to revive unused features.

A typical version asks dead or rarely used features to reconstruct residual error:

\[
r^{\text{resid}}_t = X_t - \hat{X}_t
\]

and trains a subset of inactive/dead features to explain that residual.

This is compatible with the contrastive loss, but it should be applied carefully: if a high-level feature is revived only through reconstruction residuals, it may become local rather than semantic. One possible choice is to apply dead-feature recovery more aggressively to \(r\) and \(\ell\), and more cautiously to \(h\).

### 6.2 Optional low-pass target for the high-level bank

A subtle issue: if \(h_t\) participates in reconstructing the entire stacked window \(X_t\), it may be forced to encode local details.

To avoid this, give \(h_t\) a lower-frequency target, such as a pooled activation:

\[
\bar{x}_t = \sum_{\tau \in \Omega} a_\tau x_{t+\tau}.
\]

Then include:

\[
\mathcal{L}_{H\text{-recon}}
= \|\bar{x}_t - D_H h_t\|_2^2.
\]

The full reconstruction still uses all banks:

\[
\hat{X}_t = D_H h_t + D_R r_t + D_L \ell_t + b.
\]

But the high-level bank gets an additional pressure to model pooled/state-like information rather than exact token-local detail.

### 6.3 Optional bank decorrelation

To encourage the banks to specialize, add a weak decorrelation penalty between bank reconstructions:

\[
\mathcal{L}_{\text{decor}}
=
\left\langle D_Hh_t, D_L\ell_t \right\rangle^2
+
\left\langle D_Hh_t, D_Rr_t \right\rangle^2.
\]

Use this cautiously. Too much decorrelation may prevent legitimate decomposition of a feature into shared state plus local residual.

---

## 7. Generalized Contrastive Loss

The temporal SAE paper applies a temporal contrastive loss to high-level features. For temporal crosscoders, the right generalization is to define two views that should share the same slow state, encode both, and contrast only their high-level chunks.

Let:

\[
h_t^a = E_H(M_a X_t)
\]

and:

\[
h_{t'}^b = E_H(M_b X_{t'}),
\]

where \(M_a\) and \(M_b\) are masks or view functions over temporal offsets.

Then define a symmetric InfoNCE loss:

\[
\mathcal{L}_{\text{state-contrast}}
=
\frac{1}{2}
\operatorname{CE}
\left(
\frac{\operatorname{sim}(h_i^a,h_j^b)}{\tau_c},
\ i=j
\right)
+
\frac{1}{2}
\operatorname{CE}
\left(
\frac{\operatorname{sim}(h_i^b,h_j^a)}{\tau_c},
\ i=j
\right).
\]

Here:

- \(i\) indexes examples in the batch;
- the diagonal pair \((i,i)\) is positive;
- off-diagonal pairs \((i,j)\), \(i \ne j\), are negatives;
- \(\tau_c\) is the contrastive temperature;
- \(\operatorname{sim}\) is usually cosine similarity.

A useful similarity function is:

\[
\operatorname{sim}(h,h')
=
\frac{\phi(h) \cdot \phi(h')}
{\|\phi(h)\|_2\|\phi(h')\|_2}.
\]

Possible choices:

\[
\phi(h)=h
\]

or:

\[
\phi(h)=\log(1+h/\alpha).
\]

The log-compressed version may help because sparse feature magnitudes can otherwise dominate the contrastive logits.

---

## 8. Positive Pair Sampling Strategies

### 8.1 View-split contrast within the same stacked window

For:

\[
X_t = [x_{t-3},x_{t-2},x_{t-1},x_t],
\]

sample two views:

\[
X_t^a = [x_{t-3},x_{t-1}]
\]

and:

\[
X_t^b = [x_{t-2},x_t].
\]

Then contrast:

\[
h_t^a \sim h_t^b.
\]

This asks: what information is common to different parts of this local window?

Topic, speaker, task, document style, and semantic setting should survive. Token identity, local syntax, and one-off transition features should not.

This is probably the cleanest first crosscoder generalization.

### 8.2 Non-overlapping same-sequence positives

Pick \(t'\) from the same sequence, but avoid excessive overlap:

\[
|t' - t| \ge |\Omega|.
\]

For a 4-token window, contrast \(X_t\) with \(X_{t+4}\) or \(X_{t+8}\), not just \(X_{t+1}\).

A distance kernel could be:

\[
p(t' \mid t)
\propto
\mathbf{1}[\text{same sequence}]
\mathbf{1}[\text{not across boundary}]
\exp(-|t'-t|/\ell).
\]

The parameter \(\ell\) controls the semantic timescale.

### 8.3 Masked-overlap adjacent positives

If adjacent windows are used, mask the shared positions differently.

For:

\[
X_t = [x_{t-3},x_{t-2},x_{t-1},x_t]
\]

and:

\[
X_{t+1} = [x_{t-2},x_{t-1},x_t,x_{t+1}],
\]

use masks such as:

\[
M_aX_t = [x_{t-3}, x_t]
\]

and:

\[
M_bX_{t+1} = [x_{t-2}, x_{t+1}].
\]

This prevents the contrastive objective from being satisfied by merely matching the overlapping raw activations.

---

## 9. Worked Example: Topic Versus POS

Suppose a toy activation decomposes as:

\[
x_t = s_{\text{topic}} + p_{\text{POS}(t)} + e_{\text{token}(t)}.
\]

For the sentence:

> Photosynthesis is the process by which plants convert sunlight into energy.

The topic variable is stable:

\[
s_t = \text{biology},
\]

but POS and token identity vary across positions:

\[
\text{Photosynthesis: noun},
\qquad
\text{is: verb},
\qquad
\text{the: determiner}.
\]

Suppose the high-level bank has two possible features:

\[
h = [h_{\text{biology}}, h_{\text{math}}].
\]

For two adjacent biology windows:

\[
h_t = [3,0],
\qquad
h_{t+1} = [2.8,0].
\]

Their cosine similarity is almost 1.

For a negative math window:

\[
h_{\text{neg}} = [0,3].
\]

The cosine similarity is 0.

With contrastive temperature \(\tau_c = 0.1\):

\[
\mathcal{L}
=
-\log
\frac{e^{1/0.1}}
{e^{1/0.1}+e^{0/0.1}}
\approx
-\log
\frac{e^{10}}{e^{10}+1}
\approx
0.000045.
\]

So the contrastive loss is very low.

Now suppose the high-level bank instead encodes POS. Adjacent tokens might be noun then verb:

\[
h_t = [1,0]
\]

and:

\[
h_{t+1} = [0,1].
\]

Their cosine similarity is 0. A noun negative elsewhere might have cosine similarity 1. Then:

\[
\mathcal{L}
\approx
-\log
\frac{e^0}{e^0+e^{10}}
\approx
10.
\]

So the contrastive loss strongly discourages putting POS into the high-level bank. That is the desired effect.

---

## 10. Worked Example: Why Full-Latent Contrast Is Harmful

Suppose the full crosscoder code is:

\[
z_t = [\text{topic},\ \text{current-is-noun},\ \text{current-is-verb}].
\]

At token `cat`:

\[
z_t = [1,1,0].
\]

At the next token `sat`:

\[
z_{t+1} = [1,0,1].
\]

The topic stays fixed, but the syntactic feature changes.

The cosine similarity of the full latent vectors is:

\[
\frac{1}{\sqrt{2}\sqrt{2}} = 0.5.
\]

A contrastive objective on full \(z\) will try to increase this similarity. That means it may push noun and verb features to co-activate or merge.

That is bad.

But if the contrastive loss sees only:

\[
h_t = [\text{topic}],
\]

then:

\[
h_t = [1],
\qquad
h_{t+1} = [1],
\]

and the contrastive loss is satisfied without corrupting local syntax.

---

## 11. Worked Example: Adjacent-Window Contrast Can Be Too Easy

Let:

\[
X_t = [a,b,c,d]
\]

and:

\[
X_{t+1} = [b,c,d,e].
\]

If we contrast \(h(X_t)\) and \(h(X_{t+1})\), the model can learn a high-level feature like:

\[
h = \text{``contains subsequence } b,c,d \text{''}.
\]

That feature will be highly consistent across adjacent windows, but it is not necessarily semantic. It is exploiting overlap.

A better positive pair is either non-overlapping:

\[
X_t = [a,b,c,d]
\]

and:

\[
X_{t+4} = [e,f,g,h],
\]

or two masked views:

\[
X_t^a = [a,c],
\qquad
X_t^b = [b,d].
\]

Now exact token overlap is less available, so the high-level bank is pushed toward common latent state rather than shared surface form.

---

## 12. Worked Example: Crosscoder View-Split Contrast

Suppose a 4-position window in a medical passage is:

\[
X_t = [
x_{\text{patient}},
x_{\text{presented}},
x_{\text{with}},
x_{\text{fever}}
].
\]

Split into two views:

\[
X_t^a = [x_{\text{patient}},x_{\text{presented}}]
\]

and:

\[
X_t^b = [x_{\text{with}},x_{\text{fever}}].
\]

A high-level “medical case report” feature should be inferable from both halves:

\[
h_t^a \approx [1,0,0]
\]

and:

\[
h_t^b \approx [1,0,0].
\]

But a local feature such as “preposition `with`” is present only in the second view. It should not be forced into \(h\); it should go to \(r\) or \(\ell\).

This is exactly the desired inductive bias: learn what is common across positions, not what is merely present at one position.

---

## 13. Worked Example: Sharp Temporal Features Should Not Be Smoothed

Consider an induction/copy-like event:

> The capital of France is Paris. France's capital is ...

At the final token, the model may have a feature like:

\[
r_t = \text{``copy answer associated with repeated key''}.
\]

This is a temporal-relational feature. It may fire sharply at one position. Adjacent positions may not share it:

\[
r_{t-1}=0,
\qquad
r_t=1,
\qquad
r_{t+1}=0.
\]

A temporal contrastive loss would punish this sharpness. But the sharpness is correct.

Therefore this feature belongs in \(r_t\), not \(h_t\).

---

## 14. Practical Training Recipe

A good first experiment:

\[
\Omega = \{-3,-2,-1,0\}.
\]

Use three banks:

\[
z_t = [h_t,r_t,\ell_t].
\]

Use view masks:

\[
M_aX_t = [x_{t-3},x_{t-1}]
\]

and:

\[
M_bX_t = [x_{t-2},x_t].
\]

Then train with:

\[
\mathcal{L}
=
\mathcal{L}_{\text{crosscoder recon}}
+
\lambda_{\text{sparse}}\mathcal{L}_{\text{sparse}}
+
\lambda_{\text{dead}}\mathcal{L}_{\text{dead}}
+
\lambda_c
\operatorname{InfoNCE}
\big(
E_H(M_aX_t),
E_H(M_bX_t)
\big).
\]

This preserves the temporal SAE paper's key inductive bias — smooth state variables — while respecting that temporal crosscoders should also find sharp temporal events.

---

## 15. Suggested Hyperparameters

### Bank sizes

Start with:

\[
n_H \approx 10\%-20\% \text{ of total features}.
\]

Then allocate the rest to temporal-relational and local features:

\[
n_R + n_L \approx 80\%-90\%.
\]

A reasonable first split:

\[
n_H:n_R:n_L = 20:40:40.
\]

If \(h\) learns too many local details, shrink \(n_H\) or increase contrast/masking. If \(h\) misses obvious topic/context features, enlarge \(n_H\).

### Sparsity budgets

Use separate budgets per bank. For example:

\[
K_H:K_R:K_L = 2:6:8
\]

or:

\[
K_H:K_R:K_L = 4:8:8.
\]

The high-level bank should generally have a smaller budget than the local and relational banks.

### Contrastive weight

Try:

\[
\lambda_c \in \{0.01, 0.03, 0.1, 0.3, 1.0\}.
\]

Anneal it in after reconstruction stabilizes. If the contrastive loss is active too early, the high-level bank may learn dense “same sequence” features before the dictionary has learned useful atoms.

### Temperature

Try:

\[
\tau_c \in \{0.03, 0.07, 0.1, 0.2\}.
\]

Lower temperatures make the contrastive objective sharper and more classification-like. Higher temperatures make it smoother and less aggressive.

---

## 16. What to Monitor

The success condition is not simply “contrastive loss goes down.” Instead, monitor whether the banks specialize correctly.

### 16.1 Reconstruction quality

Track:

\[
\|X_t - \hat{X}_t\|_2^2
\]

and, if possible, downstream loss recovery or CE-loss recovery.

### 16.2 Bank-specific smoothness

For each bank, compute temporal activation similarity:

\[
S_B(\Delta) = \mathbb{E}_t[\cos(z_t^B,z_{t+\Delta}^B)].
\]

Desired pattern:

\[
S_H(\Delta) \text{ high and slowly decaying},
\]

\[
S_R(\Delta) \text{ structured but not necessarily smooth},
\]

\[
S_L(\Delta) \text{ relatively local/fast-changing}.
\]

### 16.3 Probe separation

Probe each bank separately for:

- topic/context;
- POS;
- token identity;
- document type;
- named entity state;
- bracket/quote/code-block phase;
- copy/induction events.

Desired result:

- \(h\) is strong on semantic/context probes;
- \(\ell\) is strong on POS/token/local probes;
- \(r\) is strong on temporal relation/event probes.

### 16.4 Max-activating examples

Inspect max-activating examples for each bank.

Good \(h\)-features should look like:

- “medical case report”;
- “Python code block”;
- “discussion of transformer circuits”;
- “QA answer context”;
- “legal/contract language.”

Good \(r\)-features should look like:

- “closing a parenthesis opened two tokens ago”;
- “copying a previous entity”;
- “moving from question to answer”;
- “inside quoted span and about to exit.”

Good \(\ell\)-features should look like:

- “comma”;
- “capitalized proper noun”;
- “verb after auxiliary”;
- “newline indentation.”

### 16.5 Footprint entropy

For each decoder feature, measure its temporal footprint across offsets:

\[
p_{j,\tau}
=
\frac{\|d_{j,\tau}\|_2^2}{\sum_{\tau'}\|d_{j,\tau'}\|_2^2}.
\]

Then compute entropy:

\[
H_j = -\sum_\tau p_{j,\tau}\log p_{j,\tau}.
\]

Expected pattern:

- high-level features: higher footprint entropy;
- local features: lower footprint entropy;
- relational features: possibly multi-peaked footprints.

---

## 17. Failure Modes

### 17.1 High-level bank learns document ID

Symptom: high-level features are smooth, but max-activating examples are just the same document, website template, or formatting pattern.

Fixes:

- use harder negatives from the same document type;
- contrast shorter-range positives;
- use view-split positives rather than same-document positives;
- increase masking of surface tokens.

### 17.2 High-level bank learns overlapping n-grams

Symptom: adjacent-window contrast works too well, but features are phrase memorization rather than state.

Fixes:

- avoid adjacent full-window positives;
- use non-overlapping positives;
- use offset dropout;
- use different masks for the two views.

### 17.3 Contrastive loss destroys sharp temporal features

Symptom: copy/induction/delimiter features become smeared over neighboring positions.

Fixes:

- ensure contrast is only applied to \(h\);
- reduce \(\lambda_c\);
- increase capacity of \(r\);
- avoid decoder sharing that forces \(r\) and \(h\) to use the same temporal footprint.

### 17.4 High-level bank becomes dense

Symptom: \(h\) is active on almost everything and carries generic sequence similarity.

Fixes:

- reduce \(K_H\);
- use stronger top-k or thresholding;
- normalize before contrast;
- use log-compression for contrastive features;
- add firing-rate penalties.

---

## 18. Bottom Line

A temporal crosscoder should probably include a contrastive auxiliary objective, but only if the objective is aimed at the right part of the representation.

The recommended design is:

\[
\boxed{
z_t = [h_t,r_t,\ell_t]
}
\]

with:

\[
\boxed{
\mathcal{L}_{\text{contrast}} = \operatorname{InfoNCE}(h^a,h^b)
}
\]

where \(h^a\) and \(h^b\) are high-level activations from two masked or source-split views that should share slow semantic state.

Do not contrast all latents. Do not contrast adjacent full overlapping windows without masking. Do not force sharp temporal-relational features to be smooth.

The first experiment I would run is:

\[
\Omega = \{-3,-2,-1,0\},
\]

\[
M_aX_t = [x_{t-3},x_{t-1}],
\qquad
M_bX_t = [x_{t-2},x_t],
\]

and:

\[
\mathcal{L}
=
\mathcal{L}_{\text{recon}}
+
\lambda_{\text{sparse}}\mathcal{L}_{\text{sparse}}
+
\lambda_{\text{dead}}\mathcal{L}_{\text{dead}}
+
\lambda_c
\operatorname{InfoNCE}
\big(E_H(M_aX_t),E_H(M_bX_t)\big).
\]

This is the cleanest generalization of the temporal SAE contrastive idea to the temporal crosscoding setting.

---

## References

- Temporal SAE paper: <https://arxiv.org/abs/2511.05541>
- Main text rendering used for review: <https://ar5iv.org/pdf/2511.05541>
- Temporal SAE codebase: <https://github.com/AI4LIFE-GROUP/temporal-saes>
