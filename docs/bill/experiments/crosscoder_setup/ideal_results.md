---
author: bill
date: 2026-02-25
tags:
  - experimental design
---

# Some additional thoughts on Toy Model Setup

## Introduction

Earlier, we discussed ideas behind toy model setup here: [[toy_model_ideas.md]]. 

At a high level, the toy model imposes temporal-induced correlations between token features. Ideally, a crosscoder would be able to exploit the correlation/persistence of features through time in order to extract features efficiently and potentially separate "slow-moving" v.s. "fast-moving" features (as mentioned in Bhalla et. al (https://arxiv.org/pdf/2511.05541)). 

## Different ways to enforce temporal correlation
There are (broadly) two ways to "enforce" temporal correlation, and any correlation can be decomposed into a mixture of each two: correlations between activation magnitudes, and correlations between activation probabilities. "Intuitively" (dangerous word!), it makes sense for activation probabilities to be correlated in natural text as opposed to activation magnitudes, since concepts/features are sparse and probably cluster closely together across tokens in text.

I let Claude write a brief writeup comparing the two here, the math is correct: [[gating_correlation_math.md]].

It seems that, for sparse regimes, it would make sense to control correlation between activation probabilities as opposed to activation magnitudes.



## Ideal results of a crosscoder
It makes sense that features who activate with lower magnitude are recovered less often by models. For an L1-SAE, the theoretical gain from representing a feature is $p_i(v_i - \lambda s_i)$ where $v_i = E[a_i^2], s_i = E[|a_i|], \mu_i = E[a_i]$ where $a_i$ is the activation strength, and $p_i$ is the probability of firing, if we assume our features are mostly orthogonal. Let's assume that the variance of $a_i$ when our feature is activated is $\sigma_i^2$. We may also assume that our the support for our activation is bounded, $|a_i| < M$. Then we have $v_i < M s_i$. 

Then for this feature to be added, we must have $$Ms_i > \lambda s_i$$ or $M > \lambda$. This is a very loose inequality, but it helps give proof to the following idea: if our activation value is very small, it is hard a sparse SAE to recover it.


Broadly, in this setup, we can classify features with two binary labels: high/low activations, and high/low temporal correlation. 

SAEs will struggle on features that have low activations and high temporal correlation, since they do not exploit the temporal relation between variables. The hope is that a crosscoder-informed architecture would be able to identify these structures.


## Validating Toy Model
A toy model is only valid if it simplifies some higher-level idea we would expect to happen. The setup we have propose is not useful if there do not exist features that have low activations and high temporal correlations.

Another experiment to look at (potentially before our toy model) that would be useful would be the following:
1. Examine SAE-recovered features on actual text, using an open-sourced model. Look at the temporal correlations of features (maybe at lags 1-10) and look at the magnitudes of these features.
2. If there exist "slow-moving" features (e.g. something corresponding to an ominous tone or joyful tone) with relatively low feature activation, this gives us confidence to go through with our toy model setup



