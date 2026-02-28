---
title: "Introduction to the project"
author: Han Xuanyuan
date: 2026-02-16
type: document
---

## intro

Sparse autoencoders attempt to decompose a models activations into meaningful atomic “features”. Although this has been moderately successful at a given layer, SAEs have two fundamental shortcomings. They cannot explain `global' features present throughout different layers in the network, and they cannot explain features that persist across sequence positions. The second limitation is especially crucial for reasoning models. In this case, there is growing evidence that reasoning capabilities act as independent features that influence the model over many subsequent tokens (https://arxiv.org/pdf/2510.07364).

A natural proposal is hence to generalize SAEs that decompose activations at a single token into “temporal SAEs” that find features across sequence positions. Although there are a number of different architectural choices that could implement this, a natural choice is to consider SAEs based on Tensor Networks (https://arxiv.org/pdf/1306.2164, https://arxiv.org/abs/2011.12127). Tensor Network SAEs provide a way to control the extent to which features are present on a given token, and the extent to which features are delocalized throughout the network.

In this project we would demonstrate the practical utility of Temporal SAEs in the following steps:

Construct a toy model setting based on https://arxiv.org/pdf/2508.16560. Measure the Pareto fronteir for various Temporal SAE architectures (naive stacked SAEs, temporal crosscoders, and Tensor Network SAEs) for different choices of correlations between tokens.

Measure between-token feature correlations in relevant alignment scenarios, and select a minimal proof of concept on a single-GPU model. If possible, construct a dataset with stable correlation behaviours between tokens.

Train a Temporal SAE on base and reasoning models and construct the Pareto fronteir of computational cost vs. reconstruction. Conduct a detailed study of the most important features and compare this to a baseline of stacked SAEs.

If we are able to identify qualitatively different features from temporal SAEs, and better Pareto efficiency in reasoning vs. base models this would constitute a valuable new tool for understanding models and open up a bridge to established techniques in other fields. This would be a strong paper.