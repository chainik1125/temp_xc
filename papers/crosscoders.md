This note introduces sparse crosscoders, a variant of sparse autoencoders or transcoders for understanding models in superposition. Where autoencoders encode and predict activations at a single layer, and transcoders use activations from one layer to predict the next, a crosscoder reads and writes to multiple layers. Crosscoders produce shared features across layers and even models. They have several applications:

Cross-Layer Features - Crosscoders allow us to think of features as being spread across layers, resolving cross-layer superposition and tracking persistent features through the residual stream.
Circuit Simplification - By tracking features that continue to exist in the residual stream, crosscoders can remove "duplicate features" from analysis and allow features to "jump" across many uninteresting identity circuit connections, and generally simplify circuits.
Model Diffing - Crosscoders can produce shared sets of features across models. This includes one model across training or finetuning, and also completely independent models with different architectures.
This note will cover some theoretical examples motivating crosscoders, and then present preliminary experiments applying them to cross-layer superposition and model diffing. We also briefly discuss the theory of how crosscoders might simplify circuit analysis, but leave results on this for a future update.

(1) Motivating Examples
(1.1) Cross-Layer Superposition
According to the superposition hypothesis, neural networks represent more features than they have neurons by allowing features to be non-orthogonal. One consequence of this is that most features are represented by linear combinations of multiple neurons:

At first blush, the idea that this kind of superposition might be spread across layers might seem strange. But if we think about it carefully, it's actually relatively natural in the context of a transformer with a reasonable number of layers.

One interesting property of transformers is that, because the residual stream is linear, we can draw them as different, equivalent graphs. The following graph highlights the idea that two layers can be thought of as "almost parallel branches", except that they have an extra edge that allows the earlier layer to influence the later.

If we consider a one-step circuit computing a feature, we can imagine implementations where the circuit is split across two layers, but functionally is in parallel. This might actually be quite natural if the model has more layers than the length of the circuit it is trying to compute!

If features are jointly represented by multiple layers, where some of their activity can be understood as being in parallel, it's natural to apply dictionary learning to them jointly. We call this setup a crosscoder, and will return to it in the next section.

It's worth noting that jointly applying dictionary learning to multiple vectors is precisely what we do when models literally have parallel branches with cross-branch superposition!

(1.2) Persistent Features and Circuit Complexity
Crosscoders can help us when there's cross-layer superposition, but they can also help us when a computed feature stays in the residual stream for many layers. Consider the following hypothetical "feature lifecycle" through the residual stream:

If we tried to understand this in terms of a residual stream feature at every layer, we'd have lots of duplicate features across layers. This can lead to circuits which seem much more complex than they need to.

Consider the following hypothetical example, in which features 1 and 2 are present by layer L, and are combined (say via an "and") to form feature 3 via MLPs in layers L+2 and L+3, and then all three features persist in layer L+4. On the left panel of the figure below, we see that per-layer SAEs would produce 13 features in total, corresponding to features 1, 2, and 3 at each layer they are present. The causal graph relating them has many arrows, most for persistence (a feature causes itself in later layers) and two for each of the stages in which feature 3 is computed from 1 and 2. An ideal crosscoder picture, on the right, would have just three features and a simple causal graph.

This means that crosscoders may also give us a strategy for radically simplifying circuits if we use an appropriate architecture where, as in the above picture, feature encoders read in from a single residual stream layer and their decoders write out to downstream layers.

As an example suppose we have feature i, whose encoder lives in layer 10, and feature j, whose encoder lives in layer 1 (but whose decoder projects to all subsequent layers).  Suppose we determine (using ablations, gradient attributions, or some other method), the activity of a feature i is strongly attributable to the component of feature j that decodes to layer 10.  The crosscoder allows us to immediately “hop back” and assign this attribution to feature j’s activity, as computed  in layer 1, instead of attributing through a chain of per layer SAEs propagating the same underlying feature i . In doing so, we can potentially uncover circuits whose depth is much smaller than the number of layers in the model.

We note, however, that there are some conceptual risks with this approach – the causal description it provides likely differs from that of the underlying model.  We plan to explore this approach further in future updates.


(2) Crosscoder Basics
Where autoencoders encode and predict activations at a single layer, and transcoders use activations from one layer to predict the next, a crosscoder reads and writes to multiple layers (subject to causality constraints we may wish to impose, we'll discuss this later). This general idea of a sparse coder operating on multiple layers is also explored in forthcoming work by Baskaran and Sklar.

We can think of autoencoders and transcoders as special cases of the general family of crosscoders.

The basic setup of a crosscoder is as follows. First, we compute the vector of feature activations $f\left(x_j\right)$ on a datapoint $x_j$ by summing over contributions from the activations of different layers $a^l\left(x_j\right)$ for layers $l \in L$ :
$$
f\left(x_j\right)=\operatorname{ReLU}\left(\sum_{l \in L} W_{e n c}^l a^l\left(x_j\right)+b_{e n c}\right)
$$
where $W_{\text {enc }}^l$ is the encoder weights at layer $l$, and $a^l\left(x_j\right)$ is the activations on datapoint $x_j$ at layer $l$. We then try to reconstruct the layer activations with approximations $a^{l^{\prime}}\left(x_j\right)$ of the activations at layer $l$ :
$$
a^{l^{\prime}}\left(x_j\right)=W_{d e c}^l f\left(x_j\right)+b_{d e c}^l
$$

And have a loss:
$$
L=\sum_{l \in L}\left\|a^l\left(x_j\right)-a^{l^{\prime}}\left(x_j\right)\right\|^2+\sum_{l \in L} \sum_i f_i\left(x_j\right)\left\|W_{d e c, i}^l\right\|
$$

Note that the regularization term can be rewritten as:
$$
\sum_{l \in L} \sum_i f_i\left(x_j\right)\left\|W_{d e c, i}^l\right\|=\sum_i f_i\left(x_j\right)\left(\sum_{l \in L}\left\|W_{d e c, i}^l\right\|\right)
$$

That is, we weight the L1 regularization penalty by the L1 norm of the per-layer decoder weight norms $\left(\sum_{l \in L}\left\|W_{d e c, i}^l\right\|\right)$, where $\left\|W_{d e c, i}^l\right\|$ is the L 2 norm of a single feature's decoder vector at a given layer of the model. In principle, one might have expected to use the L2 norm of per-layer norms $\left(\sqrt{\sum_{l \in L}\left\|W_{\text {dec }, i}^l\right\|^2}\right)$. This is equivalent to treating all the decoder weights as a single vector and weighting by its L2 norm, which might seem like the natural thing to do.

However, there are a two reasons to prefer the L1 norm version:

Baseline Loss Comparison: The L1-of-norms version enables apples-to-apples loss comparisons between crosscoders and single-layer SAEs (or transcoders) – the loss of a crosscoder is directly comparable to the sum of losses of per-layer SAEs trained on the same set of layers.  If we instead use the L2-of-norms version, crosscoders can obtain a much lower loss than the sum of per-layer SAE losses, as they would effectively obtain a loss “bonus” by spreading features across layers.
Layer-Wise Sparsity Surfaces Layer-specific Features: Using the L2 version encourages features to be spread out across layers, as the marginal increase in L2-of-norms by allowing a feature to reconstruct more layers is decreased once the feature already has nontrivial decoder norm in other layers.  The L1-of-norms version, by contrast, provides no explicit incentive to “spread out” features. Empirically, we've found that the L1-of-normsversion sometimes exposes phenomena of interest more effectively in the context of  model diffing – in particular, it uncovers a mix of shared and model-specific features, while the L2-of-norms version results in uncovering only shared features.
On the other hand, the L2 version more efficiently optimizes the frontier of MSE and global L0 across all layers of the model.  Thus, for applications where uncovering layer or model-specific features is not important, and where it is not important to be able to compare loss values to per-layer SAEs, the L2-of-norms version may be preferable.  In this report, all experiments used the L1-of-norms version.

(2.1) Crosscoder Variants
This basic version above is what we'd call an "acausal crosscoder". Many variants are in fact possible. In particular, several important dimensions are:

Cross-Layer Approach - Are we doing crosscoders or normal SAEs?
Causality - Do we want to have a setup where early activations predict later activations, similar to transcoders?
Locality - Do we apply the crosscoder to all layers, or just some? Or maybe apply them to layers from different models?
Target - Are we modeling the residual stream or layer outputs?
The following table summarizes the variants:

We have found both weakly and strictly causal crosscoders helpful for simplifying feature interaction graphs in our circuits work, but there remain open questions as to how faithfully validate these analyses. Note that strictly causal crosscoder layers as presented here cannot capture the computation performed by attention layers.  Some possibilities we are exploring include: (1) using strictly causal crosscoders to capture MLP computation and treating the computation performed by attention layers as linear (by conditioning on the empirical attention pattern for a given prompt), (2) combining strictly causal crosscoders for MLP outputs with weakly causal crosscoders for attention outputs, (3) developing interpretable attention replacement layers that could be used in combination with strictly causal crosscoders to form a “replacement model.”


(3) Cross-Layer Features
(3.1) Performance and Efficiency of Crosscoders vs. SAEs
Can crosscoders actually uncover cross-layer structure?  To explore this question, we first trained a global, acausal crosscoder on the residual stream activations of all layers of an 18-layer model.  We compared its performance to that of 18 SAEs trained separately on each of the residual stream layers.  We used a fixed L1 coefficient for the sparsity penalty. Note that we designed our loss to be comparable to a baseline SAE loss with the same L1 penalty, as discussed above. We separately normalize the activations of each layer prior to training the crosscoder, so that each layer contributes comparably to the loss.

For each approach, we swept over the number of training steps and the number of total features to select the optimal number of features at different FLOPS budgets.  We are interested in how the dictionary performance scales with the total number of features in the crosscoder / across all SAEs, and with the amount of compute used in training.  Note that for a model with L layers, a global, acausal crosscoder with F total features uses the same number of training FLOPS as a collection of per-layer SAEs with F features each (and thus with L*F total features).  Or viewed another way, a collection of single-layer SAEs with F total dictionary features summed across all the SAEs can be trained with L times fewer FLOPS than a single crosscoder with F dictionary features.  Thus, crosscoders must substantially outperform SAE on a “per-feature efficiency” basis to be competitive in terms of FLOPs.

First we measure the eval loss of both approaches (MSE + decoder norm-weighted L1 norm, summed across layers):

Description of image:

```
The image consists of two side-by-side line charts comparing the performance of two different machine learning models or techniques. A common legend applies to both charts, and an explanatory text note is located on the far right.

Legend & Data Series

* **Blue line with circle markers:** "all-layers crosscoder"
* **Orange line with circle markers:** "per-layer SAEs"

Chart 1 (Left): "Loss vs Number of Features"

* **X-axis:** Labeled "Number of features". The scale is logarithmic (base 2), with major tick marks at $2^{13}$, $2^{15}$, $2^{17}$, $2^{19}$, and $2^{21}$.
* **Y-axis:** Labeled "Minimum loss (arb. unit, log scale)". The scale displays linear increments (5.0, 5.8, 6.6, 7.4, 8.2, 9.0, 9.8, 10.6, 11.4, 12.2, 13.0).
* **Data Trends:** Both series show a downward trend (loss decreases as the number of features increases). The "all-layers crosscoder" (blue) consistently achieves a lower minimum loss than the "per-layer SAEs" (orange) for any given number of features.
* **Annotation:** A visual indicator connects a point on the blue line (at approximately x = $2^{17}$, y = 6.6) horizontally to a point on the orange line (at approximately x = $2^{20}$, y = 6.6). A dotted vertical line extends upward from the orange point to a text box that reads: **"Crosscoders are ~8x more feature efficient."** (Note: $2^{20}$ / $2^{17}$ = $2^3$ = 8).

Chart 2 (Right): "Loss vs FLOPs"

* **X-axis:** Labeled "Training FLOPS (arb. unit, log scale)". The scale is logarithmic (base 2), with major tick marks at $2^2$, $2^4$, $2^6$, $2^8$, $2^{10}$, and $2^{12}$.
* **Y-axis:** Shares the same scale and gridlines as Chart 1, though the axis labels are omitted for visual cleanliness.
* **Data Trends:** Both series show a downward trend (loss decreases as Training FLOPS increase). In this chart, the "per-layer SAEs" (orange) line is shifted to the left and is lower than the "all-layers crosscoder" (blue) line, meaning SAEs achieve a lower loss for the same amount of compute.
* **Annotation:** A visual indicator connects a point on the orange line horizontally to a point on the blue line at a similar loss level (around y = 6.4) near the far right of the x-axis. A dotted vertical line extends upward from the blue point to a text box that reads: **"Crosscoders are ~2x less FLOP efficient."**

Supplemental Text Node (Far Right)

A standalone text block below the legend provides additional context:

* "Note: The 'Loss vs Features' plot attains lower losses because the coders are 'overtrained' past FLOP-optimality."
```

We found that, controlling for the total number of features across layers, crosscoders substantially outperform per-layer SAEs on eval loss.  This result indicates that there is a significant degree of redundant (linearly correlated) structure across layers, which are interpreted by the crosscoder as cross-layer features.  However, with respect to training FLOPS, crosscoders are less efficient than per-layer SAEs at achieving the same eval loss, by a factor of about 2 at large compute budgets.

In other words, for a fixed number of total features, crosscoders are able to make more efficient use of their resources by identifying shared structure across layers, allowing them to lump together identical features across layers as a single cross-layer feature, which frees up the crosscoder’s resources to spend on other features.  However, identifying this structure costs compute at training time.

However, eval loss is only one measure of the crosscoder’s usefulness.  Since our loss scales the sparsity penalty by the sum of decoder norms across layers, it effectively measures (an L1 relaxation of) the sparsity of (feature, layer) tuples.  Thus, it provides a sense of how well any single layer of the model can be described as a sparse sum of crosscoder features, vs. SAE features.  However, we may also be interested in how well the activity across the entire model can be described as a sparse sum of crosscoder features, vs. SAE features.  For this purpose, the metric of interest is the (MSE, L0) value of each method, where in the per-layer SAE case we sum L0 norm across all the SAEs.  We show (MSE, L0) values at optimal values of SAE / crosscoder training loss over a set of values of training FLOPS.

Description of image:
```
The image presents a single scatter plot titled **"(MSE, Total L0) for FLOP-Equivelant Coders"** (Note: "Equivalent" is misspelled in the original image source). The chart compares two models based on their reconstruction error and the total number of active features, specifically matching pairs of models that use an equivalent amount of compute (FLOPs).

Legend & Data Series

* **Blue line with circle markers:** "all-layers crosscoder"
* **Orange line with circle markers:** "per-layer SAEs"

Chart Details: "(MSE, Total L0) for FLOP-Equivelant Coders"

* **X-axis:** Labeled "Normalized reconstruction error" (which corresponds to MSE). The scale is linear, ranging from 0.00 to 0.35, with major tick marks every 0.05.
* **Y-axis:** Labeled "Total # of active features" (which corresponds to Total L0). The scale is linear, ranging from 0 to 1600, with major tick marks every 200.
* **Data Trends:** The plot displays distinct pairs of data points, with one blue point and one orange point in each pair connected by a thin, dotted gray line.
* The "all-layers crosscoder" (blue) points are clustered tightly along the bottom of the graph, showing very low numbers of active features (well below 100).
* The corresponding "per-layer SAEs" (orange) points are positioned much higher on the y-axis, ranging approximately from 700 to 1500 active features.
* For each connected pair, the blue point generally sits slightly to the left of the orange point, indicating a lower normalized reconstruction error.


* **On-Chart Annotation:** A text block on the left side of the chart area reads: **"FLOP equivalent crosscoders achieve 15-40x lower total L0 than SAEs, with better MSE"**

Supplemental Text Note (Far Right)

A standalone text block to the right of the chart provides methodology context:

* "We compare the 'Total L0' (ie. L0 summed over layers) for FLOP-equivalent crosscoders and SAEs. This is the less favorable comparison for crosscoders, and they still perform dramatically better than SAEs."
```



Viewed from this perspective, crosscoders provide a dramatic benefit over per-layer SAEs.  By consolidating shared structure across layers, they exhibit a much less redundant (and therefore more concise) decomposition of the entire model’s activations. In theory, the same consolidation might be achievable via post-hoc analysis on SAE features, e.g. by clustering features based on the similarity of their activations.  However, in practice, this analysis may be difficult, particularly due to stochasticity in SAE training. Crosscoders effectively “bake in” this clustering at training time.

Summarizing the results at a high level, the efficiency crosscoders and per-layer SAEs can be compared in essentially two ways.  In terms of the tradeoff between reconstruction error and the sparsity of the feature set used to reconstruct a single layer’s activity, crosscoders make more efficient use of dictionary features but less efficient use of training FLOPS.  In terms of the reconstruction error / sparsity tradeoff in reconstructing the entire model’s activity, crosscoders provide an unambiguous advantage, by resolving redundant structure across layers.

(3.2) Analysis of Crosscoder Features
We next conducted some basic analyses of the crosscoder features.  We were especially interested in features’ behavior across layers: (1) Do crosscoder features tend to be localized to a few layers, or do they span the whole model? (2) Do crosscoder features’ decoder vector directions remain stable across layers, or can the same feature point in different directions in different layers?

Addressing question (1), below we plot the decoder weight norms of 50 randomly sampled crosscoder features across the layers of the model (which are representative of trends we have observed in the full collection of features).   For each feature, we rescale the norms so that the maximum value is 1, for ease of visual comparison.

We see that most features tend to peak in strength in a particular layer, and decay in earlier and later layers.  Sometimes the decay is sudden, indicating a localized feature, but often it is more gradual, with many features having substantial norm across most or even all layers.


The ability to track the presence of features across layers is spiritually similar to results by Yun et al. 
[13]
, who use dictionary learning to fit one dictionary that models activations at all layers of the residual stream. As far as we know, Yun et al. were the first to ask this question in the context of features. 1 How do they differ? Conceptually, Yun et al.'s approach considers a feature to be the same if it is represented by the same direction across layers, while crosscoders consider a feature to be the same if it activates on the same data points across layers. Critically, crosscoders allow feature directions to change across layers (“feature drift”), which we'll soon observe appears to be important.

Returning to the above plot, is the existence of gradual formation of features distributed across layers evidence for cross-layer superposition? While it's definitely consistent with the hypothesis, it could also have other explanations. For example, a feature could be unambiguously produced at one layer and then amplified at the next layer. More research – ideally circuit analysis – would be needed to confidently interpret the meaning of gradual feature formation.

We now return to the second of our original questions, regarding the embedding directions of crosscoder features. Below, for a few example crosscoder features, we show:

Top:  the strength of the feature’s decoder norm in each layer (as above), the layer in which the feature’s strength peaks, and the cosine similarity between the feature’s decoder direction in that peak layer and all other layers
Bottom: the projection of the feature’s decoder vector in layer i onto its decoder vector direction in layer j (thus, the diagonal of each plot indicates the norm of the feature’s decoder vector in each layer)
The leftmost column is an example of a feature whose decoder direction drifts across layers at roughly the same spatial scale at which its norm decays.  The middle column is an example of a feature whose decoder direction is fairly stable over the layers in which the feature has appreciable norm.  The right column is an example of a feature that persists throughout the model, but with rapidly changing decoder direction.


These examples were selected to illustrate the range of feature archetypes we find.  Below, we show this information for 36 randomly selected features to give a more representative picture.

Overall, we find that most features’ decoder directions are much more stable across layers than would be expected by chance, but also that they drift substantially across layers, even in layers where the feature decoder norm remains strong. The specific behavior varies considerably by feature.  This suggests that the cross-layer features uncovered by our crosscoders are not simply passively relayed via residual connections.

Note that we do not conduct a systematic analysis of qualitative feature interpretability in this work. Anecdotally, we find that crosscoder features are similarly interpretable to sparse autoencoder features, and crosscoder features that peak in a particular layer are qualitatively similar to features obtained from a sparse autoencoder trained on that layer.  We plan to more rigorously evaluate crosscoder feature interpretability in future work.

(3.3) Masked Crosscoders
(3.3.1) Locality experiments
We experimented with locally masked “convolutional” variants of crosscoders, in which each feature is assigned a local window of K layers that it is responsible for encoding / decoding.  We hoped that this would allow us to capture the benefits of crosscoders while minimizing the FLOPS expense at crosscoder training time.  However, we found that eval loss interpolated fairly linearly as we varied the convolutional window K from 1 (per-layer SAE case) to n_layers (global acausal crosscoder) – there was no obvious inflection point that optimized the performance / cost tradeoff.  Put another way, the performance of a locally masked cross-coder was similar to that of a smaller, FLOPS-matched global crosscoder. This is consistent with the picture from the distribution of features across layers, as seen in the previous section.

(3.3.2) Causality experiments
We also experimented with “weakly causal” crosscoders.  We focused in particular on an architecture in which each feature is assigned an encoder layer i – its encoder reads in from layer i alone, and its decoder attempts to reconstruct layer i and all subsequent layers.  We found that in terms of eval loss performance, this architecture’s FLOPS efficiency is in between that of per-layer SAEs (slightly worse) and global, acausal crosscoders (slightly better).  With respect to dictionary size, its performance lagged behind that of global crosscoders by a factor of 3 to 4. 2 We have found this weakly causal crosscoder architecture promising for circuits analysis (strongly causal approaches also seem promising).

We have also conducted preliminary experiments with strictly causal “cross-layer transcoders,” in which each feature reads in from the residual stream at a layer L, and attempts to predict the output of the MLPs in layers L, L+1, L+2, … NUM_LAYERS.  When examining the decoder norms of these features, we find a mix of:

local features, that primarily predict the immediate next MLP output
global features, that predict MLP outputs at all subsequent layers with roughly equal strength
Features that lie in between these two extremes

(3.3.3) Pre/post MLP crosscoders
One interesting application of crosscoders is to analyze the differences in feature sets before and after an MLP layer, and the computations that give rise to “new” features in the MLP output (see the section on Model Diffing for related experiments analyzing cross-model differences).  To achieve this, we can train a crosscoder on the pre-MLP residual stream space and the outputs that the MLP writes back to the residual stream.  We use a masking strategy in which the features’ encoders read only from the pre-MLP space, but their decoders attempt to reconstruct both the pre-MLP activity and the MLP output. Note that this architecture differs from a regular transcoder, in which the features are only responsible for reconstructing the MLP output.

This architecture has two nice properties.  First, it allows us to identify features that are shared between the pre and post-MLP spaces, and features that are specific to one or the other.  To see this, we can plot the relative norms of the decoder vectors within each space.  Remarkably, we see a clear trimodal structure, corresponding to pre-only, shared, and post-only (i.e. “newly computed”) features.


Second, because we constrained the encoder vectors to live only in the pre-MLP space, this architecture allows us to analyze how these “newly computed” features were computed from existing features in the residual stream (similar to how one would analyze a transcoder). In particular, the inputs to a newly created feature can be computed by taking the dot product of the downstream feature’s encoder vector with the upstream features’ decoder vectors, and weighting by the source features activation (either in a particular context, or averaged over dataset examples).  Anecdotally, we often find that the post-MLP feature represents a more abstract concept and its strongest inputs are specific instances of that concept.  For instance, we find one post-MLP feature that activates on words indicating uniqueness, like “special,” “particular,” “exceptional,” etc., and its pre-MLP inputs each fire for particular words in this category, in particular contexts.

We also analyzed the extent to which “stable” features (arbitrarily those with between 0.3 and 0.7 relative decoder norm weight in post-MLP space) tend to be embedded along similar directions in pre- and post-MLP space.  Interestingly, we found a positive correlation on average, but with high variance, and relatively low in absolute terms.  This result may explain why feature directions tend to drift across layers – it appears that MLP layers relay many features without nonlinear modification, but along a different axis .




