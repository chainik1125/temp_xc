## Contents
- 1 Introduction
- 2 Preliminaries
  - 2.1 Slowness Principle
  - 2.2 Contrastive Learning and the Slowness Principle
  - 2.3 Disentanglement
  - 2.4 Fréchet Inception Distance
- 3 Slow Variational Autoencoding Framework
- 4 Slowness loss terms
  - 4.1 Lp-norm Slowness
  - 4.2 SlowVAE
  - 4.3 S-VAE
- 5 Empirical comparison
  - 5.1 Experiment Setting
  - 5.2 Evaluation of Downstream Task Performance
  - 5.3 Influence of Hyperparameters
- 6 Predicting Downstream Task Performance
- 7 Slow Autoencoding Methods from an Information Theoretic Point of View
- 8 Conclusion
- References
- Appendix A Experiment Details
  - A.1 Ball Experiment
  - A.2 Pong experiment
  - A.3 Deepmind Lab Dataset
- Appendix B Performance Result Details
- Appendix C Analysis

## Abstract

Abstract The slowness principle is a concept inspired by the visual cortex of the brain.
It postulates that the underlying generative factors of a quickly varying sensory signal change on a slower time scale.
Unsupervised learning of intermediate representations utilizing abundant unlabeled sensory data can be leveraged to perform data-efficient supervised downstream regression.
In this paper, we propose a general formulation of slowness for unsupervised representation learning adding a slowness regularization term to the estimate lower bound of the β 𝛽 \beta -VAE to encourage temporal similarity in observation and latent space.
Within this framework we compare existing slowness regularization terms such as the L1 and L2 loss used in existing end-to-end methods, the SlowVAE and propose a new term based on Brownian motion.
We empirically evaluate these slowness regularization terms with respect to their downstream task performance and data efficiency.
We find that slow representations lead to equal or better downstream task performance and data efficiency in different experiment domains when compared to representations without slowness regularization.
Finally, we discuss how the Fréchet Inception Distance (FID), traditionally used to determine the generative capabilities of GANs, can serve as a measure to predict the performance of pre-trained Autoencoder model in a supervised downstream task and accelerate hyperparameter search.

## 1 Introduction

Learning representations that capture general high level information from abundant unlabeled sensory data remains a challenge for unsupervised representation learning.
Research in neuroscience suggests that a major difference between state-of-the-art deep learning architectures and the human brain is that cells in the brain do not react to single stimuli, but instead extract invariant features from sequences of fast changing sensory input signals Bengio and Bergstra (2009).
Evidence found in the hierarchical organization of simple and complex vision cells shows that time-invariance is the principle after which the cortex extracts the underlying generative factors of these sequences and that these factors usually change slower than the observed signal Wiskott and Sejnowski (2002); Berkes and Wiskott (2005); Bengio and Bergstra (2009).

Computational neuroscientists have named this paradigm the slowness principle wherein individual measurements of a signal may vary quickly, but the underlying generative features vary slowly.
For example, individual pixel values in a video change rapidly during short periods of time, but the scene itself changes slowly.
This principle has found application in Slow Feature Analysis (SFA) as proposed in Wiskott and Sejnowski (2002).
SFA has shown promising results in computational neuroscience but little research has explored the possible applications of the underlying slowness principle to state of the art unsupervised representation learning methods.
Previous research has bridged this gap by employing temporal contrastive L1 and L2 losses in end-to-end tasks Mobahi et al. (2009); Sermanet et al. (2018); Zou et al. (2012) such as classification and view-point invariant robot imitation learning.
Recently an unsupervised representation learning method, the SlowVAE Klindt et al. (2020), has been leveraging the observed statistics of natural transitions in observation space to extended the VAE objective with a sparse temporal prior.

With this paper, we aim to provide a framework for learning slow representations in an unsupervised way using Variational Autoencoders.
The framework adds an interchangeable slowness loss term to the evidence lower bound (ELBO) of the VAE objective allowing us to investigate the effects of different slowness regularizers on the learned representations.
In addition to the L1 and L2 norm used in previous work and the slowness loss term proposed in Klindt et al. (2020), we propose a new slowness term based on a Brownian motion prior for latent space evolution which we call the S-VAE.
We empirically compare the $\beta$-VAE, the SlowVAE, S-VAE and $L1$/$L2$ slowness terms within our framework with respect to their performance and data efficiency on downstream few-shot regression tasks.
We find that although slow methods out-perform the $\beta$-VAE in both best case downstream performance and few-shot data efficiency, there is no universally superior slowness term.

This begs the question: how can one measure and predict general “goodness” of representations with respect to downstream tasks without having to evaluate performance on the downstream tasks directly.
To this end we conduct a large scale hyperparameter search and visualize latent space properties for a low dimensional problem.
Furthermore, we investigate quantitative measures for the quality of latent representations and find that the Fréchet Inception Distance as proposed by Heusel et al. (2017) correlates with the downstream task performance(^2^22The FID measures the ability to generate realistic images from random samples of the latent distribution by comparing the distribution of the generated data to the true data distribution.).
Being able to predict the downstream task performance, just by sampling from the latent distributions of the autoencoder, greatly accelerates the hyperparameter search during the unsupervised pre-training of the VAE and helps identifying good encoder model without the need to perform the downstream task.

The key contributions of this paper are as follows:

- •
A general formulation of slowness in unsupervised representation learning
- •
A review of existing slowness models in context of the formulation
- •
Proposing a new slowness model S-VAE
- •
Empirical analysis of downstream task performance when learning from sparse labeled data using unsupervised pre-training with abundant unlabeled data
- •
Proposing the Fréchet-Inception Distance (FID) to be used as a metric to compare and predict downstream task performance of VAE models

## 2 Preliminaries

### 2.1 Slowness Principle

The slowness principle is based on the assumption that the true generative factors of a signal vary on slower time scales than raw sensory signals.
Research in computational neuroscience suggests that cell structures in the visual cortex have emerged based on the underlying principle of extracting slowly varying features from the environmentBerkes and Wiskott (2005).
Leveraging this principle, we can extract higher level invariant scene information which usually changes slower than for example the individual pixel values of a video.

The most well known application of the slowness principle is the slow feature analysis method (SFA) introduced in Wiskott and Sejnowski (2002).
SFA is an unsupervised learning algorithm designed to extract linearly decorrelated features by expanding and transforming the input signal such that it can be optimized for finding the most slowly varying features from an input signal Wiskott and Sejnowski (2002).
Extending the SFA method to nonlinear features has shown that the learned features share many characteristics with those of complex cells in the V1 cortex Berkes and Wiskott (2005).
Further applications of the slowness principle include transformation invariant object detection Franzius et al. (2011), pre-training of neural networks for improved performance on the MNIST dataset  Bengio and Bergstra (2009) and the self organization of grid cells, structures in the rodent brain used for navigation Franzius et al. (2007b, a).

### 2.2 Contrastive Learning and the Slowness Principle

The objective of contrastive learning is to encode observations and place them in a latent space using a metric score that allows to express (dis-)similarity of the observations.
Contrastive learning has been successfully applied in reinforcement learning Laskin et al. (2020) and most recently for object classification in SimCLR Chen et al. (2020a, b).
These methods use a contrastive loss on augmented versions of the same observation, effectively learning transformation invariant features from images, and show that these representations benefit reinforcement learning and image classification tasks.

When using the time as the contrastive metric, similar to the slowness principle, we talk about time-contrastive learning.
Time-contrastive learning has been applied successfully to learning view-point-invariant representations for learning from demonstration with a robot Sermanet et al. (2018).
Similar to our work, in Mobahi et al. (2009), used the coherence in video material to train a Convolutional Neural Network (CNN) for a variety of specific tasks.
While training two CNNs in parallel with shared parameters, in alternating fashion a labeled pair of images was used to perform a gradient update minimizing training loss followed by selecting two unlabeled images from a large video dataset to minimize a time-contrastive loss based on the $L1$ norm of the representations at each individual layer.
The experiments showed that supervised tasks can benefit from the additional pseudo-supervisory signal and that features invariant to pose, illumination or clutter can be learned.
Compared to  Mobahi et al. (2009) where a specific task is learned end-to-end with an additional supervisory signal, the slow methods presented in this paper decouple the unsupervised representation learning step from downstream task learning with the goal of learning task-agnostic representations.

Compared to the above methods, the GP-VAE proposed by Fortuin et al. (2020) is a model for learning temporal dynamics for problems such as reconstructing missing input features especially in a medical context.
The core idea of the GP-VAE is to learn a latent embedding of high dimensional sequential data and to model latent dynamics using a Gaussian Process prior.
The authors claim this prior facilitates representations which are smoother and allow the reconstruction of missing features in the input space.
Compared to the GP-VAE, the methods presented in this paper address the problem of learning good representations for downstream tasks, instead of learning temporal dynamics of the signal.
Another noteworthy method for learning temporal dynamics is the temporal difference variational autoencoder (TD-VAE) Gregor et al. (2019) which learns representations that encode an uncertain belief state from which multiple possible future scenarios can be rolled out.

### 2.3 Disentanglement

Another concept related to unsupervised representation learning, especially when talking about the $\beta$-VAE, is disentanglement.
In this paper disentanglement is not compared as a metric, but used when discussing the properties of latent representations.

Although there is no definition of disentanglement, most works agree Bengio et al. (2013); Locatello et al. (2019b); Klindt et al. (2020) on the common notion that disentangled representations should approximate the ground truth generative factors of the observation data and the dimensions should be largely independent from each other.
Ideally each disentangled factor represents one ground truth factor that led to the generation of the observation data.
Disentanglement in the context of the $\beta$-VAE has been discussed more in depth by Burgess et al. (2018).
In the $\beta$-VAE pressure on the latent bottleneck of the autoencoder limits how much information can be transmitted per sample while at the same time trying to maximize the data log likelihood.
This is done by enforcing a unit Gaussian prior on the latent distributions which results in the embedding of data points on a set of representational axes where nearby points on the axes are also close in data space.
This regularization results in these axes being the main contributors to improvements in the data log likelihood and therefore often coincide with the ground truth generative factors.
Some of the claimed benefits of disentangled representations are better downstream task data-efficiency and interpretability Schölkopf et al. (2012); Bengio et al. (2013); Peters et al. (2017).
However, a in their research Locatello et Al. Locatello et al. (2019b, a) show that various disentanglement methods are not able to generate disentangled representations without implicit biases on model and data, and that more disentanglement does not necessarily lead to better downstream task data-efficiency.
In a later work Locatello et al. (2020) the aforementioned challenges were addressed and the authors showed that with weak supervision it is possible to learn fair and generalizable representations.

### 2.4 Fréchet Inception Distance

In this paper we will make use of the Fréchet Inception Distance (FID) which is briefly explained in this section.
The FID as introduced in Heusel et al. (2017) is a measure for the generative capabilities of deep generative models commonly used to measure how similar the images generated by GANs are to the real data distribution.
The FID is an improvement of the the Inception Score (IS) introduced by Salimans et al. (2016) which only evaluates the distribution of the generated images and does not compare it to the true data distribution which has been shown to fail when comparing models Barratt and Sharma (2018).
It is computed by comparing the activation distributions of a Inception-v3 neural network pre-trained on the ImageNet dataset for the generated and true data using the Wasserstein-2 between the distributions $(\mu,C)$, a Gaussians with mean $\mu$ and covariance $C$ describing the generated data distribution, and $(\mu_{w},C_{w})$ describing the real data distribution.

## 3 Slow Variational Autoencoding Framework

The variational autoencoder (VAE) introduced by Kingma and Welling (2013) is a representation learning method for dimensionality reduction using a loss on the reconstruction quality of observations decoded from low dimensional representations.

Let $q_{\theta}$ be the variational approximate posterior distribution obtained by an encoder network with parameters $\theta$ and $\mathbf{z}$ be the latent vector such that $\mathbf{z}\sim q_{\theta}(\mathbf{z|o})$ where $\mathbf{o}$ is the observation to be reconstructed.
The decoder network denoted by $p_{\phi}(\mathbf{o|z})$ is parameterized by $\phi$.
Since it is computationally not tractable to directly maximize the log probability of the data, the estimate lower bound (ELBO) $\mathcal{L}$ is used for optimization:

$$ $\max_{\theta,\phi}\log p_{\phi}(\mathbf{o})\geq\mathcal{L}=\mathcal{L}_{\text{rec}}-\mathcal{L}_{\text{ib}}$ (1) $$

where $\mathcal{L}_{\text{rec}}=\mathbb{E}_{q_{\theta}(\mathbf{z|o})}[\log p_{\phi}(\mathbf{o}|\mathbf{z})]$ is the reconstruction loss between the true observation and the decoded observations which is to be maximized.
$\mathcal{L}_{\text{ib}}=D_{KL}(q_{\theta}(\mathbf{z|o})||p(\mathbf{z}))$ is constraining the information bottleneck by imposing a unit Gaussian prior and is promoting disentanglement in the latent space.
To make the sampling process differentiable (and thus trainable using gradient based optimization), the variational distribution is usually reparametrized as a Gaussian $q_{\theta}=\mathcal{N}(\mu,\Sigma)$.

Higgins et al. (2017) introduced the $\beta$-VAE by adding a parameter $\beta$ to constrain the weight of the KL divergence between the prior and the posterior to allow a trade-off between disentanglement of the latent factors and reconstruction quality.

We extend the formulation of Higgins et al. (2017) by an additional slowness constraint which is added to the constrained optimization problem as

$$ $\begin{split}\mathcal{L}_{\text{rec}}\text{ subject to }&\mathcal{L}_{\text{ib}}<\epsilon_{1},\\ &\mathcal{L}_{\text{slow}}<\epsilon_{2}\end{split}$ (2) $$

Rewriting Eq. [2](#S3.E2) under the KKT conditions results in

$$ $\mathcal{F}=\mathcal{L}_{\text{rec}}-\beta(\mathcal{L}_{\text{ib}}-\epsilon_{1})-\lambda(\mathcal{L}_{\text{slow}}-\epsilon_{2})$ (3) $$

With the simplification of $\beta,\epsilon_{1},\lambda,\epsilon_{2}\geq 0$, we can rewrite the above equation as free energy objective function

$$ $\mathcal{F}\geq\mathcal{L}=\mathcal{L}_{\text{rec}}-\beta\mathcal{L}_{\text{ib}}-\lambda\mathcal{L}_{\text{slow}}.$ (4) $$

with $\beta$ being the $\beta$-VAE parameter and $\lambda$ being a parameter to control the weight of the slowness regularization.

Compared to disentanglement in the $\beta$-VAE, slow latent methods assume temporal continuity in both training data and downstream task as the underlying rule to represent the data in latent space.
The slowness principle here serves as a bias applied to the model and training data that is grounded in neuroscientific findings as opposed to the concept of disentanglement which is based on the assumption that independent factors are generally good, which has been subject to criticism lately Locatello et al. (2019b).

## 4 Slowness loss terms

The slowness loss term $\mathcal{L}_{\text{slow}}$ is an interchangeable component used to apply the slowness principle to the $\beta$-VAE formulation.
Let $D=(\tilde{o}_{1},\ldots,\tilde{o}_{T})$ be a sequence of unlabeled data points $\tilde{o}=(\mathbf{o},t)$ consisting of an observation $(\mathbf{o})$ and the time index $(t)$.
The index is used to determine the metric $\Delta t$ that for how far apart a pair of observations at times $i,j$ are, where $i<j$.
Specifically,

$$ $\Delta t(\tilde{\mathbf{o}}_{i},\tilde{\mathbf{o}}_{j})=j-i.$ (5) $$

### 4.1 Lp-norm Slowness

The most obvious way to describe $\mathcal{L}_{\text{slow}}$ is to take a $L_{p}$-norm of the mean $\mathbf{\mu}_{i}$ and $\mathbf{\mu}_{j}$ of two encoded latent distributions from two different points in time

$$ $\mathcal{L}_{\text{slow}}(\mathbf{\tilde{o}}_{i},\mathbf{\tilde{o}}_{j})=(\mathbf{\mu_{i}-\mu_{j}})^{p}$ (6) $$

where $q_{\theta}(\mathbf{z|o})=N(\mu,\Sigma)$.

This way of enforcing temporal similarity has been explored in Mobahi et al. (2009) where a $L1$ norm was used to enforce similarity between two halfs of a siamese CNN architecture to leverage temporal similarity of observations during training.
Other use cases of a $L_{p}$-norm can be found in Zou et al. (2012) and Sermanet et al. (2018), where a $L1$ norm and a triplet loss (based on L2 norm) respectively were used to achieve viewpoint invariance by enforcing similarity of views form different angles according to temporal similarity.
Cadieu and
Olshausen (2012) claim that there are no significant differences between $L1$ and $L2$ norm for enforcing slowness in latent space.

While the above works show the benefits of $L1$ and $L2$ norm based slowness models, they are learning the task end-to-end and do not take the uncertainty $\Sigma$ of the latent distributions into account.

### 4.2 SlowVAE

The SlowVAE by Klindt et al. (2020), is based on a study of the statistics of natural transitions of object and mask positions in two acknowledged video-object segmentation datasets which follow generalized Laplace distributions.
The SlowVAE uses these Laplace distributions with a kurtosis $\alpha<2$, assuming that the temporal transitions are sparse.
By formulating the SlowVAE ELBO in the proposed framework we get

$$ $\begin{split}\mathcal{L}(&\mathbf{\theta,\phi,\beta,\lambda,o_{i},o_{i+1}})=\\ &\mathbb{E}_{q_{\theta}(\mathbf{z_{i},z_{i-1}|o_{i},o_{i-1}})}[\log p_{\phi}(\mathbf{o_{i},o_{i-1}|z_{i},z_{i-1}})]\\ &-\beta D_{KL}(q_{\theta}(\mathbf{z_{i-1}|o_{i-1}})||p(\mathbf{z_{i-1}}))\\ &-\gamma\mathbb{E}_{q_{\theta}(z_{i-1}|o_{i-1})}[D_{KL}(q_{\theta}(\mathbf{z}_{i}|\mathbf{o}_{i}))|p(\mathbf{z}_{i}|\mathbf{z}_{i-1})).\end{split}$ (7) $$

where $\beta=1$ and corresponds to the KL loss term in the variational autoencoder.
The parameter $\gamma$ is a weight for the slowness loss term which is the KL divergence between approximate posterior at the current time and a factorized generalized Laplace distribution describing the prior on the sparse temporal transitions $p(\mathbf{z}_{i}|\mathbf{z}_{i-1})$(^3^33Following the general notation from Eq. [4](#S3.E4) we will refer to this weight as $\lambda$ from here on.).
The expectation is taken over the posterior of the previous step.

Compared to SVAE, by taking the expectation over the slowness KL-term, the SlowVAE compares a Gaussian centered at $\mu_{i}$ to a Laplace distribution at $\mu_{i-1}$ while in SVAE both terms are Gaussian. Moreover, in SVAE the slowness KL-term considers both covariances $({\Sigma}_{i},{\Sigma}_{i-1})$ while in SlowVAE only ${\Sigma}_{i}$ is considered.

### 4.3 S-VAE

We now describe our proposed Slow Variational Autoencoder (S-VAE).
Considering two distinct yet sequential observations $\mathbf{o}_{i},\mathbf{o}_{j}\in D~{}|~{}i<j$, the difference of the corresponding latent representations is given by the approximate difference distribution,

$$ $q_{\theta}(\mathbf{z}_{j}-\mathbf{z}_{i}|\mathbf{o}_{j},\mathbf{o}_{i})=\mathcal{N}(\mathbf{\mu}_{j}-\mathbf{\mu}_{i},\Sigma_{j}+\Sigma_{i})\equiv q_{\theta}(\Delta\mathbf{z}|\mathbf{o}_{j},\mathbf{o}_{i}).$ (8) $$

To express the decaying similarity with growing temporal separation $\Delta t$, we assume a prior that the latent vector exhibits Brownian motion.
This allows the S-VAE to benefit from the closed form solution of the Brownian motion and to handle varying sampling intervals as opposed to the SlowVAE which does not incorporate the length of time interval in the formulation.
Moreover, all additive stochastic processes with stationary uncertainty approach Brownian motion in the limit.
Denoting by $Z_{i},Z_{j}$ two increments of the Brownian motion at times $i$ and $j$, respectively, the prior distribution $p(\Delta\mathbf{z})$ is given by

$$ $Z_{j}-Z_{i}=\sqrt{\Delta t}\cdot N\sim\mathcal{N}(0,\Delta tI)\equiv p(\Delta\mathbf{z}),$ (9) $$

where $N\sim\mathcal{N}(0,I)$.
This prior also encodes the time, resulting in observations further apart in time to have a prior distribution with a larger covariance.

The similarity of two observations can be computed as the Kullback-Leibler (KL) divergence between the approximate posterior and the prior distributions as

$$ $\mathcal{L}_{\text{slow}}(\mathbf{\tilde{o}}_{i},\mathbf{\tilde{o}}_{j})=D_{KL}(q_{\theta}(\Delta\mathbf{z}|\mathbf{o}_{j},\mathbf{o}_{i})||p(\Delta\mathbf{z})).$ (10) $$

$\Delta t$ can be considered a scaling factor for the variance of the Brownian motion.
Thus, when considering two consequent elements in the sequence, we want to constrain $\mathcal{L}_{\text{slow}}$ to be smaller than a certain bound.
We consider only pairs of consecutive elements because the bound (scaled according to $\Delta t)$ becomes weaker as the temporal distance between elements increases.
Using the slowness loss formulation form Eq. [4](#S3.E4) we arrive at the S-VAE optimization target

$$ $\begin{split}\mathcal{L}(&\mathbf{\theta,\phi,\beta,\lambda,o_{i},o_{i-1}})=\\ &\mathbb{E}_{q_{\theta}(\mathbf{z_{i},z_{i-1}|o_{i},o_{i-1}})}[\log p_{\phi}(\mathbf{o_{i},o_{i-1}|z_{i},z_{i-1}})]\\ &-\beta D_{KL}(q_{\theta}(\mathbf{z_{i}|o_{i}})||p(\mathbf{z}))\\ &-\lambda D_{KL}(q_{\theta}(\Delta\mathbf{z}|\mathbf{o}_{i},\mathbf{o}_{i-1})||p(\Delta\mathbf{z})).\end{split}$ (11) $$

where $\theta$ and $\phi$ parameterize encoder and decoder and the parameters $\beta$ and $\lambda$ are hyperparameters for the strength of the disentanglement and slowness regularization.
The reconstruction loss term describes how well $\mathbf{o}_{i}$ can be reconstructed given its latent representation $\mathbf{z}_{i}$.
The disentanglement term is the Kullback-Leibler Divergence between the latent representation $\mathbf{z}_{i}$ and unit Gaussian prior $p(\mathbf{z})$.

## 5 Empirical comparison

In this section, we compare the previously introduced slow autoencoding methods to the $\beta$-VAE with respect to their performance when learning downstream regression tasks.
Although the downstream task performance benefits of disentangled representations are still subject of discussion, we include the $\beta$-VAE as a baseline in our comparison as it is one of the most established unsupervised representation learning methods.
Lastly, we will investigate the influence of the latent bottleneck term of the $\beta$-VAE and the slowness enforced by $\lambda$ on the representations and the downstream task performance.

Although the use case is different, we compared the TD-VAE to the slow methods and the $\beta$-VAE and found that it is outperformed by all methods.
We did not include those results as we were not confident in their thoroughness.
The problem being that to our knowledge there is no official code repository for the TD-VAE and we were not able to reproduce the results on the more complex DeepMind Lab experiment with the given implementation instructions.

### 5.1 Experiment Setting

Three experiments have been conducted in which the goal was to learn downstream tasks in a semi-supervised way from abundant unlabeled video data and sparse labeled data.
The semi-supervised process consists of a two step approach, first an autoencoding model is trained to extract representation from the abundant unlabeled data in an unsupervised way.
Second, a downstream task is learned using the encoded latent representations from the pre-trained models while keeping the encoder network weights frozen.
The downstream tasks involve temporal tasks like estimating odometry or velocity of an object from consecutive frames.
The performance is evaluated by observing the downstream task performance for varying amounts of labeled data.

The three experiments, described in more detail in Appendix [A](#A1), were conducted in different domains, a simple synthetic dataset of a ball bouncing in a $2D$ arena, two agents playing the game of Pong and a reinforcement learning agent exploring a $3D$ world in the DeepMind Lab environment Beattie et al. (2016).

Figure: Figure 1: Images (top) and their reconstructions (bottom) obtained from the S-VAE. Two random samples of the full dataset for each of the three datasets.
Refer to caption: /html/2012.06279/assets/fig1.png

Figure [1](#S5.F1) shows random observations of all three environments and reconstructions generated using the S-VAE.

Figure: (a) Mean Squared Error Loss between true and predicted velocity vector vs. unseen test dataset size for the Ball experiment.
Refer to caption: /html/2012.06279/assets/x1.png

### 5.2 Evaluation of Downstream Task Performance

The downstream task performance is measured by computing the loss on a previously unseen labeled test set.
During this analysis we will look at two main aspects of the downstream performance.
First, the best case performance which refers to the downstream task performance given abundant labeled data.
Second, the data efficiency measured by the downstream task performance for sparse labeled data.

Figure: (a) $\beta$-VAE ($\lambda=0.0$).
Refer to caption: /html/2012.06279/assets/x4.png

Fig. [2](#S5.F2) shows the best performing model for each method as test loss given different amounts of labeled data.
We can see that in the ball and pong experiments the slow methods consistently outperform their $\beta$-VAE counterparts in both best case performance and data efficiency.
The $L2$ slowness loss method in the ball experiment achieved $75\%$ better best case performance (lower downstream task loss) and achieved similar performance as the $\beta$-VAE with only $2.73\%$ of the data showing significant benefits.
In the ball experiment, the best performing method used the $L2$ loss term and achieved $30\%$ better best case performance and performed as good as the $\beta$-VAE method with only $7.5\%$ of the data.
In the DeepMind Lab experiment all methods performed similarly across all dataset sizes. This is likely due to the hyperparameter only partially exploring the optimal hyperparameter region.
Some evidence for this hypothesis can be seen from Fig. [6(i)](#A3.F6.sf9) in Appendix [C](#A3) where the best performing methods are clustered in the bottom right corner hinting that smaller values for $\lambda$ and $\beta$ would likely have yielded better performance.
The mean performance of all models trained with a given slowness loss term shows $L1$ and $L2$ loss performed similar compared to each other and better than the SlowVAE or the SVAEin many cases in all three experiments.
This indicates that in our experiments the slow methods do not necessarily benefit from taking the covariances of the latent distributions into account.
The finding about the $L1$ and $L2$ methods is in line with the findings in Cadieu and
Olshausen (2012) claiming that there is no significant difference between $L1$ and $L2$ temporal loss terms.
More details and the average performance numbers can be found in Appendix [B](#A2).

### 5.3 Influence of Hyperparameters

Next we investigate how the hyperparameters $\lambda$ and $\beta$ influence the latent representations by visualizing the $2D$ latent space used in the ball experiment.
Fig. [3](#S5.F3) shows a scatter plot where the x- and y-axis are the ground truth position of the ball in the arena and the color represents the latent value.
For simplification we only show latent dimension $1$ for each autoencoder model.
From left to right the slowness regularization is increased from 0 0 (equivalent to $\beta$-VAE) to $1e-04$ (strongest regularization).
We can see that from left to right the continuity of the latent space increases.
For the beta-VAE abrupt changes of latent values create steep valleys in the latent space whereas for the S-VAE model with increasing slowness regularization the latent space becomes almost perfectly smooth from left to right ($\lambda=1e-04$).
The S-VAE method with $\lambda=1e-09$ and $\beta=1e-08$ shown in Fig. [3(b)](#S5.F3.sf2) is the overall best performing S-VAE method shown in Fig. [2(a)](#S5.F2.sf1).
This indicates that the strongest slowness regularization does not necessarily lead to better downstream task performance, however, having both slowness and disentanglement regularization performs better than only disentanglement.
Looking at Figures [6(a)](#A3.F6.sf1), [6(b)](#A3.F6.sf2), [6(e)](#A3.F6.sf5), [6(f)](#A3.F6.sf6) and [6(i)](#A3.F6.sf9), [6(j)](#A3.F6.sf10) in the Appendix [C](#A3), this effect is highlighted for the S-VAE for all three experiments, strongest regularization (highest $\lambda$ or $\beta$) does not lead to best performance.

To demonstrate the influence of the hyperparameters $\lambda$ and $\beta$ on the latent channels, we visualize the $20$ latent dimensions of the pong experiment for each model in Fig. [4](#S5.F4) and in more detail for the S-VAE in Fig. [6](#A3.F6).
Latent dimensions encoding no information have mean and variance close to zero.
Accordingly, a threshold on the variance for each latent distribution is selected to compute the number of used dimensions.
As expected, higher values for $\beta$ and therefore stronger regularization reduce the number of used latent dimensions.
On the same note it can be hypothesized that stronger slowness regularization leads to more pressure on the latent channels and therefore reduce the number of used dimensions.
However, the data highlighted in Fig. [4(a)](#S5.F4.sf1) and [4(c)](#S5.F4.sf3) and in more detail in Fig. [6(g)](#A3.F6.sf7) and [6(k)](#A3.F6.sf11) shows multiple configurations of higher $\lambda$ where the number of used dimensions increases, thus providing evidence that the hypothesis does not hold.
Further investigation is needed to explore this phenomenon.
Looking at Figures
[6(a)](#A3.F6.sf1), [6(b)](#A3.F6.sf2), [6(e)](#A3.F6.sf5) and [6(f)](#A3.F6.sf6), we observe no correlation between performance and used dimensions.

Although the qualitative continuity of the latent space and the number of dimensions used help us understand the effects of the hyperparameters $\beta$ and $\lambda$ on the latent space, it is not sufficient to reliably predict downstream task performance.

Figure: (a) Pong experiment, latent distributions with $\lambda=1e-04$ and $\beta$-VAE with $\beta=1e-04$.
Refer to caption: /html/2012.06279/assets/x8.png

Figure: (a) Ball, full dataset
Refer to caption: /html/2012.06279/assets/x12.png

## 6 Predicting Downstream Task Performance

As we have seen the qualitative latent space properties discussed above do not give the necessary predictive power to select an encoder model with good downstream task performance.
To predict general downstream task performance, we want to study each of the three components of the general formulation in Eq. ([4](#S3.E4)): reconstruction performance, disentanglement and slowness.

The extent to which disentanglement can predict downstream task performances has been investigated by Locatello et al. in Locatello et al. (2019b).
In their work the authors questioned the benefits that disentangled representations have for learning downstream tasks and criticised that currently all disentanglement metrics require ground truth labels.
Since we assume sparse labeled data and want to predict performance for any downstream tasks, we are not able to use existing disentanglement metrics.

The second option is to measure if a latent space exhibits the slowness properties and correlate this measure with downstream task performance.
We devised a metric that measures the length of a trajectory in latent space defined by $N$ encoded latent representations $z_{1},\ldots,z_{N}$ and compared it to the euclidean distance between $z_{1}$ and $z_{N}$.
According to the slowness principle and the qualitative analysis in Fig. [3](#S5.F3) we hypothesize that higher slowness regularization with a qualitatively smoother latent space has less jumps and therefore the trajectory in latent space is shorter and more similar to the euclidean distance.
Based on a preliminary study we could observe that higher slowness regularization indeed leads to less fragmented and shorter latent trajectories.
However, the metric does not correlate with downstream task performance.

Third, we can investigate if the generative capabilities of a model allow us to predict downstream task performance.
The core idea is that a model capable of generating ”realistic” images from representations sampled from the latent distribution encodes more useful information.
We use the FID which is usually used to evaluate the generative capabilities of Generative Adverserial Networks (GANs).
After training the autoencoder we draw random samples from the latent distributions and decode them into images.
Then we compute the FID between the training data distribution and the generated data distribution.
The results of this analysis on the three experiments in this paper are shown in Fig. [5](#S5.F5).
Furthermore, Figure [6](#A3.F6) is showing the FID for all $\lambda$, $\beta$ combinations for the S-VAE.
For both cases, abundant and sparse labeled training data during the downstream task, we can see a correlation between low FID and low downstream task loss.
In the ball experiment the best performing $L2$ loss model can clearly be identified by its low FID.
In the other experiments it is not as obvious to find a clearly superior method, but in all experiments the FID can be used to identify a small set of good performing VAE models.
In conclusion, the FID score is a good measure to predict downstream task performance of an autoencoding model and therefore greatly improves hyperparamter tuning speed saving valuable time.

## 7 Slow Autoencoding Methods from an Information Theoretic Point of View

Lastly we want to discuss the connection between the FID score as a predictive metric for downstream task performance of a model and slowness regularization.
Let us look at the $\beta$-VAE from the Information theoretic point of view.
We can write the $\beta$-VAE objective as $\max[I(z;y)-\beta I(o;z)]$ Burgess et al. (2018); Alemi et al. (2016) where $I(z;y)$ is the mutual information of the intermediate representation $z$ and the reconstruction task.
The subtracted term $\beta I(o;z)$, usually referred to as the latent bottleneck, effectively encourages the model to discard less relevant information present in the input $o$ by limiting the capacity of the latent information channels.
In our case such information would be properties of the input signal that are not relevant for reconstruction.
As observed in the previous analysis, increasing the latent bottleneck pressure by varying the parameter $\beta$ reduces the number of latent dimensions used.
As we have discussed in [5.3](#S5.SS3), the influence of the slowness regularization on the latent bottleneck pressure remains somewhat unclear.
We theorize that, similar to how the $\beta$-VAE discards less relevant information, slowness regularization adds a way for the model to include or discard information when learning temporal tasks.
This draws a parallel to the core idea of the slowness principle: Discourage the encoding of irrelevant quickly changing components of the input signal and encourage encoding of slowly changing features.

## 8 Conclusion

In this paper, we propose a general framework for applying the slowness principle to the state-of-the-art $\beta$-VAE.
Within this framework we compare existing methods of enforcing slowness such as $L1$ and $L2$ loss and the SlowVAE.
Furthermore, we introduce a new slowness regularization term based on a Brownian motion.
We find that slow methods outperform the baseline $\beta$-VAE with respect to downstream task data efficiency and performance for experiments in simple dynamic environments.
The results indicate that $L1,L2$ slowness loss terms perform as good as the more complex slowness loss terms taking uncertainty into account.
A qualitative analysis of the latent space structure illustrates the effects of the slowness loss on the latent space structure.
Lastly we find that the Fréchet Inception Distance is a good measure to predict downstream task performance by quantifying how much of the encoded information in the latent space is useful for reconstructing images.

An important open question is, how the properties of the latent space influences the optimization of downstream tasks.
The results indicate that a smooth latent space is not always the best for learning downstream tasks and that there is some trade-off between the slowness regularization and latent bottleneck pressure.

## References

- Alemi et al. [2016]
Alexander A Alemi, Ian Fischer, Joshua V Dillon, and Kevin Murphy.
Deep variational information bottleneck.
arXiv preprint arXiv:1612.00410, 2016.
- Barratt and Sharma [2018]
Shane Barratt and Rishi Sharma.
A note on the inception score.
arXiv preprint arXiv:1801.01973, 2018.
- Beattie et al. [2016]
Charles Beattie, Joel Z Leibo, Denis Teplyashin, Tom Ward, Marcus Wainwright,
Heinrich Küttler, Andrew Lefrancq, Simon Green, Víctor Valdés,
Amir Sadik, et al.
Deepmind lab.
arXiv preprint arXiv:1612.03801, 2016.
- Bengio and Bergstra [2009]
Yoshua Bengio and James S Bergstra.
Slow, decorrelated features for pretraining complex cell-like
networks.
In Advances in neural information processing systems, pages
99–107, 2009.
- Bengio et al. [2013]
Yoshua Bengio, Aaron Courville, and Pascal Vincent.
Representation learning: A review and new perspectives.
IEEE transactions on pattern analysis and machine intelligence,
35(8):1798–1828, 2013.
- Berkes and Wiskott [2005]
Pietro Berkes and Laurenz Wiskott.
Slow feature analysis yields a rich repertoire of complex cell
properties.
Journal of vision, 5(6):9–9, 2005.
- Burgess et al. [2018]
Christopher P Burgess, Irina Higgins, Arka Pal, Loic Matthey, Nick Watters,
Guillaume Desjardins, and Alexander Lerchner.
Understanding disentangling in $\beta$-vae.
arXiv preprint arXiv:1804.03599, 2018.
- Cadieu and
Olshausen [2012]
Charles F Cadieu and Bruno A Olshausen.
Learning intermediate-level representations of form and motion from
natural movies.
Neural computation, 24(4):827–866, 2012.
- Chen et al. [2020a]
Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton.
A simple framework for contrastive learning of visual
representations.
arXiv preprint arXiv:2002.05709, 2020.
- Chen et al. [2020b]
Ting Chen, Simon Kornblith, Kevin Swersky, Mohammad Norouzi, and Geoffrey
Hinton.
Big self-supervised models are strong semi-supervised learners.
arXiv preprint arXiv:2006.10029, 2020.
- Fortuin et al. [2020]
Vincent Fortuin, Dmitry Baranchuk, Gunnar Rätsch, and Stephan Mandt.
Gp-vae: Deep probabilistic time series imputation.
In International Conference on Artificial Intelligence and
Statistics, pages 1651–1661. PMLR, 2020.
- Franzius et al. [2007a]
Mathias Franzius, Henning Sprekeler, and Laurenz Wiskott.
Slowness and sparseness lead to place, head-direction, and
spatial-view cells.
PLoS Comput Biol, 3(8):e166, 2007.
- Franzius et al. [2007b]
Mathias Franzius, Roland Vollgraf, and Laurenz Wiskott.
From grids to places.
Journal of computational neuroscience, 22(3):297–299, 2007.
- Franzius et al. [2011]
Mathias Franzius, Niko Wilbert, and Laurenz Wiskott.
Invariant object recognition and pose estimation with slow feature
analysis.
Neural computation, 23(9):2289–2323, 2011.
- Gregor et al. [2019]
Karol Gregor, George Papamakarios, Frederic Besse, Lars Buesing, and Theophane
Weber.
Temporal difference variational auto-encoder.
In International Conference on Learning Representations, 2019.
- Heusel et al. [2017]
Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp
Hochreiter.
Gans trained by a two time-scale update rule converge to a local nash
equilibrium.
In Proceedings of the 31st International Conference on Neural
Information Processing Systems, pages 6629–6640, 2017.
- Higgins et al. [2017]
Irina Higgins, Loic Matthey, Arka Pal, Christopher Burgess, Xavier Glorot,
Matthew Botvinick, Shakir Mohamed, and Alexander Lerchner.
beta-vae: Learning basic visual concepts with a constrained
variational framework.
In International Conference on Learning Representations, 2017.
- Kingma and Ba [2014]
Diederik P Kingma and Jimmy Ba.
Adam: A method for stochastic optimization.
arXiv preprint arXiv:1412.6980, 2014.
- Kingma and Welling [2013]
Diederik P Kingma and Max Welling.
Auto-encoding variational bayes.
arXiv preprint arXiv:1312.6114, 2013.
- Klindt et al. [2020]
David Klindt, Lukas Schott, Yash Sharma, Ivan Ustyuzhaninov, Wieland Brendel,
Matthias Bethge, and Dylan Paiton.
Towards nonlinear disentanglement in natural data with temporal
sparse coding.
arXiv preprint arXiv:2007.10930, 2020.
- Laskin et al. [2020]
Michael Laskin, Aravind Srinivas, and Pieter Abbeel.
Curl: Contrastive unsupervised representations for reinforcement
learning.
In Proceedings of the 37th Annual International Conference on
Machine Learning (ICML), 2020.
- Locatello et al. [2019a]
Francesco Locatello, Gabriele Abbati, Tom Rainforth, Stefan Bauer, Bernhard
Schölkopf, and Olivier Bachem.
On the fairness of disentangled representations.
arXiv preprint arXiv:1905.13662, 2019.
- Locatello et al. [2019b]
Francesco Locatello, Stefan Bauer, Mario Lucic, Gunnar Raetsch, Sylvain Gelly,
Bernhard Schölkopf, and Olivier Bachem.
Challenging common assumptions in the unsupervised learning of
disentangled representations.
In international conference on machine learning, pages
4114–4124, 2019.
- Locatello et al. [2020]
Francesco Locatello, Ben Poole, Gunnar Rätsch, Bernhard Schölkopf,
Olivier Bachem, and Michael Tschannen.
Weakly-supervised disentanglement without compromises.
In International Conference on Machine Learning, pages
6348–6359. PMLR, 2020.
- Mobahi et al. [2009]
Hossein Mobahi, Ronan Collobert, and Jason Weston.
Deep learning from temporal coherence in video.
In Proceedings of the 26th Annual International Conference on
Machine Learning, pages 737–744, 2009.
- Peters et al. [2017]
Jonas Peters, Dominik Janzing, and Bernhard Schölkopf.
Elements of causal inference: foundations and learning
algorithms.
The MIT Press, 2017.
- Salimans et al. [2016]
Tim Salimans, Ian J Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford,
and Xi Chen.
Improved techniques for training gans.
In NIPS, 2016.
- Schölkopf et al. [2012]
Bernhard Schölkopf, Dominik Janzing, Jonas Peters, Eleni Sgouritsa, Kun
Zhang, and Joris Mooij.
On causal and anticausal learning.
arXiv preprint arXiv:1206.6471, 2012.
- Sermanet et al. [2018]
Pierre Sermanet, Corey Lynch, Yevgen Chebotar, Jasmine Hsu, Eric Jang, Stefan
Schaal, Sergey Levine, and Google Brain.
Time-contrastive networks: Self-supervised learning from video.
In 2018 IEEE International Conference on Robotics and Automation
(ICRA), pages 1134–1141. IEEE, 2018.
- Wiskott and Sejnowski [2002]
Laurenz Wiskott and Terrence J Sejnowski.
Slow feature analysis: Unsupervised learning of invariances.
Neural computation, 14(4):715–770, 2002.
- Zou et al. [2012]
Will Zou, Shenghuo Zhu, Kai Yu, and Andrew Ng.
Deep learning of invariant features via simulated fixations in video.
Advances in neural information processing systems,
25:3203–3211, 2012.

## Appendix A Experiment Details

All experiments have been implemented in the same VAE framework.
We used the Adam optimizer as introduced in Kingma and Ba [2014].
The reconstruction loss is computed using the binary cross entropy between predicted and true images.

### A.1 Ball Experiment

The goal in this task is to predict the velocity of the ball from two consecutive frames.
We generated sequences of 20 frames of a ball bouncing in a $100\times 100$ uni-color 2D environment.
For each sequence the ball is placed in a random position in the environment and initialized with a random direction and random (within some limits) velocity vector.
Upon reaching the border of the environment, the velocity vector of the ball is flipped to mimic the principle ”incident angle equals emergence angle” while keeping the balls velocity constant.
The environment performs 20 update steps by applying the velocity vector to the ball and stores the frames and the ball’s x/y-velocity.
Overall 10000 labeled sequences consisting of 20 datapoints each were generated.

During the unsupervised representation learning step, we use the full dataset without labels to train multiple models of all methods by varying the hyperparameters $\beta$ and $\lambda$.
The training consisted of 150 epochs with a batch size of 16 and the learning rate $1e-03$.
The encoder consists of 3 fully connected layers of size 100x100, 300 and 2 (latent dimensions). Mean and log-variance of the latent distributions is computed using two more layers with size 2.
The decoder is a mirrored version of the encoder without the two mean and log-variance layers.

In the supervised downstream task we use subsets of ($1,1/2,1/4,\ldots,1/128$) of the full labeled dataset with their labels to train the downstream task of predicting the ball velocity from two consecutive frames.
The downstream task is trained by freezing the encoder networks trained in the previous step and feeding the latent representations into a fully connected neural network with $2$ layers of $50$ hidden nodes.
The downstream loss is computed as the MSE between true and predicted velocity vector.
Overall 140 models were trained.

### A.2 Pong experiment

The goal was to predict the actions that both agents are taking from two consecutive frames.
A dataset consisting of 5000 sequences with 20 data points each was collected by observing two agents play the game of Pong against each other.
Sequences that contained reset events of the environment, for example when a game was won/lost were discarded during dataset generation to obtain uninterrupted sequences..
The training of this experiment was conducted similar to the ball experiment, obtaining two consecutive latent representations, concatenating them and obtaining the predicted action probabilities.
Subsets of ($1,1/2,1/4,\ldots,1/64$) of the full labeled dataset with their labels were used to train the downstream task.
The same 2 layer, 50 nodes downstream neural network structure was used.
The loss is computed as the Binary Cross Entropy loss between the predicted and one-hot encoded action probabilities.
The encoder network consists of 4 convolutional layers with batch normalization followed by 2 fully connected layers with 1024 and 512 hidden nodes.
The decoder network mirrors the encoder network.
In this experiment a batch size of 32 was used and the training was performed for 250 epochs.

### A.3 Deepmind Lab Dataset

In this experiment the goal is to learn the $6$-DOF visual odometry of an agent exploring a DeepMind Lab environment Beattie et al. [2016].
A dataset of 19000 sequences with 20 data points each was generated using a random walker exploring the environment.
Subsets of ($1,1/2,1/4,\ldots,1/32$) of the full labeled dataset with their labels were used to train the downstream task.
The same 2 layer, 50 nodes downstream neural network structure was used.
The loss is computed as the MSE loss between the predicted and true $6D$ motion vector.
The encoder network consists of 6 convolutional layers with 1 intermediate residual block for each layer followed by a full connected layer with 8192 hidden nodes.
The decoder network mirrors the encoder network.

## Appendix B Performance Result Details

The mean downstream performance for all trained models on the full and $\frac{1}{4}$ downstream dataset is summarized in Appendix [B](#A2).
In the tables [1](#A2.T1) to [3](#A2.T3) we can see that in most cases with abundant data the $L2$ slowness loss term leads to the best or comparable performance to the S-VAE and SlowVAE.
In the pong experiment, the S-VAE and SlowVAE performed both better compared to the $L1$ and $L2$ methods when labeled data was sparse.
Overall there seems to be little difference between the simpler $L1$ and $L2$ methods and also between S-VAE and SlowVAE which take the covariances of the latent distributions into account.

**Table 1: Mean and STD downstream task loss for all slow methods in the ball experiment.The best performing methods are highlighted in blue.**
|  | Abundant data |  | Sparse data |  |
| --- | --- | --- | --- | --- |
| Method | Mean | STD | Mean | STD |
| S-VAE | 1.259 | 0.447 | 1.464 | 0.408 |
| SlowVAE | 1.095 | 0.219 | 1.339 | 0.219 |
| $L1$ | 1.310 | 0.441 | 1.525 | 0.422 |
| $L2$ | 1.153 | 0.430 | 1.292 | 0.438 |

**Table 2: Mean and STD downstream task loss for all slow methods in the pong experiment.The best performing methods are highlighted in blue.**
|  | Abundant data |  | Sparse data |  |
| --- | --- | --- | --- | --- |
| Method | Mean | STD | Mean | STD |
| S-VAE | 0.376 | 0.139 | 0.657 | 0.440 |
| SlowVAE | 0.373 | 0.126 | 0.650 | 0.427 |
| $L1$ | 0.332 | 0.078 | 1.034 | 0.752 |
| $L2$ | 0.326 | 0.079 | 1.040 | 0.707 |

**Table 3: Mean and STD downstream task loss for all slow methods in the DeepMind Lab experiment. The best performing methods are highlighted in blue.**
|  | Abundant data |  | Sparse data |  |
| --- | --- | --- | --- | --- |
| Method | Mean | STD | Mean | STD |
| S-VAE | 0.031 | 0.021 | 0.035 | 0.022 |
| SlowVAE | 0.028 | 0.014 | 0.032 | 0.014 |
| L1 | 0.026 | 0.020 | 0.033 | 0.015 |
| $L2$ | 0.021 | 0.010 | 0.029 | 0.007 |

## Appendix C Analysis

In Fig. [6](#A3.F6) we summarize the results of all three experiments (from top to bottom: Ball experiment, Pong experiment, DeepMind Lab experiment) for the S-VAE.
From left to right we plot different metrics for all models trained (different combinations of hyperparameters $\lambda$ and $\beta$:
First, the downstream task performance for the full and partial labeled datasets.
Then the number of latent channels used by each model and the FID.

Figure: (a) Ball experiment, downstream task performance full labeled data, method SVA.
Refer to caption: /html/2012.06279/assets/x18.png