Before analyzing how different approaches represent the temporal structure of language, we demonstrate that Temporal Feature Analysis performs on par with SAEs on standard metrics such as reconstruction error.
Specifically, we train a Temporal Feature Analyzer and standard SAEs (ReLU, TopK, BatchTopK) on $1$B token activations extracted from Gemma-2-2B (team2024gemma) from the Pile-Uncopyrighted dataset (monology2021pile-uncopyrighted).
We also analyze a baseline of the prediction only module from Temporal Feature Analysis, reported as ‘Pred. only’, which can be expected to underperform since predicting the next-token representation is likely to be more difficult than reconstructing it.
Results are provided in Tab. [1](https://arxiv.org/html/2511.01836v3#S5.T1), [2](https://arxiv.org/html/2511.01836v3#S5.T2) and show competitive performance between all protocols, except Pred. only.
One can also assess which part of a fully trained Temporal Feature Analyzer is more salient in defining its performance, i.e., does the estimated predictive part $\hat{\bm{x}}_{p,t}$ contribute more to the reconstruction $\hat{\bm{x}}$ or does the estimated novel part $\hat{\bm{x}}_{n,t}$.
Results are reported in Tab. [3](https://arxiv.org/html/2511.01836v3#S5.T3).
We see that the error vectors, i.e., $\bm{x}-\hat{\bm{x}}_{p}$ and $\bm{x}-\hat{\bm{x}}_{n}$, are approximately orthogonal, suggesting the modules computing predictive and novel codes capture separate bits of information from the input signal.
Furthermore, we find that a bulk of the reconstructed signal $\hat{\bm{x}}_{t}$ (in the sense of norm) is captured by the predictive code—in fact, the percentage contribution of the predictive code is $\sim$80%, in line with numbers observed in Fig. [2](https://arxiv.org/html/2511.01836v3#S3.F2)d.
However, analyzing the reconstruction performance, we see the predictive component primarily contributes to achieving a good NMSE, while the novel component is more responsible for explaining the input signal variance.
These results align with the generative model assumed in Temporal Feature Analysis (Eq. [3](https://arxiv.org/html/2511.01836v3#S5.E3)).
Specifically, NMSE captures the average reconstruction, and hence a slower moving, contextual signal can expect to dominate its calculation; in fact, we see the attention layer used for defining the prediction module produces block structured attention maps related to sub-event chunks (see Fig. [13](https://arxiv.org/html/2511.01836v3#S8.F13)), suggesting more granular temporal dynamics are being captured by predictive part (we elaborate on this last point in the next section).
Meanwhile, variance assesses changes per dimension and timestamp in the signal, which better matches the inductive bias imposed on the novel part.

\begin{table}
\captionsetup{labelformat=empty}
\caption{Table 1: Temporal Feature Analysis and SAEs achieves similar NMSE across domains (Simple Stories, Webtext, Code).}
\begin{tabular}{l|ccc|c|c}
\hline & ReLU & TopK & BTopK & Pred. Only & Temporal \\
\hline Story & 0.20 & 0.155 & 0.152 & 0.34 & 0.139 \\
Web & 0.19 & 0.144 & 0.139 & 0.36 & 0.139 \\
Code & 0.20 & 0.154 & 0.149 & 0.38 & 0.152 \\
\hline
\end{tabular}
\end{table}

\begin{table}
\captionsetup{labelformat=empty}
\caption{Table 2: Temporal Feature Analysis explains similar amount of signal variance as SAEs.}
\begin{tabular}{l|ccc|c|c}
\hline & ReLU & TopK & BTopK & Pred. Only & Temporal \\
\hline Story & 0.60 & 0.71 & 0.72 & 0.29 & 0.73 \\
Web & 0.69 & 0.78 & 0.79 & 0.40 & 0.79 \\
Code & 0.65 & 0.75 & 0.75 & 0.33 & 0.75 \\
\hline
\end{tabular}
\end{table}

\begin{table}
\captionsetup{labelformat=empty}
\caption{Table 3: Predictive and novel codes explain different parts of the input. See main text for details.}
\begin{tabular}{l|l|ll|ll|ll}
\hline & $\mid\left\langle\boldsymbol{x}_p, \boldsymbol{x}_n\right\rangle$ & \multicolumn{2}{|c|}{$\%$ Norm } & \multicolumn{2}{c|}{ NMSE } & \multicolumn{2}{c}{ Var. Expl. } \\
& & \multicolumn{2}{c}{$\mid$ Pred. } & Novel $\mid$ & \multicolumn{2}{c}{ Pred. } & Novel \\
& & \multicolumn{2}{c}{ Pred. } & Novel \\
\hline Story & -0.02 & 76.2 & 23.5 & 0.53 & 4.03 & 0.11 & 0.64 \\
Web & -0.02 & 80.5 & 19.5 & 0.49 & 4.28 & 0.17 & 0.66 \\
Code & -0.02 & 74.2 & 26.0 & 0.57 & 3.84 & 0.14 & 0.65 \\
\hline
\end{tabular}
\end{table}