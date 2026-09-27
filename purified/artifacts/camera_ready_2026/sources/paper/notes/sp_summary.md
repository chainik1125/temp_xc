# Sparse Probing — Narrative Summary

## Context

The §6.1 Sparse Probing subsection is the *macroscopic* benchmark complement to the *zoomed-in* causal case studies (backtracking §6.2, EM §6.3). This file lays out why sparse probing is in the paper, what story we are telling with it, and how it connects to the rest of the narrative. We anchor our methodology in Kantamneni, Engels, Rajamanoharan, Tegmark & Nanda (ICML 2025) and our temporal-features hook in Bhalla et al.'s T-SAE paper.

---

## The three narrative points

### 1. Sparse probing is the cleanest measurement axis we have for dictionary-learning architectures.

Following \citet{kantamneni2025are}, sparse probing asks: given an architecture's features, can you build a $k$-feature linear classifier (with $k$ small) for a binary concept, and how well does it do? Three properties make this the right macro-benchmark for temporal architectures:

- **Judge-free.** No LLM rater, no rubric, no $\kappa$ deferral — just AUC.
- **Calibration-free.** No magnitude grid, no coherence floor, no per-feature mining procedure that has to be matched across architectures.
- **Comparable.** All architectures see the same activation cache, the same concept panel, the same $k$-grid. Differences are attributable to the architecture, not the eval.

**Why this matters.** It's the only axis on which we can confidently claim *"TXC matches/beats baseline X under matched conditions"* without reviewers having to trust a judge prompt or a steering protocol. The case studies give us depth; sparse probing gives us breadth + cleanliness.

### 2. Temporal architectures should gain on the *semantic/contextual* slice and tie on the *syntactic* slice.

This is the load-bearing temporal-features claim of the section, and it is *not* ours — it is Bhalla et al.'s. Their argument:

> "Semantic content has long-range dependencies and tends to be smooth over a sequence, whereas syntactic information is much more local."

A per-token SAE evaluates each token in isolation, which is fine for locally-defined (syntactic) concepts but bad for concepts that *live* across a window (semantic / contextual). T-SAE's empirical demonstration: on Pythia-160m and Gemma2-2b across MMLU / Wikipedia / FineFineWeb at $k \in \{1, 5, 10, 20\}$, *"T-SAEs significantly outperform baseline SAEs for semantics and context"* while staying competitive on syntactic.

**Why this matters.** It gives us a *falsifiable* prediction for the TXC: matched expansion + sparsity, the TXC should gain on semantic-tagged probing tasks and tie on syntactic ones. This turns sparse probing from a vague "we measure on X" benchmark into a directional test of whether the TXC's window-shared latent inherits the same inductive-bias advantage that T-SAE's temporal smoothness exploits.

### 3. Strong baselines are the point. SAE features do not automatically win.

Kantamneni et al. is explicit: SAEs do *not* systematically beat residual-stream linear probes or neuron probes across their four regimes (data scarcity, class imbalance, label noise, covariate shift). They flag multi-token probes specifically as a regime where SAEs *initially* appear promising but where strong baselines catch up.

**Why this matters.** Two reasons:

- **Methodological.** Our sparse-probing comparison must include the strong baselines Kantamneni identified — not just other SAEs/T-SAE/MLC, but the residual-stream linear probe and (where applicable) the neuron probe. Reviewers trained on Kantamneni will look for these immediately.
- **Narrative.** It re-affirms the framing memo from the EM section: case-study subsections are benchmark applications, not mini-papers. A sparse-probing tie on syntactic and a small win on semantic is *exactly* the right shape of result for the TXC's headline framing ("ties or wins, simplest version"). Overclaiming here would invite the same skepticism Kantamneni applied to the broader SAE literature.

---

## How this maps onto §6.1 prose

A reasonable four-paragraph structure mirroring EM §6.3:

1. **Motivation.** What sparse probing is (Kantamneni protocol), why it's a good macroscopic axis, and the temporal hook (Bhalla's semantic-vs-syntactic argument).
2. **Method.** Datasets (probably Gemma-2 2B on the SAEBench multi-task panel; per `data_index.md` §6.1 we have c3 results on `gemma_2_2b_{base,it}_l11to15_fineweb_24k128`), $k$-grid, baselines (TXC-base, TXC-pro, T-SAE, MLC, residual-stream probe).
3. **Results.** Refer to $\cref{fig:sparse_probing}$ subfigs Gemma + Gemma-IT; report AUC at the $k$-grid; state where TXC wins / ties / loses.
4. **Takeaway.** One sentence — "TXC inherits T-SAE's semantic-slice advantage" or "TXC ties baselines, consistent with our framework's prediction at $T = 5$" depending on what the data actually shows.

## References (already in `refs.bib`)

- `kantamneni2025are` — Kantamneni, Engels, Rajamanoharan, Tegmark & Nanda. *Are Sparse Autoencoders Useful? A Case Study in Sparse Probing.* ICML 2025.
- `bhalla2025tsae` — Bhalla, Oesterling, Verdun, Lakkaraju, Calmon. *Temporal Sparse Autoencoders.* arXiv 2511.05541.
