# GUM personal-pronoun antecedent-distance decoding

This is a semi-synthetic temporal probe over native GUM coreference edges. For
each direct `ana` or `coref` edge whose destination is one UD personal-pronoun
token, the three-way label is the exact Llama-token distance from the final
subtoken of the antecedent mention to the final subtoken of the pronoun:
`d ∈ {2, 3, 4}`. The five layer-10 `resid_post` states end at the pronoun.

The endpoint matters scientifically. This is an ordered contextual
disambiguation experiment, not a prediction of coreference before a pronoun
appears. Shuffle and reverse controls transform only the four history positions
and leave the pronoun endpoint fixed. The required 2,016-event sensitivity is
balanced within each pronoun form across all three labels, so endpoint identity
cannot by itself explain a TXC advantage.

The source is the official GUM release at commit
`22fdf87f9c71c96bcc771461d06e689b1f90020d`, using its native WebAnno
coreference TSV, matching UD CoNLL-U, genre, and split annotations. The subject
and tokenizer are pinned to `NousResearch/Meta-Llama-3.1-8B` revision
`1f47e50cdbe801ad8a5174156ec3a0655108fb9f`. See
[Zeldes (2017)](https://doi.org/10.1007/s10579-016-9343-x) and the
[GUM annotation guidelines](https://wiki.gucorpling.org/gum/entities).

## Locked cohorts

- Primary: 2,638 events from 269 documents; class counts 880 / 938 / 820.
- Required same-pronoun/label-balanced sensitivity: 2,016 events.
- Task-semantic SHA-256, binding the prefix/window, native edge, model-token
  endpoints, target, and sensitivity membership:
  `c08fddc183eccbf72667ef3780a8bf9a2ec9be25729f9efbc1f85bf5e740c7da`.
- Canonical balanced-row SHA-256:
  `9f7e0b535b7179d68c3191a23c6c4ea52cf1b3a196a60930c3639b20a4d58113`.
- CPU opportunity audit, not an inferential result: ordered lexical positions
  reached log loss 0.418 and balanced accuracy 0.854, while endpoint-only was
  1.063 / 0.403 and fixed history reversal was 2.010 / 0.216. The descriptive
  best-offset comparator was selected post hoc and is not part of the gate.

The cohort preserves source document, native split and genre, relation type,
pronoun form, UD feature bundle, source/target row and entity IDs, model spans,
and audit-only source entity type/information status. The latter two never
affect eligibility.

## Frozen-dictionary gate

The ordered TXC is compared to:

1. The same ordered-TXC probe applied without refitting to a fixed,
   per-event shuffled history.
2. The same ordered-TXC probe applied without refitting to reversed history.
3. Positional SAE codes over all five positions.
4. An order-invariant max bag over the four history SAE codes concatenated
   with the anchored endpoint SAE code.
5. Endpoint-only SAE codes.

Every probe uses the same five document-grouped outer folds and the same sparse
feature budget. Feature selection, scaling, and fitting use only training
documents. The primary statistic is equal-document multiclass log loss with a
paired 2,000-draw document bootstrap. In each bootstrap draw, the best of the
three SAE baselines is reselected, which is conservative for the SAE
competitor.

The preregistered gate passes only if both the full and balanced cohorts show
at least `0.02` lower equal-document log loss for ordered TXC than fixed
shuffle, fixed reverse, and the strongest SAE, and all three paired 95%
bootstrap intervals have lower bounds above zero.

## Reviewed launch command

Do not launch until the protocol and GPU allocation have been approved. On the
four-A40 pod, the intended command is:

```bash
cd /workspace/txc-neurips-aniket
TXC_RUNPOD_ROOT=/workspace/txc-neurips-aniket \
GUM_PRONOUN_GPU=1 \
bash purified/experiments/gum_pronoun_distance/launch_tmux.sh
```

The session is `gum-pronoun-distance-t5`; the log is
`purified/logs/gum_pronoun_distance/gpu1.log`. The worker checks that the
current branch is exactly `neurips-aniket`, installs the exit trap before
preflight, requires 26 GiB free by default, sparsely checks out only 54 MB of
GUM annotation files, rebuilds and validates the exact cohort, resumes eleven
activation shards, and then writes JSON, held-out predictions, Markdown, PDF,
and a 300-DPI PNG. Activation shards and sparse-code files have checksum-bound
sidecars and complete manifests; stale or orphaned cache artifacts fail closed.

Worst-case incremental storage is 16.06 GB for the pinned Llama weights,
3.76 GB for both submitted dictionaries, about 0.11 GB for activation shards,
0.06 GB for the sparse GUM checkout, 0.01 GB for tokenizer/cohort files, and
well under 0.1 GB for sparse codes and reports. The model and dictionaries use
shared cache/checkpoint paths, so their incremental cost is zero if another
job has already downloaded the exact revisions.
