# Parameter counts and inference cost per architecture

Counted by instantiating each registered class at the paper's own hyperparameters. FLOPs analytic, 2 per multiply-accumulate; decode quoted for a sparse implementation; per-token cost at stride T (windows tiling the sequence, which the paper's one-code-per-tile evaluation implies).

## probing — google/gemma-2-2b-it (d_model=2304, layer 13, 2.61B params)

Training: 20,000 steps x 4,096 = 81.9M tokens seen.

| architecture | d_sae | k | T | parameters | % of subject | inference FLOPs/token | % of subject forward | training FLOPs |
|---|---|---|---|---|---|---|---|---|
| topk_sae | 18432 | 20 | 1 | 85.0M | 3.25% | 85.03M | 1.629% | 20.90P |
| tsae | 16384 | 20 | 1 | 75.5M | 2.89% | 75.59M | 1.448% | 18.58P |
| txc_base | 18432 | 20 | 5 | 424.7M | 16.27% | 85.03M | 1.629% | 20.90P |
| mlc | 18432 | 20 | 1 | 424.7M | 16.27% | 425.13M | 8.144% | 104.48P |

## backtracking — deepseek-ai/DeepSeek-R1-Distill-Llama-8B (d_model=4096, layer 10, 8.03B params)

Training: 25,000 steps x 1,024 = 25.6M tokens seen.

| architecture | d_sae | k | T | parameters | % of subject | inference FLOPs/token | % of subject forward | training FLOPs |
|---|---|---|---|---|---|---|---|---|
| topk_sae | 32768 | 20 | 1 | 268.5M | 3.34% | 268.60M | 1.672% | 20.63P |
| tsae | 16384 | 20 | 1 | 134.2M | 1.67% | 134.38M | 0.837% | 10.32P |
| txc_base | 32768 | 20 | 5 | 1342.2M | 16.72% | 268.60M | 1.672% | 20.63P |
| mlc | 18432 | 20 | 1 | 755.0M | 9.40% | 755.79M | 4.706% | 58.04P |

## em — Qwen/Qwen2.5-7B-Instruct (d_model=3584, layer 15, 7.62B params)

Training: 20,000 steps x 4,096 = 81.9M tokens seen.

| architecture | d_sae | k | T | parameters | % of subject | inference FLOPs/token | % of subject forward | training FLOPs |
|---|---|---|---|---|---|---|---|---|
| topk_sae | 18432 | 20 | 1 | 132.1M | 1.73% | 132.26M | 0.868% | 32.51P |
| tsae | 16384 | 20 | 1 | 117.5M | 1.54% | 117.58M | 0.772% | 28.90P |
| txc_base | 32768 | 25 | 5 | 1174.5M | 15.41% | 235.06M | 1.542% | 57.77P |
| mlc | 18432 | 20 | 1 | 660.6M | 8.67% | 661.32M | 4.339% | 162.53P |
