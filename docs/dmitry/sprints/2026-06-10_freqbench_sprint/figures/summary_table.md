---
author: Claude (10h unsupervised sprint)
date: 2026-06-10
tags:
  - results
---

| task | arch | H | FVU | linear acc | MLP acc | shuffled lin | S_temp(lin) |
|---|---|---|---|---|---|---|---|
| ac_sign | conv | 128 | 0.326 | 0.999 | 1.000 | 0.501 | 1.00 |
| ac_sign | dcac | 128 | 0.432 | 0.904 | 1.000 | 0.501 | 0.81 |
| ac_sign | multiband | 128 | 0.424 | 1.000 | 1.000 | 0.504 | 1.00 |
| ac_sign | token_sae | 128 | 0.344 | 0.501 | 1.000 | 0.498 | 0.00 |
| ac_sign | txc | 128 | 0.386 | 0.903 | 1.000 | 0.501 | 0.81 |
| dc | conv | 64 | 0.172 | 0.989 | 0.992 | 0.989 | 0.98 |
| dc | dcac | 64 | 0.575 | 0.989 | 0.989 | 0.988 | 0.98 |
| dc | multiband | 64 | 0.496 | 0.989 | 0.989 | 0.988 | 0.98 |
| dc | token_sae | 64 | 0.162 | 0.987 | 0.992 | 0.987 | 0.97 |
| dc | txc | 64 | 0.421 | 0.990 | 0.990 | 0.989 | 0.98 |
| multifreq | conv | 64 | 0.835 | 0.223 | 1.000 | 0.112 | 0.14 |
| multifreq | conv | 256 | 0.785 | 0.117 | 1.000 | 0.103 | 0.02 |
| multifreq | conv | 2048 | 0.753 | 0.137 | 0.212 | 0.105 | 0.04 |
| multifreq | dcac | 64 | 0.955 | 0.184 | 0.661 | 0.169 | 0.09 |
| multifreq | dcac | 256 | 0.905 | 0.293 | 0.729 | 0.234 | 0.21 |
| multifreq | dcac | 2048 | 0.787 | 0.982 | 0.997 | 0.250 | 0.98 |
| multifreq | multiband | 64 | 0.956 | 0.200 | 0.632 | 0.158 | 0.11 |
| multifreq | multiband | 256 | 0.911 | 0.397 | 0.703 | 0.242 | 0.33 |
| multifreq | multiband | 2048 | 0.796 | 1.000 | 1.000 | 0.247 | 1.00 |
| multifreq | token_sae | 64 | 0.845 | 0.199 | 0.999 | 0.116 | 0.11 |
| multifreq | token_sae | 256 | 0.786 | 0.114 | 1.000 | 0.107 | 0.02 |
| multifreq | token_sae | 2048 | 0.760 | 0.114 | 0.235 | 0.101 | 0.02 |
| multifreq | txc | 64 | 0.952 | 0.121 | 0.622 | 0.136 | 0.02 |
| multifreq | txc | 256 | 0.896 | 0.170 | 0.883 | 0.164 | 0.08 |
| multifreq | txc | 2048 | 0.747 | 0.993 | 0.998 | 0.206 | 0.99 |
| multifreq_circle | conv | 64 | 0.158 | 0.373 | 0.980 | 0.180 | 0.31 |
| multifreq_circle | conv | 256 | 0.156 | 0.357 | 0.979 | 0.172 | 0.29 |
| multifreq_circle | conv | 2048 | 0.153 | 0.331 | 0.976 | 0.166 | 0.26 |
| multifreq_circle | conv7 | 256 | 0.145 | 0.478 | 0.926 | 0.875 | 0.42 |
| multifreq_circle | dcac | 64 | 0.260 | 0.898 | 0.993 | 0.425 | 0.89 |
| multifreq_circle | dcac | 256 | 0.186 | 0.969 | 0.984 | 0.423 | 0.97 |
| multifreq_circle | dcac | 2048 | 0.146 | 0.906 | 0.938 | 0.339 | 0.90 |
| multifreq_circle | multiband | 64 | 0.234 | 0.944 | 0.992 | 0.410 | 0.95 |
| multifreq_circle | multiband | 256 | 0.141 | 0.966 | 0.981 | 0.406 | 0.97 |
| multifreq_circle | multiband | 2048 | 0.117 | 0.923 | 0.951 | 0.362 | 0.92 |
| multifreq_circle | token_sae | 64 | 0.173 | 0.133 | 0.953 | 0.114 | 0.04 |
| multifreq_circle | token_sae | 256 | 0.136 | 0.116 | 0.953 | 0.101 | 0.02 |
| multifreq_circle | token_sae | 2048 | 0.147 | 0.119 | 0.861 | 0.111 | 0.02 |
| multifreq_circle | txc | 64 | 0.212 | 0.862 | 0.993 | 0.305 | 0.85 |
| multifreq_circle | txc | 256 | 0.102 | 0.953 | 0.968 | 0.345 | 0.95 |
| multifreq_circle | txc | 2048 | 0.097 | 0.841 | 0.911 | 0.277 | 0.83 |
