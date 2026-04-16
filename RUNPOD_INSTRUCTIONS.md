# RunPod Sweep Instructions

## What to do

Run a 36-run NLP sweep comparing TFA-pos vs TXCDR vs Stacked SAE on real LLM activations (Gemma-2-2B-IT + DeepSeek-R1-8B).

## Setup

The uv venv lives at `/workspace/temp_xc/.venv` and the HF cache at `/workspace/hf_cache` — both on RunPod's persistent `/workspace` volume, so they survive pod stop/start cycles. First session creates them; later sessions just re-activate.

```bash
cd /workspace
[ -d temp_xc ] || git clone https://github.com/chainik1125/temp_xc.git
cd temp_xc && git checkout han && git pull

# Install uv if the pod image doesn't ship it
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh

# Persist HF cache and auth on /workspace (add to ~/.bashrc so new shells pick it up)
export HF_HOME=/workspace/hf_cache
grep -q 'HF_HOME=/workspace/hf_cache' ~/.bashrc || echo 'export HF_HOME=/workspace/hf_cache' >> ~/.bashrc

# Create .venv on the persistent volume once; reuse on every later session
if [ ! -x .venv/bin/python ]; then
    uv venv .venv --python 3.12
    source .venv/bin/activate
    uv pip install torch transformers datasets huggingface_hub tqdm plotly kaleido scipy scikit-learn numpy
else
    source .venv/bin/activate
fi

# Only needed the first time — token is stored under $HF_HOME and persists
huggingface-cli whoami >/dev/null 2>&1 || huggingface-cli login
```

Verify:
```bash
PYTHONPATH=/workspace/temp_xc python -c "
import torch; print(f'GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.0f} GB')
from src.bench.model_registry import list_models; print('Models:', list_models())
"
```

### Reconnecting to an existing pod

```bash
cd /workspace/temp_xc
source .venv/bin/activate
export HF_HOME=/workspace/hf_cache   # already in ~/.bashrc after first setup
```

### GitHub push access (persistent across pod sessions)

The token lives at `/workspace/.github-token` (persistent volume) and a per-repo credential helper reads it at push time. Both survive pod stop/start, so `git push origin han` works in every new session with no re-auth.

First-time setup (once per persistent volume):

```bash
# 1) Write the PAT to the persistent volume. Use printf (no trailing newline).
printf '%s' 'ghp_YOUR_CLASSIC_PAT' > /workspace/.github-token
chmod 600 /workspace/.github-token   # NB: RunPod volume may ignore chmod; pod is single-tenant

# 2) Wire it into the repo's git config (once per clone of temp_xc)
cd /workspace/temp_xc
git config --local credential.helper \
    '!f() { echo username=x-access-token; printf "password=%s\n" "$(cat /workspace/.github-token)"; }; f'

# 3) Verify
git push --dry-run origin han
```

Token type: use a **classic** PAT with the `repo` scope. Fine-grained PATs must also list `chainik1125/temp_xc` under "selected repositories" — forgetting this is the #1 reason pushes 403 even when you're a collaborator.

Rotation: overwrite `/workspace/.github-token` in place; no other changes needed.

## Run the sweep

```bash
mkdir -p logs
TQDM_DISABLE=1 nohup bash scripts/run_nlp_sweep_16h.sh > logs/nlp_sweep.log 2>&1 &
tail -f logs/nlp_sweep.log
```

The script is **resumable** — if interrupted, restart the same command and it skips completed runs.

## What the script does (in order)

1. **Cache Gemma-2-2B-IT activations** (~2h): 24K seqs × 128 tokens, layers 13+25, FineWeb
2. **Train Gemma sweeps** (~7.2h): TFA-pos + TXCDR T=5 + Stacked T=5, k=50/100, 2 layers × shuffled/unshuffled = 24 runs
3. **Cache DeepSeek-R1-8B activations** (~3h): 12K seqs × 128 tokens, layers 12+24, FineWeb
4. **Train DeepSeek sweeps** (~3.1h): same 3 architectures, k=50/100, 1 layer × shuffled/unshuffled = 12 runs
5. **Aggregate** all results to `results/nlp_sweep/all_results.json`

## Architecture details

| Model | What it is | Data format | Key parameter |
|---|---|---|---|
| TFA-pos | Causal attention SAE with positional encoding | Full 128-token sequences | bottleneck_factor=8 (controls attention param count) |
| TXCDR T=5 | Shared-latent crosscoder, 5-token window | 5-token sliding windows | k×T active latents |
| Stacked T=5 | Independent per-position SAEs, 5-token window | 5-token sliding windows | k active per position |

## Important implementation notes

- **TFA input scaling**: TFA has `lam = 1/(4*dimin)` internally which assumes norm ~ sqrt(d). Real LM activations have norms ~200-1200. The TFASpec computes `scaling_factor = sqrt(d)/mean(||x||)` on the first training batch and applies it to all inputs. Without this, TFA produces NaN.
- **Memory-efficient data loading**: The cached activations pipeline reads slices from numpy mmap on demand (not materializing the full 28GB array). This keeps RAM usage at ~3GB.
- **Incremental saves**: Each completed run writes to JSON immediately. On resume, existing results are loaded and completed runs are skipped.
- **Shuffled control**: Running every experiment twice (unshuffled + shuffled) decomposes temporal vs architectural advantage.

## Expansion factors

- Gemma (d_model=2304): **8× expansion** → d_sae=18,432
- DeepSeek (d_model=4096): **4× expansion** → d_sae=16,384 (TFA too expensive at 8× for 4096)

## Monitoring

```bash
# Check progress
grep -c "NMSE=" logs/nlp_sweep.log        # completed runs (target: 36)
tail -5 logs/nlp_sweep.log                  # latest activity
nvidia-smi                                  # GPU usage

# Check results so far
find results/nlp_sweep -name "*.json" -exec sh -c 'echo "$1: $(python3 -c "import json; print(len(json.load(open(\"$1\"))))" 2>/dev/null) runs"' _ {} \;
```

## After completion

Push results back:
```bash
git add results/nlp_sweep/ && git commit -m "NLP sweep results: TFA vs TXCDR on Gemma + DeepSeek" && git push origin han
```

## If the A40 has 48GB VRAM

You can optionally use `--tfa-bottleneck-factor 4` instead of 8 for better TFA attention capacity. Edit `scripts/run_nlp_sweep_16h.sh` and change `--tfa-bottleneck-factor 8` to `--tfa-bottleneck-factor 4`. This makes TFA ~20% slower per run but gives it a stronger attention mechanism.
