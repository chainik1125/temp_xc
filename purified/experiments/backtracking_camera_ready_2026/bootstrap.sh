#!/usr/bin/env bash
set -euo pipefail
cd /workspace/backtracking
python -m venv --system-site-packages venv
venv/bin/python -m pip install 'huggingface_hub>=0.34,<1' 'transformers==4.56.2' 'pydantic>=2,<3' safetensors scikit-learn scipy pandas pyyaml einops jaxtyping tqdm openai matplotlib pytest accelerate datasets
tar -xf historical.tar -C historical
printf '%s\n' 284a8bf5e3e5a7cc094dd68c6fa5a92a9fd4eec3 > historical/purified/HISTORICAL_COMMIT
venv/bin/python -m pip freeze > logs/environment.txt
venv/bin/python code/fetch_assets.py
