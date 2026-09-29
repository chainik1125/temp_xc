"""Fetch the allowlisted historical inputs on the GPU host, validating hashes."""
import hashlib
import json
from pathlib import Path
from huggingface_hub import hf_hub_download

ROOT = Path('/workspace/backtracking/assets')
REPO = 'han1823123123/temp-bench-data'
REVISION = '6ef9b1debf863dedcef9555cad3a4903fb9e8c43'
FILES = {
    'act_cache/fb2a74be884e512a/resid_post_L10.npy': 'dc34dfb117f77abddef4b4396d0d00afc707c39876d0ee36015de1e7b8406914',
    'c7_backtracking/stage_a/sentence_acts_L10.npz': '1656f6be2cd85fb85c8b246b9b27933f73ef40cfaac84078169dfd3bbbe27810',
    'act_cache/fb2a74be884e512a/corpus.json': None,
    'act_cache/fb2a74be884e512a/layer_specs.json': None,
}

def main():
    receipts = []
    for filename, expected in FILES.items():
        path = Path(hf_hub_download(REPO, filename, repo_type='dataset', revision=REVISION, local_dir=ROOT))
        sha = hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda: f.read(8 << 20), b''):
                sha.update(chunk)
        digest = sha.hexdigest()
        if expected and digest != expected:
            raise RuntimeError(f'Hash mismatch: {filename}')
        row = dict(file=filename, size_bytes=path.stat().st_size, sha256=digest)
        receipts.append(row)
        print(json.dumps(row), flush=True)
    (ROOT / 'receipts.json').write_text(json.dumps(dict(repo=REPO, revision=REVISION, files=receipts), indent=2))

if __name__ == '__main__':
    main()
