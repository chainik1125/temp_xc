"""Remote-only, revision-pinned generation weights; no dataset sweep."""
from huggingface_hub import snapshot_download

if __name__ == '__main__':
    print(snapshot_download('deepseek-ai/DeepSeek-R1-Distill-Llama-8B',
        revision='6a6f4aa4197940add57724a7707d069478df56b1',
        allow_patterns=['*.safetensors', '*.json', 'tokenizer.model', '*.tiktoken', '*.txt']), flush=True)
