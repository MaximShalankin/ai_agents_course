"""Скачивание GGUF-моделей (FP16 и Q4_K_M) из Hugging Face.

Репозиторий: Qwen/Qwen2.5-Coder-7B-Instruct-GGUF.
Форматы: Q4_K_M — оптимальное соотношение качество/размер (Production Standard).
После скачивания добавьте модели в Ollama: FROM ./models/...gguf, затем ollama create.
"""
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF"
SCRIPT_DIR = Path(__file__).resolve().parent
LOCAL_DIR = SCRIPT_DIR / "models"

FILENAMES = [
    "qwen2.5-coder-7b-instruct-fp16.gguf",
    "qwen2.5-coder-7b-instruct-q4_k_m.gguf",
]


def main():
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    for filename in FILENAMES:
        local_path = LOCAL_DIR / filename
        label = "FP16" if "fp16" in filename else "Q4_K_M"
        if local_path.is_file():
            print(f"{label} ({filename}) already exists at {local_path}, skipping download.")
            continue
        print(f"Downloading {label} ({filename}) from {REPO_ID}...")
        hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            local_dir=str(LOCAL_DIR),
        )
        print(f"{label} download complete.")
    print(f"Models in {LOCAL_DIR}")


if __name__ == "__main__":
    main()
