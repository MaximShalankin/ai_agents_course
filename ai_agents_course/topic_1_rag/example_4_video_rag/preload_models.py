"""Один раз загрузить модели в проект: Ollama (qwen3-vl:2b, bge-m3) и Whisper в example_4_video_rag/models."""
from pathlib import Path

import ollama
from faster_whisper import WhisperModel

SCRIPT_DIR = Path(__file__).resolve().parent
WHISPER_DOWNLOAD_ROOT = SCRIPT_DIR / "models" / "whisper"
OLLAMA_VL_MODEL = "qwen3-vl:2b"
OLLAMA_EMBED_MODEL = "bge-m3"


def main():
    print("Ollama: загрузка", OLLAMA_VL_MODEL, "...")
    ollama.pull(OLLAMA_VL_MODEL)
    print("Ollama: загрузка", OLLAMA_EMBED_MODEL, "...")
    ollama.pull(OLLAMA_EMBED_MODEL)

    WHISPER_DOWNLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    print("Whisper: загрузка large-v3-turbo в", WHISPER_DOWNLOAD_ROOT, "...")
    WhisperModel(
        "large-v3-turbo",
        device="cpu",
        compute_type="int8",
        cpu_threads=4,
        download_root=str(WHISPER_DOWNLOAD_ROOT),
    )
    print("Готово. Модели лежат в проекте (Ollama — в кэше Ollama, Whisper — в", WHISPER_DOWNLOAD_ROOT, ").")


if __name__ == "__main__":
    main()
