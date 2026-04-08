"""Транскрипция аудио в текст с помощью локальной модели Whisper. Сохраняет результат в .txt."""
import argparse
from pathlib import Path

from faster_whisper import WhisperModel

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.resolve().parents[2]
WHISPER_DOWNLOAD_ROOT = SCRIPT_DIR / "models" / "whisper"

DEFAULT_AUDIO = REPO_ROOT / "data" / "audio" / "новое_производство_газотурбинного_оборудования.mp3"


def main():
    parser = argparse.ArgumentParser(description="Транскрипция MP3 в TXT через Whisper.")
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO, help="Путь к MP3.")
    parser.add_argument("--out", type=Path, default=None, help="Путь к выходному TXT (по умолчанию рядом с аудио).")
    parser.add_argument("--language", default="ru", help="Язык (ru/en/None для авто).")
    args = parser.parse_args()

    audio_path = args.audio.resolve()
    if not audio_path.exists():
        raise FileNotFoundError(f"Аудио не найдено: {audio_path}")

    out_path = args.out.resolve() if args.out else audio_path.with_suffix(".txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    use_local = WHISPER_DOWNLOAD_ROOT.exists() and any(WHISPER_DOWNLOAD_ROOT.iterdir())
    model = WhisperModel(
        "large-v3-turbo",
        device="cpu",
        compute_type="int8",
        cpu_threads=4,
        download_root=str(WHISPER_DOWNLOAD_ROOT),
        local_files_only=use_local,
    )

    segments, _ = model.transcribe(
        str(audio_path),
        beam_size=5,
        language=args.language or None,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
    )

    lines = [seg.text.strip() for seg in segments if seg.text and seg.text.strip()]
    text = "\n".join(lines)

    out_path.write_text(text, encoding="utf-8")
    print(f"Сохранено: {out_path} ({len(text)} символов)")


if __name__ == "__main__":
    main()
