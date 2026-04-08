"""Build video metadata: extract audio (Whisper) and frames every 5s (qwen3-vl:2b), save to video_metadata.json."""
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import ollama
from faster_whisper import WhisperModel

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.resolve().parents[2]
ARTIFACTS_DIR = SCRIPT_DIR / "artifacts"
WHISPER_DOWNLOAD_ROOT = SCRIPT_DIR / "models" / "whisper"
DEFAULT_VIDEO = REPO_ROOT / "data" / "video" / "video_2.mp4"
METADATA_PATH = ARTIFACTS_DIR / "video_metadata.json"
VISUAL_PROMPT = "Опиши кратко что на изображении."
MAX_VIDEO_SEC = 180


def check_ffmpeg():
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg не найден. Установите ffmpeg и добавьте в PATH.")


def extract_audio(video_path: Path, audio_path: Path, max_sec: int = MAX_VIDEO_SEC) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-t",
            str(max_sec),
            "-i",
            str(video_path),
            "-vn",
            "-acodec",
            "libmp3lame",
            "-q:a",
            "2",
            str(audio_path),
        ],
        check=True,
        capture_output=True,
    )


def transcribe_audio(audio_path: Path) -> list[dict]:
    WHISPER_DOWNLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    model = WhisperModel(
        "large-v3-turbo",
        device="cpu",
        compute_type="int8",
        cpu_threads=4,
        download_root=str(WHISPER_DOWNLOAD_ROOT),
        local_files_only=any(WHISPER_DOWNLOAD_ROOT.iterdir()) if WHISPER_DOWNLOAD_ROOT.exists() else False,
    )
    segments, _ = model.transcribe(
        str(audio_path),
        beam_size=5,
        language="ru",
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
    )
    records = []
    for seg in segments:
        records.append({
            "start_sec": round(seg.start, 2),
            "end_sec": round(seg.end, 2),
            "type": "audio",
            "text": (seg.text or "").strip(),
        })
    return records


def extract_frames_every_5s(video_path: Path, out_dir: Path, max_sec: int = MAX_VIDEO_SEC) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = out_dir / "frame_%04d.png"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-t",
            str(max_sec),
            "-i",
            str(video_path),
            "-vf",
            "fps=1/5",
            str(pattern),
        ],
        check=True,
        capture_output=True,
    )
    frames = sorted(out_dir.glob("frame_*.png"))
    return frames


def describe_frame(image_path: Path, model: str = "qwen3-vl:2b") -> str:
    resp = ollama.chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": VISUAL_PROMPT,
                "images": [str(image_path)],
            }
        ],
    )
    return (resp.get("message", {}).get("content") or "").strip()


def build_visual_records(video_path: Path, max_sec: int = MAX_VIDEO_SEC) -> list[dict]:
    records = []
    with tempfile.TemporaryDirectory(prefix="example4_frames_") as tmpdir:
        frames_dir = Path(tmpdir)
        frames = extract_frames_every_5s(video_path, frames_dir, max_sec=max_sec)
        for i, frame_path in enumerate(frames):
            start_sec = i * 5
            end_sec = start_sec + 5
            text = describe_frame(frame_path)
            records.append({
                "start_sec": start_sec,
                "end_sec": end_sec,
                "type": "visual",
                "text": text,
            })
    return records


def main():
    parser = argparse.ArgumentParser(description="Сбор метаданных видео (аудио + визуал).")
    parser.add_argument("--force", action="store_true", help="Пересобрать даже если video_metadata.json уже есть.")
    parser.add_argument("--video", type=Path, default=None, help=f"Путь к видео (по умолчанию: {DEFAULT_VIDEO.name}).")
    parser.add_argument("--max-sec", type=int, default=MAX_VIDEO_SEC, help=f"Ограничить длительность обработки в секундах (по умолчанию {MAX_VIDEO_SEC}).")
    args = parser.parse_args()

    video_path = args.video.resolve() if args.video else DEFAULT_VIDEO
    max_sec = max(1, args.max_sec)

    check_ffmpeg()
    if not video_path.exists():
        raise FileNotFoundError(f"Видео не найдено: {video_path}")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    if METADATA_PATH.exists() and not args.force:
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("video_path") == str(video_path):
            print(f"Метаданные уже есть для {video_path}. Используйте --force для пересборки.")
            return

    audio_path = ARTIFACTS_DIR / "audio.mp3"
    extract_audio(video_path, audio_path, max_sec=max_sec)

    audio_records = transcribe_audio(audio_path)
    visual_records = build_visual_records(video_path, max_sec=max_sec)

    entries = audio_records + visual_records
    entries.sort(key=lambda x: x["start_sec"])

    from datetime import datetime, timezone
    payload = {
        "video_path": str(video_path),
        "built_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "entries": entries,
    }
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Сохранено {len(entries)} записей в {METADATA_PATH}")


if __name__ == "__main__":
    main()
