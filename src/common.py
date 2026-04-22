"""Shared utilities: RTF extraction, chunking, data path."""
import re
from pathlib import Path

from striprtf.striprtf import rtf_to_text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RTF_PATH = PROJECT_ROOT / "data" / "text" / "data_v1.rtf"
DATA_VIDEO_PATH = PROJECT_ROOT / "data" / "video" / "video_2.mp4"
MIRAGE_BENCH_DIR = PROJECT_ROOT / "data" / "benchmarks" / "mirage-bench"


def extract_text_from_rtf(rtf_path):
    try:
        with open(rtf_path, "r", encoding="utf-8", errors="ignore") as f:
            rtf_content = f.read()
        return rtf_to_text(rtf_content)
    except Exception as e:
        print(f"Ошибка чтения RTF: {e}")
        return None


def chunk_by_points(text):
    pattern = r"\n(\d+\.\s+[A-Яа-яA-Za-z])"
    matches = list(re.finditer(pattern, text))

    if not matches:
        return [text.strip()] if text.strip() else []

    chunks = []
    for i, match in enumerate(matches):
        chunk_end = matches[i + 1].start() if i < len(matches) - 1 else len(text)
        chunk = text[match.start() : chunk_end].strip()
        chunks.append(chunk)

    first_part = text[: matches[0].start()].strip()
    if first_part:
        chunks.insert(0, first_part)

    return chunks
