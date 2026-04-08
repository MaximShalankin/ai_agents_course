"""
Способ получения данных nthakur/mirage-bench: скачивание датасета из Hugging Face
в data/benchmarks через huggingface_hub.snapshot_download.
"""
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.common import MIRAGE_BENCH_DIR

REPO_ID = "nthakur/mirage-bench"
REPO_TYPE = "dataset"


def download_mirage_bench(local_dir=None):
    local_dir = local_dir or MIRAGE_BENCH_DIR
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=REPO_ID, repo_type=REPO_TYPE, local_dir=str(local_dir))
    return local_dir


if __name__ == "__main__":
    dest = download_mirage_bench()
    print(f"Downloaded {REPO_ID} to {dest}")
