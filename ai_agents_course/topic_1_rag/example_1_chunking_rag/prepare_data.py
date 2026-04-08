"""Load data_v1.rtf, chunk by numbered points, print stats and save chunk length distribution plot."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.common import extract_text_from_rtf, chunk_by_points, DATA_RTF_PATH

import matplotlib.pyplot as plt


def main():
    text = extract_text_from_rtf(DATA_RTF_PATH)
    if not text:
        return
    chunks = chunk_by_points(text)
    lengths = [len(c) for c in chunks]
    print(f"Chunks: {len(chunks)}")
    print(f"Length: min={min(lengths)}, max={max(lengths)}, mean={sum(lengths)/len(lengths):.0f}")

    artifacts_dir = Path(__file__).resolve().parent / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(lengths, bins=50, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Chunk length (chars)")
    ax.set_ylabel("Count")
    ax.set_title("Chunk length distribution")
    fig.tight_layout()
    fig.savefig(artifacts_dir / "chunk_lengths.png", dpi=100)
    plt.close()
    print(f"Saved {artifacts_dir / 'chunk_lengths.png'}")


if __name__ == "__main__":
    main()
