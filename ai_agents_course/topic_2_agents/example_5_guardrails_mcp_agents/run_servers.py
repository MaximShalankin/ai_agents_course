"""Запуск обоих MCP-серверов: honest на 8000, evil на 8001. Остановка: Ctrl+C."""
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def main():
    honest = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "honest_mcp_server:app", "--host", "0.0.0.0", "--port", "8000"],
        cwd=SCRIPT_DIR,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    evil = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "evil_mcp_server:app", "--host", "0.0.0.0", "--port", "8001"],
        cwd=SCRIPT_DIR,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    print("Honest MCP :8000, Evil MCP :8001 started. Ctrl+C to stop.")
    try:
        honest.wait()
        evil.wait()
    except KeyboardInterrupt:
        honest.terminate()
        evil.terminate()
        honest.wait()
        evil.wait()
        print("Stopped.")


if __name__ == "__main__":
    main()
