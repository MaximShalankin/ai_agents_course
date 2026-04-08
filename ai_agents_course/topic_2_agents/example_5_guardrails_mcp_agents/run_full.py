"""Полный прогон: запуск обоих MCP-серверов, run_demo, остановка серверов."""
import sys
import time
from pathlib import Path

import urllib.error
import urllib.request

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

HONEST_BASE = "http://127.0.0.1:8000"
EVIL_BASE = "http://127.0.0.1:8001"
MAX_WAIT_S = 15
POLL_INTERVAL_S = 0.5


def _server_ok(url: str, timeout: float = 2.0) -> bool:
    try:
        req = urllib.request.Request(url, method="GET")
        urllib.request.urlopen(req, timeout=timeout)
        return True
    except urllib.error.HTTPError:
        return True
    except Exception:
        return False


def _wait_servers() -> bool:
    deadline = time.monotonic() + MAX_WAIT_S
    while time.monotonic() < deadline:
        if _server_ok(HONEST_BASE) and _server_ok(EVIL_BASE):
            return True
        time.sleep(POLL_INTERVAL_S)
    return False


def main():
    import subprocess

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
    try:
        if not _wait_servers():
            print("Серверы 8000/8001 не поднялись за {} с.".format(MAX_WAIT_S), file=sys.stderr)
            return 1
        import run_demo
        run_demo.main()
        return 0
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 1
    finally:
        honest.terminate()
        evil.terminate()
        honest.wait()
        evil.wait()


if __name__ == "__main__":
    sys.exit(main())
