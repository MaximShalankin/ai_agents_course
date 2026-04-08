"""
Отправка sample_submition.csv в соревнование mws-ai-agents-2026 и ожидание результата (public score).
Аутентификация: kaggle-mcp/.env (KAGGLE_USERNAME, KAGGLE_KEY) или ~/.kaggle/kaggle.json.
"""
import os
import sys
import time
import traceback
from pathlib import Path

_dir = Path(__file__).resolve().parent
load_dotenv_path = _dir / "kaggle-mcp" / ".env"
if load_dotenv_path.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(load_dotenv_path)
    except ImportError:
        pass

from kaggle.api.kaggle_api_extended import KaggleApi

COMPETITION = "mws-ai-agents-2026"
SUBMISSION_FILE = _dir / "sample_submition.csv"
WAIT_INTERVAL = 25
WAIT_TIMEOUT = 600


def _get_submission_info(s):
    return {
        "fileName": getattr(s, "fileName", getattr(s, "file_name", "N/A")),
        "date": getattr(s, "date", "N/A"),
        "description": getattr(s, "description", "N/A"),
        "status": getattr(s, "status", "N/A"),
        "publicScore": getattr(s, "publicScore", getattr(s, "public_score", None)),
        "privateScore": getattr(s, "privateScore", getattr(s, "private_score", None)),
    }


def _log_submission_raw(s, label="submission"):
    """Лог атрибутов объекта submission для отладки."""
    attrs = [a for a in dir(s) if not a.startswith("_")]
    print("  [DEBUG] {} type={} attrs={}".format(label, type(s).__name__, attrs), file=sys.stderr)
    for name in ("status", "publicScore", "public_score", "privateScore", "private_score", "fileName", "file_name"):
        if hasattr(s, name):
            print("  [DEBUG]   {}={!r}".format(name, getattr(s, name)), file=sys.stderr)


def main():
    if not SUBMISSION_FILE.exists():
        print(f"Файл не найден: {SUBMISSION_FILE}", file=sys.stderr)
        sys.exit(1)

    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as e:
        print(f"Ошибка аутентификации Kaggle: {e}", file=sys.stderr)
        sys.exit(1)

    print("Отправка submission...")
    try:
        api.competition_submit(
            str(SUBMISSION_FILE),
            "Test submission from run_submit_and_metric.py",
            COMPETITION,
            quiet=False,
        )
        print("Submission отправлен. Ожидание результата (poll каждые {} с, макс {} с)...".format(WAIT_INTERVAL, WAIT_TIMEOUT))
    except Exception as e:
        print(f"Ошибка отправки: {e}", file=sys.stderr)
        sys.exit(1)

    deadline = time.monotonic() + WAIT_TIMEOUT
    while time.monotonic() < deadline:
        try:
            submissions = api.competition_submissions(COMPETITION, page_token="", page_size=5)
            if not submissions:
                print("  [DEBUG] submissions пустой список", file=sys.stderr)
                time.sleep(WAIT_INTERVAL)
                continue
            latest = submissions[0]
            _log_submission_raw(latest, "latest")
            info = _get_submission_info(latest)
            status = info["status"]
            pub = info["publicScore"]
            status_ok = bool(status and str(status).lower() == "complete")
            pub_ok = pub is not None
            print("  [DEBUG] status={!r} status_ok={} pub={!r} pub_ok={}".format(status, status_ok, pub, pub_ok), file=sys.stderr)
            if status_ok and pub_ok:
                print("\nРезультат:")
                print("  status:", status)
                print("  publicScore:", pub)
                if info["privateScore"] is not None:
                    print("  privateScore:", info["privateScore"])
                return
            print("  status={}, publicScore={} — ждём...".format(status, pub))
        except Exception as e:
            print("  Ошибка при опросе:", e, file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
        time.sleep(WAIT_INTERVAL)

    print("\nТаймаут. Последние submission:")
    try:
        subs = api.competition_submissions(COMPETITION, page_token="", page_size=5)
        if subs:
            _log_submission_raw(subs[0], "timeout_first")
        for i, s in enumerate(subs or [], 1):
            info = _get_submission_info(s)
            print("  {}: {} | status={} | publicScore={}".format(i, info["fileName"], info["status"], info["publicScore"]))
    except Exception as e:
        print("  ", e, file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
