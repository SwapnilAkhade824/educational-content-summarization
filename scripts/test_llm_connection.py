"""Optional live check for the configured Google Gemini model.

Run explicitly with ``python scripts/test_llm_connection.py``. This script makes
one small request only when GEMINI_API_KEY is configured in ``.env`` or the
environment.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from utils.config import load_config
from summarization.llm_client import test_connection


def main() -> int:
    cfg = load_config()
    if not cfg.is_configured:
        print("Gemini is not configured. Set GEMINI_API_KEY in .env.")
        return 1

    ok, message = test_connection(cfg)
    print(message)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
