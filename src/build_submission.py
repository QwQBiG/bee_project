"""Compatibility wrapper for Attachment 4 style commands."""

import sys
from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parent
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from deployment.build_submission import main


if __name__ == "__main__":
    raise SystemExit(main())
