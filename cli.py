#!/usr/bin/env python
"""Root entry point so the CLI works without installing the package.

    python cli.py demo
    python cli.py ask "what is the lock-in period?"
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from verirag.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
