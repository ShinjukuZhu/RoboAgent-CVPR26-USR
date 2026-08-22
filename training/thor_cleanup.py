#!/usr/bin/env python3
"""Kill orphaned AI2-THOR children for the current user."""
from __future__ import annotations

import getpass
import subprocess
import sys


def cleanup_thor(user: str | None = None, sig: str = "TERM") -> int:
    user = user or getpass.getuser()
    flag = f"-{sig}" if not sig.startswith("-") else sig
    return subprocess.call(
        ["pkill", flag, "-u", user, "-f", "thor-201909061227-Linux64"],
        stderr=subprocess.DEVNULL,
    )


if __name__ == "__main__":
    sig = sys.argv[1] if len(sys.argv) > 1 else "TERM"
    raise SystemExit(cleanup_thor(sig=sig))
