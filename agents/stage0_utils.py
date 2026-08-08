"""Stage-0 helpers: safe OG JSON parse + append-only trace.jsonl.

Keep semantics close to the original RoboAgent code so SR should not regress.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Optional


def parse_og_response(res: str) -> Any:
    """Mirror previous OG parse logic, but use json.loads instead of eval().

    Original:
      assert (res.startswith("```json") and res.endswith("```")) or res.lower() == "no"
      ret = eval(res[8:-3].strip()) if (res.startswith("```json") and res.endswith("```")) else False
    """
    res = (res or "").strip()
    if res.lower() == "no":
        return False
    if res.startswith("```json") and res.endswith("```"):
        payload = res[8:-3].strip()
        return json.loads(payload)
    # Preserve old hard-fail behavior for unexpected formats.
    assert False, res


def append_trace(save_path: Optional[str], event: dict) -> None:
    if not save_path:
        return
    os.makedirs(save_path, exist_ok=True)
    path = os.path.join(save_path, "trace.jsonl")
    row = dict(event)
    row.setdefault("ts", time.time())
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
