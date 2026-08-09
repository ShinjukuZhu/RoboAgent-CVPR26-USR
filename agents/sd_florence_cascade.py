"""SD cascade: Florence-2 (A) → optional style adapter (B) → gated Qwen verify (C).

Env:
  ROBOAGENT_SD_BACKEND=florence2_qwen_verify
  ROBOAGENT_FLORENCE_PATH / ROBOAGENT_FLORENCE2_PATH
  ROBOAGENT_SD_ALWAYS_VERIFY=0
  ROBOAGENT_SD_MIN_CHARS=40
  ROBOAGENT_SD_ADAPTER=1   (0 = pass-through Stage B)

Return contract (RoboAgent SD):
  free-text str; "" if grounding_label is missing.
"""
from __future__ import annotations

import os
from typing import Any, Callable, List, Optional, Tuple


def min_chars(default: int = 40) -> int:
    return int(os.environ.get("ROBOAGENT_SD_MIN_CHARS", str(default)))


def always_verify() -> bool:
    return os.environ.get("ROBOAGENT_SD_ALWAYS_VERIFY", "0").strip() in (
        "1",
        "true",
        "True",
        "yes",
    )


def adapter_enabled() -> bool:
    return os.environ.get("ROBOAGENT_SD_ADAPTER", "1").strip() not in (
        "0",
        "false",
        "False",
        "no",
    )


def should_invoke_stage_c(
    florence_text: str,
    grounding_label: Optional[str],
    *,
    invent: bool = False,
    min_len: Optional[int] = None,
) -> Tuple[bool, List[str]]:
    """Pure gate (unit-testable, no GPU).

    Invoke Stage C (FT Qwen SD) iff any reason fires.
    """
    reasons: List[str] = []
    if min_len is None:
        min_len = min_chars()

    if always_verify():
        reasons.append("always_verify")

    text = (florence_text or "").strip()
    if not text or len(text) < min_len:
        reasons.append("short_or_empty")

    spatial_markers = (
        "left",
        "right",
        "on ",
        "in ",
        "next to",
        "beside",
        "above",
        "below",
        "near",
    )
    if text and not any(m in text.lower() for m in spatial_markers):
        reasons.append("no_spatial_cue")

    if invent:
        reasons.append("invent")

    if grounding_label and len(str(grounding_label).split()) >= 4:
        reasons.append("complex_label")

    return (len(reasons) > 0, reasons)


def describe_florence_stage_a(
    rgb_path: str,
    grounding_label: str,
    invent: bool = False,
) -> str:
    """Stage A: Florence-2 caption via lab florence2_sd.describe_naive."""
    from agents.florence2_sd import describe_naive

    return describe_naive(rgb_path, grounding_label, invent=invent)


def adapt_sd_style(raw_caption: str, grounding_label: str) -> str:
    """Stage B: light alignment toward FT scene-graph style.

    - strip whitespace
    - ensure grounding_label token appears when missing (append cue)
    - trim runaway length
    """
    text = (raw_caption or "").strip()
    if not adapter_enabled():
        return text
    label = (grounding_label or "").strip()
    if label:
        # case-insensitive containment of the first token of label
        key = label.split()[0].lower()
        if key and key not in text.lower():
            text = (text + f" Near {label}.").strip()
    if len(text) > 600:
        text = text[:597].rstrip() + "..."
    return text


def describe_cascade(
    rgb_path: str,
    grounding_label: Optional[str],
    invent: bool,
    prompt_sd: Any,
    qwen_infer: Callable[..., str],
    *,
    save_path: Optional[str] = None,
) -> str:
    """Full A→B→gate→C path. Mirrors SD wire contract used by agent.py."""
    if grounding_label is None:
        return ""

    florence_raw = describe_florence_stage_a(rgb_path, grounding_label, invent=invent)
    adapted = adapt_sd_style(florence_raw, grounding_label)
    need_c, reasons = should_invoke_stage_c(
        adapted, grounding_label, invent=invent
    )

    if not need_c:
        path = "florence_accept"
        out = adapted
    else:
        path = "qwen_verify"
        # prompt_sd is typically a format-string or callable-ready template
        try:
            prompt_text = prompt_sd.format(grounding_label) if hasattr(prompt_sd, "format") else str(prompt_sd)
        except Exception:
            prompt_text = str(prompt_sd)
        note = ""
        if invent:
            note = " (invent spatial relations if helpful)"
        try:
            out = qwen_infer(prompt_text + note)
        except TypeError:
            out = qwen_infer(prompt_text)
        out = (out or "").strip()
        if not out:
            # fall back to Florence rather than empty
            out = adapted
            path = "qwen_empty_fallback_florence"

    if save_path:
        try:
            from agents.stage0_utils import append_trace

            append_trace(
                save_path,
                {
                    "event": "sd_cascade",
                    "path": path,
                    "gate_reasons": reasons,
                    "florence_raw_len": len(florence_raw or ""),
                    "out_len": len(out or ""),
                },
            )
        except Exception:
            pass

    return out


if __name__ == "__main__":
    # GPU-free unit checks
    ok, why = should_invoke_stage_c("", "apple", invent=False)
    assert ok and "short_or_empty" in why
    long_ok = (
        "A red apple is on the left of the bowl near the plate on the counter top area."
    )
    ok2, why2 = should_invoke_stage_c(long_ok, "apple", invent=False)
    assert not ok2, why2
    ok3, why3 = should_invoke_stage_c(long_ok, "apple", invent=True)
    assert ok3 and "invent" in why3
    print("SD_CASCADE_UNIT_OK", adapt_sd_style("A bowl.", "Apple 1"))
