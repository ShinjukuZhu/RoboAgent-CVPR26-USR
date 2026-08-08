"""USR backends for SD and EG — emit USR to trace, keep Brain text equivalent.

Interface modes (gate 2 comparison):
  - raw:          Qwen native text (baseline, already in agent.py)
  - text_adapter: existing florence2_adapter / propose_eg_ft_qwen
  - usr:          this module — parse raw into USR; Brain still receives
                  decision-equivalent text; USR stored in trace for audit.

The USR backend does NOT change Brain-facing strings (preserves Soft/Explicit
Contract); it only ADDS structured USR to the trace so downstream Composition
and offline swap can consume facts.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

from agents.usr_sd_eg import (  # type: ignore
    sd_raw_to_usr,
    sd_usr_to_text,
    eg_raw_to_usr,
    eg_usr_to_text,
)


def sd_usr_backend(
    raw_text: str,
    target: Optional[str] = None,
    detector: str = "qwen25",
) -> Dict[str, Any]:
    """Wrap raw SD output -> (brain_text, usr). Brain text unchanged."""
    usr = sd_raw_to_usr(raw_text, detector=detector, target=target)
    text = sd_usr_to_text(usr)
    return {"text": text, "usr": usr}


def eg_usr_backend(
    direction: str,
    detector: str = "qwen25",
    observed_objects: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Wrap raw EG output -> (brain_text, usr). Brain text unchanged."""
    usr = eg_raw_to_usr(direction, detector=detector,
                        observed_objects=observed_objects)
    text = eg_usr_to_text(usr)
    return {"text": text, "usr": usr}
