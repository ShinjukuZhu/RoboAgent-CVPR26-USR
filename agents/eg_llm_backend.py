"""EG text-LLM backend: validated FT path + optional 7B (disk-gated).

Contract (must match current exploration_guidance):
  return "in|on|target <obj>" validated against observed \\ explored,
  or None after max retries (caller drains ability_buffer → ["pass"]).

Env:
  ROBOAGENT_EG_BACKEND=qwen|validated_ft|qwen25_7b
  ROBOAGENT_EG_MODEL_PATH=...   (local dir for 7B Instruct)
  ROBOAGENT_EG_CONSTRAINED=1
  ROBOAGENT_EG_MAX_RETRIES=10
  ROBOAGENT_EG_DEVICE=...
"""
from __future__ import annotations

import os
import re
import threading
from typing import Any, Callable, List, Optional, Sequence, Set, Tuple


_EG_RE = re.compile(r"^(in|on|target)\s+(.+)$", re.IGNORECASE)

_LOCK = threading.Lock()
_MODEL = None
_TOKENIZER = None
_DEVICE = None
_7B_SKIP_REASON: Optional[str] = None


def max_retries(default: int = 10) -> int:
    return int(os.environ.get("ROBOAGENT_EG_MAX_RETRIES", str(default)))


def constrained_enabled() -> bool:
    return os.environ.get("ROBOAGENT_EG_CONSTRAINED", "1").strip() not in (
        "0",
        "false",
        "False",
        "no",
    )


def _normalize_obj(name: str, env_name: str) -> str:
    s = (name or "").strip()
    if env_name == "alfworld":
        return s.lower()
    return s


_EG_RELATIONS = frozenset({"in", "on", "target", "near"})


def legal_objects(
    observed_objects: Sequence[str],
    explored: Sequence[str],
    env_name: str,
) -> Set[str]:
    """Objects that still admit at least one unexplored relation."""
    obs = {_normalize_obj(x, env_name) for x in observed_objects if str(x).strip()}
    tried = {str(x).lower() for x in explored if str(x).strip()}
    out: Set[str] = set()
    for obj in obs:
        for rel in _EG_RELATIONS:
            if f"{rel} {obj}" not in tried:
                out.add(obj)
                break
    return out


def exploration_exhausted(
    observed_objects: Sequence[str],
    explored: Sequence[str],
    env_name: str,
) -> bool:
    return not legal_objects(observed_objects, explored, env_name)


def parse_eg_response(
    raw: str,
    allowed: Set[str],
    env_name: str,
    explored: Optional[Sequence[str]] = None,
) -> Tuple[Optional[str], Optional[str]]:
    if raw is None:
        return None, "empty"
    line = str(raw).strip().splitlines()[0].strip().strip("`")
    line = line.replace("{", "").replace("}", "").replace("<", "").replace(">", "")
    m = _EG_RE.match(line)
    if not m:
        return None, "schema_mismatch"
    rel, obj = m.group(1).lower(), m.group(2).strip()
    key = _normalize_obj(obj, env_name)
    if explored is not None:
        exp = {_normalize_obj(x, env_name) for x in explored}
        # full phrase also checked by caller historically; keep object-set check here
        full = f"{rel} {key}" if env_name == "alfworld" else f"{rel} {obj}"
        if full in explored or (env_name == "alfworld" and full.lower() in {str(x).lower() for x in explored}):
            return None, "already_explored"
    if key not in allowed:
        return None, "object_not_in_unexplored"
    if env_name == "alfworld":
        return f"{rel} {key}", None
    return f"{rel} {obj}", None


def _disk_free_gb(path: str) -> float:
    try:
        st = os.statvfs(path)
        return (st.f_bavail * st.f_frsize) / (1024**3)
    except Exception:
        return 0.0


def eg_7b_skip_reason() -> Optional[str]:
    """Return None if 7B can be used; else human reason."""
    global _7B_SKIP_REASON
    if _7B_SKIP_REASON is not None:
        return _7B_SKIP_REASON
    path = os.environ.get("ROBOAGENT_EG_MODEL_PATH", "").strip()
    if not path:
        _7B_SKIP_REASON = "ROBOAGENT_EG_MODEL_PATH unset; 7B deferred"
        return _7B_SKIP_REASON
    if not os.path.isdir(path):
        # Check free disk before claiming missing
        parent = "/mnt/autodl_tmp2/zhuyanhao" if os.path.isdir("/mnt/autodl_tmp2/zhuyanhao") else "/mnt/autodl_tmp1/zhuyanhao"
        free = _disk_free_gb(parent)
        if free < 20:
            _7B_SKIP_REASON = (
                f"EG 7B skipped: path missing and disk free {free:.1f}G < 20G "
                f"(Qwen2.5-7B-Instruct ~15G BF16). Use validated_ft or defer."
            )
        else:
            _7B_SKIP_REASON = f"EG 7B model path missing: {path}"
        return _7B_SKIP_REASON
    # rough size check
    try:
        total = 0
        for root, _, files in os.walk(path):
            for f in files:
                total += os.path.getsize(os.path.join(root, f))
        if total < 5 * (1024**3):
            _7B_SKIP_REASON = f"EG 7B path too small ({total/1024**3:.2f}G): {path}"
            return _7B_SKIP_REASON
    except Exception as e:
        _7B_SKIP_REASON = f"EG 7B path unreadable: {e}"
        return _7B_SKIP_REASON
    return None


def _ensure_eg_model() -> None:
    global _MODEL, _TOKENIZER, _DEVICE, _7B_SKIP_REASON
    if _MODEL is not None:
        return
    reason = eg_7b_skip_reason()
    if reason:
        raise RuntimeError(reason)
    with _LOCK:
        if _MODEL is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        path = os.environ["ROBOAGENT_EG_MODEL_PATH"].strip()
        device = os.environ.get(
            "ROBOAGENT_EG_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"
        )
        _TOKENIZER = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        _MODEL = AutoModelForCausalLM.from_pretrained(
            path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto"
        )
        _MODEL.eval()
        _DEVICE = device


def _generate_eg_candidate(
    prompt_eg: str,
    target_obj: str,
    observed_objects: Sequence[str],
    explored: Sequence[str],
    *,
    temperature: float = 0.8,
) -> str:
    _ = (target_obj, observed_objects, explored)
    _ensure_eg_model()
    import torch

    messages = [
        {"role": "system", "content": "You propose one exploration subgoal. Reply with exactly: in|on|target <object>."},
        {"role": "user", "content": prompt_eg},
    ]
    text = _TOKENIZER.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = _TOKENIZER(text, return_tensors="pt")
    inputs = {k: v.to(_MODEL.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = _MODEL.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=True,
            temperature=float(temperature),
            top_p=0.9,
        )
    gen = out[0][inputs["input_ids"].shape[-1] :]
    return _TOKENIZER.decode(gen, skip_special_tokens=True).strip()


def propose_eg(
    target_obj: str,
    observed_objects: Sequence[str],
    explored: Sequence[str],
    prompt_eg: str,
    *,
    env_name: str,
    qwen_infer: Optional[Callable[..., str]] = None,
    max_tries: Optional[int] = None,
) -> Optional[str]:
    """7B path; falls back to validated_ft if 7B unavailable and qwen_infer given."""
    tries = max_retries() if max_tries is None else max_tries
    allowed = legal_objects(observed_objects, explored, env_name)
    reason = eg_7b_skip_reason()
    if reason:
        if qwen_infer is not None:
            return propose_eg_ft_qwen(
                target_obj,
                observed_objects,
                explored,
                prompt_eg,
                qwen_infer,
                env_name=env_name,
                max_tries=tries,
            )
        raise RuntimeError(reason)

    last_reason: Optional[str] = None
    for i in range(tries):
        raw = _generate_eg_candidate(
            prompt_eg,
            target_obj,
            observed_objects,
            explored,
            temperature=0.8 + i * 0.1,
        )
        ok, why = parse_eg_response(raw, allowed, env_name, explored=explored)
        if ok is not None:
            return ok
        last_reason = why
    _ = last_reason
    # optional Stage C: FT fallback
    if qwen_infer is not None and constrained_enabled():
        return propose_eg_ft_qwen(
            target_obj,
            observed_objects,
            explored,
            prompt_eg,
            qwen_infer,
            env_name=env_name,
            max_tries=min(3, tries),
        )
    return None


def propose_eg_ft_qwen(
    target_obj: str,
    observed_objects: Sequence[str],
    explored: Sequence[str],
    prompt_eg: str,
    qwen_infer: Callable[..., Any],
    *,
    env_name: str,
    max_tries: Optional[int] = None,
) -> Optional[str]:
    """FT VLM EG with shared validator (mirrors agent.py resample schedule)."""
    allowed = legal_objects(observed_objects, explored, env_name)
    tries = max_retries() if max_tries is None else max_tries
    last_reason: Optional[str] = None
    for i in range(tries + 1):
        if i == 0:
            raw = qwen_infer(prompt_eg)
        else:
            more_args = {
                "do_sample": True,
                "temperature": 0.8 + i * 0.1,
                "top_k": 50,
                "top_p": 0.9,
            }
            try:
                raw = qwen_infer(prompt_eg, more_args)
            except TypeError:
                raw = qwen_infer(prompt_eg)
        if raw is None:
            raw = ""
        raw = str(raw).strip().replace("{", "").replace("}", "").replace("<", "").replace(">", "")
        ok, why = parse_eg_response(raw, allowed, env_name, explored=explored)
        if ok is not None:
            return ok
        last_reason = why
        if i >= tries:
            break
    _ = (target_obj, last_reason)
    return None
