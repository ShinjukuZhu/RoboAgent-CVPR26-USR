"""Step7: Run Manifest — immutable experiment metadata + per-episode tracking.

Each run writes a manifest (run_manifest.json) BEFORE execution with:
  - run id, timestamp
  - config (seed, episodes, split, backends)
  - model registry snapshot (hashes)
  - git commit (repo + prompt hash)
  - prompt hash (task instruction template)
Then per-episode results are tracked in episode_manifest.jsonl (append) instead of
a cumulative results.jsonl that mixes runs.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
import uuid
from typing import Any, Dict, Optional

from model_registry import ModelRegistry


def git_commit(repo: str) -> Optional[str]:
    try:
        out = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        if out.returncode == 0:
            return out.stdout.strip()[:12]
    except Exception:
        pass
    return None


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


class RunManifest:
    def __init__(self, save_dir: str, config: Dict[str, Any],
                 registry: Optional[ModelRegistry] = None):
        self.save_dir = save_dir
        self.config = config
        self.registry = registry or ModelRegistry()
        self.run_id = uuid.uuid4().hex[:12]
        os.makedirs(save_dir, exist_ok=True)

    def build(self) -> Dict[str, Any]:
        repo = "/mnt/autodl_tmp1/zhuyanhao/code/RoboAgent_CVPR26"
        manifest = {
            "run_id": self.run_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "config": self.config,
            "git_commit": git_commit(repo),
            "prompt_hash": hash_text(json.dumps(self.config.get("prompts", {}), sort_keys=True)),
            "models": self.registry.snapshot(),
        }
        path = os.path.join(self.save_dir, "run_manifest.json")
        with open(path, "w") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=1)
        return manifest

    def log_episode(self, episode_dir: str, episode: int, result: Dict[str, Any]) -> None:
        """Append per-episode record (no cross-run contamination)."""
        rec = {"episode": episode, "ts": time.time(), **result}
        path = os.path.join(self.save_dir, "episode_manifest.jsonl")
        with open(path, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        # also write per-episode result.txt (already exists in run_ebalf)
        ep_dir = os.path.join(self.save_dir, f"episode_{episode}")
        os.makedirs(ep_dir, exist_ok=True)
        with open(os.path.join(ep_dir, "result.json"), "w") as f:
            json.dump(rec, f, ensure_ascii=False, indent=1)


def build_manifest(save_dir: str, config: Dict[str, Any]) -> Dict[str, Any]:
    m = RunManifest(save_dir, config)
    return m.build()
