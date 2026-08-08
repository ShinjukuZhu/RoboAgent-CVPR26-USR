"""Step7: Model Registry — unified skill->model config with hash verification.

Every skill (Brain/OG/SD/EG) can be replaced independently via config/env.
The registry records which model serves which skill + its identity hash.
Supports: resolve(path)->hash, registry lookup by skill+variant, validation.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, Optional


DEFAULT_REGISTRY = {
    "brain": {
        "default": "/mnt/autodl_tmp1/zhuyanhao/ckpt/RoboAgent_CVPR26",
        "base": "/mnt/autodl_tmp2/zhuyanhao/Qwen2.5-VL-3B-Instruct.git",
    },
    "og": {
        "default": "/mnt/autodl_tmp1/zhuyanhao/ckpt/llmdet_large",   # LLMDet
        "gdino": "/mnt/autodl_tmp2/zhuyanhao/ckpts/grounding-dino-base",
        "qwen": "/mnt/autodl_tmp1/zhuyanhao/ckpt/RoboAgent_CVPR26",  # native OG
    },
    "sd": {
        "default": "/mnt/autodl_tmp2/zhuyanhao/ckpts/Florence-2-large-ft",
        "qwen": "/mnt/autodl_tmp1/zhuyanhao/ckpt/RoboAgent_CVPR26",
    },
    "eg": {
        "default": "/mnt/autodl_tmp1/zhuyanhao/ckpt/RoboAgent_CVPR26",  # native EG
        "explore": "/mnt/autodl_tmp2/zhuyanhao/Qwen2.5-VL-3B-Instruct.git",
    },
}


class ModelRegistry:
    def __init__(self, path: Optional[str] = None):
        self._reg: Dict[str, Dict[str, str]] = json.loads(json.dumps(DEFAULT_REGISTRY))
        if path and os.path.exists(path):
            with open(path) as f:
                user = json.load(f)
            for skill, variants in user.items():
                self._reg.setdefault(skill, {}).update(variants)

    def resolve(self, skill: str, variant: Optional[str] = None) -> Optional[str]:
        v = self._reg.get(skill, {})
        if variant and variant in v:
            return v[variant]
        return v.get("default")

    def set(self, skill: str, variant: str, path: str) -> None:
        self._reg.setdefault(skill, {})[variant] = path

    def dump(self) -> Dict[str, Any]:
        return self._reg

    # ---- hash helpers ----
    @staticmethod
    def hash_file(path: str, sample_bytes: int = 1 << 26) -> str:
        """SHA256 over first 64MB (fast) — sufficient for identity."""
        try:
            h = hashlib.sha256()
            with open(path, "rb") as f:
                h.update(f.read(sample_bytes))
            return h.hexdigest()[:16]
        except Exception as e:
            return f"ERR:{e}"

    @staticmethod
    def hash_dir(dirpath: str, pattern: str = "*.safetensors") -> Dict[str, str]:
        import glob
        out = {}
        for f in sorted(glob.glob(os.path.join(dirpath, pattern)))[:4]:
            out[os.path.basename(f)] = ModelRegistry.hash_file(f)
        return out

    def snapshot(self) -> Dict[str, Any]:
        """Full model identity snapshot for run manifest."""
        snap: Dict[str, Any] = {}
        for skill, variants in self._reg.items():
            snap[skill] = {}
            for variant, path in variants.items():
                if os.path.isdir(path):
                    snap[skill][variant] = {
                        "path": path,
                        "weights": self.hash_dir(path),
                    }
                elif os.path.isfile(path):
                    snap[skill][variant] = {"path": path, "sha256": self.hash_file(path)}
                else:
                    snap[skill][variant] = {"path": path, "missing": True}
        return snap


def load_registry(path: Optional[str] = None) -> ModelRegistry:
    return ModelRegistry(path)
