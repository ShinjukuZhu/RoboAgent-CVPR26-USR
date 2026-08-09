# RoboAgent-CVPR26-USR

**Embodied Skill Replacement: Contract Mismatch → Decision-Compatible Adapter → Unified Skill Representation (USR) → SkillChannel → Decision-Aware Training**

This repository contains the **frozen experimental code** for our work on replacing specialized Foundation Model (FM) skill modules inside a pretrained embodied agent (RoboAgent, fine-tuned Qwen2.5-VL-3B) while keeping the Brain's contract intact.

> This is the **pre-optimization frozen snapshot** (commit `c1663fe4`). The `topconf-contract-evolution` branch contains subsequent optimization experiments.

---

## Problem

When a heterogeneous FM (e.g., Grounding DINO, LLMDet) replaces a skill module (e.g., object grounding) of a pretrained embodied agent:

- the FM's output **does not match the Brain's expected contract** (vocabulary, format, semantics);
- naive replacement causes **catastrophic degradation**;
- multiple skills need a **unified, typed, auditable communication channel**;
- the Brain needs to **consume decision signals** (found / confidence / uncertainty), not just object labels.

## Main Results (EB-ALFRED base, 50 episodes, seed 42)

| Config | Meaning | SR | GCR |
|--------|---------|-----|-----|
| Native | RoboAgent native | **78%** | 0.78 |
| Naive | GDINO direct replace, no adapter | **34%** | 0.34 |
| **Align** | GDINO + Decision-Compatible Adapter | **80%** | 0.80 |
| Align+USR | Align + USR Channel | 78% | 0.78 |
| **FullIndep** | OG + independent EG-LoRA + SD + USR | **80%** | 0.80 |

- **Exact McNemar (Align vs Naive): p < 0.001** (23 recovered, 0 regression).
- **DA-FT decision accuracy: 98%** (34/35 signal-sensitive cases).
- **Skill-aware ablation**: canonical 48% / canonical+signals 61% / flat-USR 58% / shuffled-USR 28%.
- **SkillChannel contract tests: 12/12.**

## Skill Architecture

The framework turns heterogeneous Foundation Models into **pluggable Embodied Skills** that communicate with the pretrained Brain through a **Unified Skill Representation (USR)** gated by a **SkillChannel**.

```mermaid
flowchart TB
    subgraph FM["Foundation Models (heterogeneous)"]
        DET["OG: LLMDet / Grounding DINO"]
        FLOR["SD: Florence-2"]
        EGV["EG: Qwen-EG / eg-LoRA"]
    end

    subgraph LOGIC["Skill Logic + Decision-Aware Adapter"]
        A1["remap v3 canonicalization
functional override / no-veto / found / fallback"]
        A2["SD parse (object → location/relation)"]
        A3["EG parse (in|on|target <obj>) + validator"]
    end

    subgraph USR["Unified Skill Representation (v2.0)"]
        F["environment_facts
(object.class / relation / location)"]
        T["task_semantics
(subgoal / role)"]
        S["decision_signals
(found / confidence / uncertainty)"]
        P["provenance
(detector / alignment_path)"]
    end

    subgraph CH["SkillChannel"]
        C1["publish(skill, USR)
schema + producer + episode_id + step_id"]
        C2["consume(skill, public_field)
whitelist only · raw blocked"]
        C3["contract audit + temporal isolation"]
    end

    subgraph BRAIN["Pretrained Brain (fine-tuned Qwen)"]
        SCH["Scheduler
(Think → Query protocol)"]
        LPM["LPM / action decoder
execute / guard / reobserve"]
    end

    DET --> A1 --> USR
    FLOR --> A2 --> USR
    EGV --> A3 --> USR
    USR --> C1
    C2 --> SCH
    C2 --> LPM
    SCH -->|skill call| CH
```

**ASCII version** (render-safe):

```
                        ┌─────────────────────────────────────────────┐
                        │       Foundation Models (heterogeneous)       │
                        │  OG: LLMDet / Grounding DINO                 │
                        │  SD: Florence-2                              │
                        │  EG: Qwen-EG / eg-LoRA (independent skill)   │
                        └───────────────┬──────────────┬───────────────┘
                                        │              │
                        ┌───────────────▼──────────────▼───────────────┐
                        │   Skill Logic + Decision-Aware Adapter        │
                        │  remap v3 (canonicalize / functional /        │
                        │  no-veto / found / fallback)                  │
                        └───────────────────────┬───────────────────────┘
                                                │
                        ┌───────────────────────▼───────────────────────┐
                        │   Unified Skill Representation (USR v2.0)     │
                        │  environment_facts · task_semantics          │
                        │  decision_signals · provenance               │
                        └───────────────────────┬───────────────────────┘
                                                │  publish(skill, USR)
                        ┌───────────────────────▼───────────────────────┐
                        │              SkillChannel                     │
                        │  consume(skill, public_field)  [whitelist]    │
                        │  contract audit · temporal isolation          │
                        │  raw model output BLOCKED                     │
                        └───────────────────────┬───────────────────────┘
                                                │  consume(USR fields)
                        ┌───────────────────────▼───────────────────────┐
                        │       Pretrained Brain (fine-tuned Qwen)      │
                        │  Scheduler (Think → Query) → LPM / action     │
                        │  execute · guard · reobserve                  │
                        └───────────────────────────────────────────────┘
```

**Role of each layer**

| Layer | Responsibility | Evidence |
|-------|---------------|----------|
| **Foundation Models** | heterogeneous, replaceable skill backends | OG=LLMDet/GDINO, SD=Florence-2, EG=independent |
| **Skill Logic + Adapter** | convert raw FM output into Brain-compatible contract | naive 34% → aligned 80% |
| **USR** | typed, temporal, auditable shared representation | 100% OG→USR→SD propagation, no raw bypass |
| **SkillChannel** | publish/consume gate + contract audit + raw-leak isolation | 12/12 contract tests |
| **Brain** | consumes USR facts + decision signals | DA-FT 98% counterfactual accuracy |

## Core Findings

1. **Contract Mismatch + Decision-Compatible Adapter** is the primary performance contribution: naive 34% → aligned 80% (+46pp). In audited failures, the dominant failure mode was downstream contract mismatch (OG detection succeeds but the label does not match the Brain's expected contract → wrong downstream action).

2. **USR** provides a **typed / temporal / auditable** unified interface for heterogeneous skill outputs and decision signals. The observed behavioral gain is attributable primarily to **explicit decision signals** (canonical+signals 61% ≈ flat-USR 58%), not the USR serialization itself.

3. **Decision-aware training** enables the Brain to consume decision signals (DA-FT 98% counterfactual accuracy vs raw-FT 40%).

## Repository Layout

```
agents/
├── agent.py                 # RoboAgent agent: scheduler + 7 skills + USR Channel hooks
├── og_remap_only.py         # Decision-Compatible Adapter (remap v3) for OG
├── eg_llm_backend.py        # EG text-LLM backend + validator
├── eg_lora_backend.py       # Independent EG-LoRA skill backend
├── eg_adapter_backend.py    # EG Skill Logic + Adapter (Base Qwen)
├── eg_explore_backend.py    # ExploreVLM-style EG (Base Qwen variants)
├── usr.py                   # Unified Skill Representation construction/ablation
├── usr_channel.py           # SkillChannel: publish/consume + contract audit + raw-leak guard
├── usr_og_backend.py        # OG → USR backend (decision-equivalent to remap v3)
├── usr_sd_eg.py             # SD/EG → USR parse + round-trip
├── usr_reliability.py       # USR schema/type/missing/conflict validation
├── usr_og_adapter.py        # OG USR adapter
├── contract_adapter.py      # Contract adapter (canonical/functional)
├── skill_alignment.py       # canonicalize + functional override
├── detector_registry.py     # detector registry (LLMDet / GDINO)
├── model_registry.py        # model registry (skill+variant → path/hash)
├── run_manifest.py          # run manifest (git/hash/config/episode tracking)
├── florence2_sd.py          # Florence-2 SD backend
├── sd_florence_cascade.py   # SD cascade
├── llmdet_og.py / llmdet_qwen_og.py / naive_detector.py / og_cascade_gated*.py
├── skill_memory.py / skill_interface.py / stage0_utils.py
└── prompt_aw.py / prompt_ebalf.py
runners/                     # EB-ALFRED + ALFWorld runners
run_ebalf.py                 # EB-ALFRED evaluation entry
run_aw.py                    # ALFWorld evaluation entry
eval_config.yaml
```

## Reproduction Notes

- **Environment**: AI2-THOR (thor-201909061227) via EmbodiedBench; Python 3.12, torch 2.8.0, transformers 4.57.0, RTX 6000D.
- **Brain**: fine-tuned Qwen2.5-VL-3B (RoboAgent_CVPR26, frozen).
- **Data**: EB-ALFRED base (50 ep) / ALFWorld eval_out_of_distribution (134 ep).
- Checkpoints and model weights are **not** included (large binary files); see `FINAL_MODEL_MANIFEST.json` in the frozen archive for paths and SHA-256.

## Frozen Archive

A read-only frozen snapshot of the full project (code + runs + training data + checkpoints + SHA256SUMS) is archived under `frozen_final_closure_<TS>/` on the experiment machine. All final numbers in this README are computed from per-episode manifests.

## References / Related Work

- RoboAgent (CVPR 2026): single fine-tuned Qwen2.5-VL-3B as a unified embodied Brain.
- Grounding DINO / LLMDet: open-vocabulary object detectors.
- Florence-2: prompt-based unified vision model.
