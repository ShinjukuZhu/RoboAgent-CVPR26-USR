
# --- Fix ai2thor Flask-Jinja2 compatibility ---
import jinja2
import markupsafe

# Provide missing symbols that Flask (old) expects
if not hasattr(jinja2, "escape"):
    jinja2.escape = markupsafe.escape
if not hasattr(jinja2, "Markup"):
    jinja2.Markup = markupsafe.Markup
# ------------------------------------------------------
from embodiedbench.envs.eb_alfred.EBAlfEnv import EBAlfEnv
import copy
import numpy as np
import torch
import random
import sys
import os
import json
import argparse


from runners.ebalf_runner import EBAlfRunner as Runner
from agents.agent import Agent
from env_monkey_patch.eb_alf import EBAlfEnv_init_patch
EBAlfEnv.__init__ = EBAlfEnv_init_patch




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=50)
    parser.add_argument("--server-num", type=str, default='99')
    parser.add_argument("--split", type=str, default='base')
    parser.add_argument("--save_path", type=str, default='imgs/AW_eval')
    parser.add_argument("--qwen_path", type=str, default='../CKPT')
    parser.add_argument("--data_path", type=str, default="../EmbodiedBench/embodiedbench/envs/eb_alfred/data/splits/splits.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    seed = args.seed
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    START = args.start
    END = args.end
    DISPLAY = args.server_num
    
    env = EBAlfEnv(eval_set=args.split, data_path=args.data_path,  log_path="", down_sample_ratio=1.0, selected_indexes=[], x_display=DISPLAY, start_idx=START)
    SAVE_PATH = f"{args.save_path}-{args.split}"
    os.makedirs(SAVE_PATH, exist_ok=True)
    # Step7: run manifest (immutable metadata + model hashes + git commit)
    try:
        import sys as _sys
        _sys.path.insert(0, "/mnt/autodl_tmp1/zhuyanhao/code/RoboAgent_CVPR26/agents")
        from run_manifest import RunManifest
        from model_registry import ModelRegistry
        _man_config = {
            "seed": args.seed, "start": START, "end": END, "split": args.split,
            "qwen_path": args.qwen_path, "server_num": args.server_num,
            "og_backend": os.environ.get("ROBOAGENT_OG_BACKEND", ""),
            "eg_backend": os.environ.get("ROBOAGENT_EG_BACKEND", ""),
            "sd_backend": os.environ.get("ROBOAGENT_SD_BACKEND", ""),
            "llmdet_path": os.environ.get("ROBOAGENT_LLMDET_PATH", ""),
            "sd_comp": os.environ.get("ROBOAGENT_SD_COMP", ""),
            "usr_channel": os.environ.get("ROBOAGENT_USR_CHANNEL", ""),
        }
        _reg = ModelRegistry()
        # override brain with actual qwen_path
        _reg.set("brain", "run", args.qwen_path)
        _manifest = RunManifest(SAVE_PATH, _man_config, registry=_reg)
        _manifest.build()
        print("RUN_MANIFEST:", os.path.join(SAVE_PATH, "run_manifest.json"))
    except Exception as _e:
        print("MANIFEST_WARN:", str(_e)[:200])
    
    agent = Agent(args.qwen_path, "eb-alfred")
    agent.env = env
    for idx_episode in range(START, len(env.dataset)):
        if idx_episode >= END:
            break
        
        data = env.dataset[env._current_episode_num]
        env.reset()
        
        objs = set()
        for skill in env.language_skill_set:
            if skill.startswith("find a"):
                obj = skill.split(" ")[2]
                if '_' not in obj:
                    objs.add(obj + " 1")
                else:
                    objs.add(obj.replace("_", " "))
        print(objs)
        save_path_trial = os.path.join(SAVE_PATH, "episode_%d" % idx_episode)
        os.makedirs(save_path_trial, exist_ok=True)
        agent.reset(save_path_trial, obj_list=list(objs))
        
        runner = Runner(env, agent)

        agent.process_task(None, data["instruction"])
        score = runner.run()
        print(f"\n************RESULT FOR TASK {idx_episode}: {score}\n")
        # Persist scores (do not rely on console alone)
        try:
            gcr = float(score) if score is not None else 0.0
        except Exception:
            gcr = 0.0
        success = 1 if gcr >= 1.0 - 1e-8 else 0
        task_id = None
        try:
            task_id = data.get("task_id") or data.get("id") or data.get("task")
        except Exception:
            task_id = None
        with open(os.path.join(save_path_trial, "score.txt"), "w") as f:
            f.write(f"{gcr}\n")
        with open(os.path.join(save_path_trial, "result.txt"), "w") as f:
            f.write(f"task_idx={idx_episode}\ntask_id={task_id}\nGCR={gcr}\nSR={success}\n")
        try:
            if "_manifest" in dir():
                _manifest.log_episode(SAVE_PATH, idx_episode, {
                    "task_id": task_id, "GCR": gcr, "SR": success,
                    "instruction": (data.get("instruction") if isinstance(data, dict) else None),
                })
        except Exception as _e:
            pass
        agg_path = os.path.join(SAVE_PATH, "results.jsonl")
        rec = {
            "task_idx": idx_episode,
            "task_id": task_id,
            "GCR": gcr,
            "SR": success,
            "instruction": (data.get("instruction") if isinstance(data, dict) else None),
            "episode_dir": save_path_trial,
        }
        with open(agg_path, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        with open(os.path.join(SAVE_PATH, "scores_all.txt"), "a") as f:
            f.write(f"RESULT FOR TASK {idx_episode}: GCR={gcr} SR={success} task_id={task_id}\n")
        
    env.close()