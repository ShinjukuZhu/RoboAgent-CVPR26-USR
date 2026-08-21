import torch
import cv2

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from agents.qwen import inference as qwen_inference
from agents.stage0_utils import parse_og_response, append_trace
from agents.llmdet_og import ground as llmdet_ground
from agents.llmdet_qwen_og import ground_cascade as llmdet_qwen_ground
from agents.og_cascade_gated import ground_gated as llmdet_qwen_gated_ground
from agents.og_cascade_gated_v2 import ground_gated_v2 as llmdet_qwen_gated_v2_ground
from agents.og_remap_only import ground_remap_only as llmdet_qwen_remap_ground
from agents.usr_og_backend import ground_usr as llmdet_qwen_usr_ground
from agents.usr_sd_eg_backend import sd_usr_backend, eg_usr_backend
from agents.eg_explore_backend import propose_explore_eg as explore_eg_naive
from agents.eg_explore_backend import propose_explore_eg as explore_eg_aligned
from agents.eg_adapter_backend import propose_eg_adapter as explore_eg_adapter
from agents.eg_lora_backend import propose_eg_lora as explore_eg_lora
from agents.usr_channel import get_channel, reset_channel
from agents.skill_alignment_og import ground_aligned as llmdet_qwen_aligned_ground
from agents.naive_detector import ground_naive as naive_detector_ground
from agents.eg_llm_backend import propose_eg, propose_eg_ft_qwen
from agents.skill_memory import SkillMemory, env_mode as skill_memory_env_mode
from agents.evo_skill import EffectVerifiedSkill
from agents.florence2_sd import describe_naive, describe_adapter
from agents.sd_florence_cascade import describe_cascade as florence_qwen_verify_describe
import os
import time
from peft import PeftModel


class Agent(object):
    def __init__(self, vlm_model_path, env_name="alfworld"):
        import os as _os
        self.vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(vlm_model_path, torch_dtype=torch.bfloat16, device_map="auto")
        _br_adapt = _os.environ.get("ROBOAGENT_BRAIN_ADAPTER", "").strip()
        if _br_adapt:
            from peft import PeftModel
            self.vlm = PeftModel.from_pretrained(self.vlm, _br_adapt)
            self.vlm.eval()
            print("BRAIN_ADAPTER_LOADED:", _br_adapt)
        self.vlm_processer = AutoProcessor.from_pretrained(vlm_model_path)
        
        self.env_name = env_name
        if env_name == "alfworld":
            from agents.prompt_aw import prompt_og, prompt_sd, prompt_lpe, prompt_lpm, prompt_eg, prompt_es, prompt_ct
            self.prompt_og = prompt_og
            self.prompt_sd = prompt_sd
            self.prompt_lpe = prompt_lpe
            self.prompt_lpm = prompt_lpm
            self.prompt_eg = prompt_eg
            self.prompt_es = prompt_es
            self.prompt_ct = prompt_ct
        elif env_name == "eb-alfred":
            from agents.prompt_ebalf import prompt_og, prompt_sd, prompt_lpe, prompt_lpm, prompt_eg, prompt_es, prompt_ct
            self.prompt_og = prompt_og
            self.prompt_sd = prompt_sd
            self.prompt_lpe = prompt_lpe
            self.prompt_lpm = prompt_lpm
            self.prompt_eg = prompt_eg
            self.prompt_es = prompt_es
            self.prompt_ct = prompt_ct
        else:
            raise ValueError(f"Invalid environment name: {env_name}")
    
    def reset(self, save_path, obj_list):
        self.last_goto = None
        self.save_path = save_path
        # Phase3 Skill Memory: log-only by default; adapt OFF (no cross-ep leak).
        self._skill_memory = None
        _sm_mode = skill_memory_env_mode()
        if _sm_mode in ("log", "adapt"):
            self._skill_memory = SkillMemory(
                enable_cross_episode_adapt=False,  # hard OFF regardless of env
                persist_path=f"{self.save_path}/skill_memory.jsonl",
            )
        self._evo_interrupt_actions = False
        self._evo_skill = EffectVerifiedSkill.from_environment(
            trace_fn=lambda payload: append_trace(self.save_path, payload)
        )
        if self._evo_skill is not None:
            append_trace(self.save_path, {
                "event": "evo_skill_loaded",
                **self._evo_skill.manifest(),
            })
        self.observation_version = 0
        self.save_i = 0
        
        self.observed_objects_list = [x for x in sorted(obj_list)]
        reset_channel()
        self.core_history = ""
        self.explored = []
        self.invent = "nothing"
        self.last_local_traj = []
        self.ability_buffer = []
        self.ability_buffer_idx = 0
        self.last_to_find = None
        self.last_grounding_label = None
        self.last_og_usr = None
        self.exploration_subgoal = None
        self.slice_idx = 3 # for assigning ID of the sliced object in alfworld
        # initial placeholder frame so early image-based skills have a path
        import numpy as _np
        _placeholder = _np.zeros((480, 640, 3), dtype=_np.uint8)
        cv2.imwrite(f"{self.save_path}/step_0.png", _placeholder)
        self.cur_rgb_path = f"{self.save_path}/step_0.png"
        
        with open(f"{self.save_path}/qwen_log.txt", "w") as f:
            f.write("BEGIN!!!\n")
        
    def process_observation(self, rgb, env_step_id):
        cv2.imwrite(f"{self.save_path}/step_{env_step_id}.png", rgb[:, :, ::-1])
        self.cur_rgb_path = f"{self.save_path}/step_{env_step_id}.png"
        self.observation_version += 1
        if self._evo_skill is not None:
            self._evo_skill.note_new_observation()
        
        
    def process_task(self, task_info, task_instruction):
        print("[TASK] ", task_instruction)
        self.task_instruction = task_instruction
        if self._evo_skill is not None:
            self._evo_skill.set_task(task_instruction)
        return
    
    def process_feedback(self, message, last_action):
        assert last_action not in ["examine", "pass", "do nothing"]
        if self.env_name == "alfworld":
            assert last_action.split(" ")[0] in ["take", "open", "close", "put", "slice", "heat", "cool", "clean", "go", "use"], last_action
        elif self.env_name == "eb-alfred":
            assert last_action.split(" ")[0] in ["pick", "open", "close", "put", "slice", "turn", "find"], last_action

        self.last_action = last_action
        aid = len(self.last_local_traj) // 2 + 1
        self.last_local_traj.append(f"[action {aid}] {last_action}")
        self.last_local_traj.append(f"[feedback {aid}] {'success' if message else 'failure'}")

        if self._evo_skill is not None:
            intervention = self._evo_skill.observe_action_result(last_action, bool(message))
            if intervention is not None:
                self._activate_evo_intervention(intervention.reason, intervention.invalidate_suffix)
            if os.environ.get("ROBOAGENT_USR_CHANNEL", "0") == "1":
                try:
                    ch = get_channel()
                    og = ch.get_usr("og") or {}
                    if og:
                        temporal = dict(og.get("temporal_context") or {})
                        temporal["observation_version"] = self.observation_version
                        extra = dict(og.get("skill_specific") or {})
                        extra["confirmed_progress"] = list(self._evo_skill.confirmed_effects)
                        extra["last_effect"] = {
                            "expected": self._evo_skill.expected_effect(last_action),
                            "verified": bool(message),
                        }
                        og["temporal_context"] = temporal
                        og["skill_specific"] = extra
                        ch.publish("og", og, producer="effect_verified_skill", episode_step=self.observation_version)
                except Exception:
                    pass
        
        self.scene_description = ""
        if message:
            if self.env_name == "alfworld":
                if last_action.startswith("take "):
                    self.invent = last_action.split("take ")[1].split(" from")[0]
                elif last_action.startswith("put "):
                    self.invent = "nothing"
                    if message:
                        self.slice_idx += 1
                elif last_action.startswith("go to"):
                    self.last_goto = last_action.split("go to ")[1]
            elif self.env_name == "eb-alfred":
                if last_action.startswith("pick "):
                    self.invent = last_action.split("pick up the ")[1]
                    if "_" in self.invent:
                        self.invent = self.invent.split("_")[0]
                elif last_action.startswith("put down "):
                    self.invent = "nothing"
                elif last_action.startswith("find a "):
                    self.last_goto = last_action.split("find a ")[1]
                    if "_" in self.last_goto:
                        self.last_goto = self.last_goto.split("_")[0]
        return
    
    def get_qwen_action(self, ):
        raw_actions = self.get_qwen_action_raw()
        
        exec_actions = [] # to be executed in the environment
        actions = [] # to be passes to process_feedback

        if self.env_name == "alfworld":
            for raw_action in raw_actions:
                for x in ["LettuceSliced", "AppleSliced", "PotatoSliced", "TomatoSliced", "BreadSliced"]:
                    assert not f"{x} {None}" in raw_action
                    if f"{x}" in raw_action:
                        raw_action = raw_action.replace(f"{x}", f"sliced-{x[:-6]} {self.slice_idx}")
                exec_actions.append(raw_action)
            actions = exec_actions
        elif self.env_name == "alfworld_text":
            for raw_action in raw_actions:
                for x in ["LettuceSliced", "AppleSliced", "PotatoSliced", "TomatoSliced", "BreadSliced"]:
                    assert not f"{x} {None}" in raw_action
                    if f"{x}" in raw_action:
                        raw_action = raw_action.replace(f"{x}", f"sliced-{x[:-6]} {self.slice_idx}")
                if raw_action.split(" ")[0] == "put":
                    raw_action = raw_action.replace("put ", "move ")
                exec_actions.append(raw_action)
            actions = exec_actions
        elif self.env_name == "eb-alfred":
            for raw_action in raw_actions:
                # rename some objects to match the skill set of EmbodiedBench
                # generally this can be done by computing semantic similarity between the raw_action and the vocabulary of the environment, here we just use a rule-based matching for simplicity
                raw_action = raw_action.replace("Spray bottle", "SprayBottle").replace("the sponge", "the DishSponge").replace(" Sliced ", " ").replace("Floor lamp", "FloorLamp").replace("Desk lamp", "DeskLamp").replace("lettuce", "Lettuce").replace("apple", "Apple").replace("potato", "Potato").replace("tomato", "Tomato").replace("bread", "Bread").replace("the Sponge", "the DishSponge").replace("the Key ", "the KeyChain ").replace("Garbage can", "GarbageCan").replace("the towel", "the HandTowel").replace("Inkpen", "Pen").replace("Watering can", "WateringCan").replace("Armchair", "ArmChair").replace(" sliced ", " ").replace("Glass bottle", "Glassbottle").replace("Soap bottle", "SoapBottle").replace("SlicedApple", "Apple").replace("SlicedLettuce", "Lettuce").replace("SlicedBread", "Bread").replace("SlicedTomato", "Tomato").replace("SlicedPotato", "Potato").replace("GlassBottle", "Glassbottle").replace("the knife", "the Knife").replace("AppleSliced", "Apple").replace("LettuceSliced", "Lettuce").replace("BreadSliced", "Bread").replace("TomatoSliced", "Tomato").replace("PotatoSliced", "Potato")
                if raw_action.endswith("the Key"):
                    raw_action = raw_action.replace("the Key", "the KeyChain")

                # deal with the object index in EB's skill name
                if raw_action.split(" ")[0] in ["find", "open", "close"]:
                    if raw_action.endswith(" 1"):
                        action_str = raw_action[:-2]
                    else:
                        action_str = " ".join(raw_action.split(" ")[:-1]) + "_" + raw_action.split(" ")[-1]
                elif raw_action.split(" ")[0] in ["turn", "slice", "pick"]:
                    if raw_action in ["pick up the Apple", "pick up the Lettuce", "pick up the Tomato", "pick up the Potato", "pick up the Bread"]:
                        action_str = raw_action
                    else:
                        action_str = " ".join(raw_action.split(" ")[:-1])
                elif raw_action.split(" ")[0] in ["put", "pass", "examine", "fail"]:
                    action_str = raw_action
                else:
                    # print(raw_action)
                    # raise NotImplementedError
                    action_str = "fail"
                
                exec_actions.append(action_str)
                actions.append(raw_action)
        return exec_actions, actions

    def get_qwen_action_raw(self, ):
        if self.ability_buffer_idx >= len(self.ability_buffer):
            self.ability_buffer = []
            ret = self.get_core_result()
            if not ret:
                return ["fail"]
            assert len(self.ability_buffer)
            self.ability_buffer_idx = 0
        
        ability_name, ability_args = self.ability_buffer[self.ability_buffer_idx]
        ability_res = self.get_ability_result(ability_name, ability_args)
        if ability_name == "object_grounding" and self._evo_skill is not None:
            ability_res, intervention = self._evo_skill.validate_grounding(ability_args, ability_res)
            if intervention is not None:
                self._activate_evo_intervention(intervention.reason, intervention.invalidate_suffix)
        try:
            _ok = ability_res is not None and ability_res is not False and ability_res != ""
            self._maybe_record_skill(ability_name, ability_args, ability_res, _ok)
        except Exception:
            pass

        self.ability_buffer_idx += 1
        if ability_name == "exploration_guidance":
            self.last_to_find = ability_args    
            
            place = ability_res
            if place is None:
                self.core_history += "Exploration feedback: all locations exhausted. Consider alternative strategy.\n"
                # Drain remaining queued abilities (e.g. exploration_planner)
                # so we re-enter the scheduler instead of asserting on
                # empty exploration_subgoal.
                self.ability_buffer = []
                self.ability_buffer_idx = 0
                return ["pass"]
            self.explored.append(place)
            if place.startswith("target "):
                place = "arrive at " + place[len("target "):]
            elif place.startswith("in "):
                place = "check the inside of " + place[len("in "):]
            elif place.startswith("on "):
                place = "check " + place[len("on "):]
            else:
                place = "check " + place
            self.exploration_subgoal = place
        elif ability_name == "object_grounding":
            if ability_res == False:
                if not self.core_history.strip().endswith("Grounding feedback: the target object is not found"):
                    self.core_history += "Grounding feedback: the target object is not found\n"
            else:
                ocls = ability_res[0]["label"]
                if self.env_name == "alfworld":
                    ocls = ocls.replace("sliced ", "")
                if not self.core_history.strip().endswith("Grounding feedback: the target object is not found"):
                    self.core_history += f"Grounding feedback: the target object ({ocls}) is found at {self.last_goto}\n"
                else:
                    self.core_history = self.core_history.strip()[:-len("Grounding feedback: the target object is not found")] + f"Grounding feedback: the target object ({ocls}) is found at {self.last_goto}\n"
        elif ability_name == "exploration_planner":
            steps = ability_res
            self.last_local_traj = []
            self.exploration_subgoal = None
            return steps
        elif ability_name == "manipulation_planner":
            steps = ability_res
            self.last_local_traj = []
            return steps
        elif ability_name == "scene_description":
            self.scene_description = ability_res
        elif ability_name == "experience_summarization":
            self.core_history += f"Summarization feedback: {ability_res}\n"
            self.manipulation_subgoal = None
        else:
            print(ability_name)
            raise NotImplementedError
        
        return ["pass"]
    
    def get_core_result(self,):
        scheduler_history = self.core_history
        if self._evo_skill is not None:
            scheduler_history += self._evo_skill.scheduler_context()
        res = qwen_inference(
            self.vlm_processer, self.vlm, 
            [], 
            self.prompt_ct.format(self.task_instruction, scheduler_history),
            log_file=f"{self.save_path}/qwen_log.txt",
            role="scheduler",
            save_path=self.save_path,
        )
        append_trace(self.save_path, {
            "event": "scheduler_raw",
            "role": "scheduler",
            "raw_output": res,
            "has_query": "Query:" in res or "query:" in res,
            "has_stop": "Stop" in res,
        })
        if "Query:" not in res and "query:" in res:
            res = res.replace("query:", "Query:")
        if "Query:" in res:
            think_text = res.split("Query:")[0].split("Think:")[1].strip()
            queries_text = res.split("Query:")[1].strip()
            
            if self.core_history.strip().endswith("Grounding feedback: the target object is not found") and self.core_history.strip()[:-len("Grounding feedback: the target object is not found")].strip().endswith(queries_text):
                pass
            else:
                self.core_history += "Query: " + queries_text + "\n"
            queries = queries_text.split("\n")
            parsed_q = []
            for q in queries:
                q = q.strip()
                if not q:
                    continue
                # prefer "<n>. skill(args)" (FT Brain), fallback "skill(args)" (Base Brain)
                if ". " in q and q.split(". ")[1].strip():
                    q = q.split(". ")[1].strip()
                parsed_q.append(q)
            queries = parsed_q
            for iq, query in enumerate(queries):
                ability_name = query.split("(")[0].strip()
                args = "(".join(query.split("(")[1:])
                assert args.endswith(")"), args
                args = args[:-1]
                self.ability_buffer.append([ability_name, args])
            return True
        else:
            if "Stop" not in res:
                return False
            think_text = res.split("Stop")[0].split("Think:")[0].strip()
            return False
        
            

    def _maybe_record_skill(self, skill: str, inputs, outputs, success: bool, failure_type=None):
        sm = getattr(self, "_skill_memory", None)
        if sm is None:
            return
        try:
            sm.record(
                skill,
                inputs if isinstance(inputs, dict) else {"args": inputs},
                outputs,
                bool(success),
                failure_type=failure_type,
                episode_id=str(getattr(self, "episode_id", "") or getattr(self, "cur_episode", "")),
                step=int(getattr(self, "step_count", -1) or -1),
            )
        except Exception as e:
            # never break the agent for logging
            try:
                append_trace(self.save_path, {"event": "skill_memory_error", "error": str(e)[:300]})
            except Exception:
                pass

    def _activate_evo_intervention(self, reason, invalidate_suffix=True):
        note = f"Effect verification: {reason}\n"
        if not self.core_history.endswith(note):
            self.core_history += note
        if invalidate_suffix:
            self.ability_buffer = []
            self.ability_buffer_idx = 0
            self._evo_interrupt_actions = True
        append_trace(self.save_path, {
            "event": "evo_suffix_invalidated" if invalidate_suffix else "evo_warning",
            "reason": reason,
            "confirmed_progress": (
                list(self._evo_skill.confirmed_effects) if self._evo_skill is not None else []
            ),
            "observation_version": self.observation_version,
        })

    def consume_evo_interrupt(self):
        value = bool(self._evo_interrupt_actions)
        self._evo_interrupt_actions = False
        return value

    def should_skip_evo_action(self, action):
        if self._evo_skill is None:
            return False
        intervention = self._evo_skill.precheck_action(action)
        if intervention is None:
            return False
        append_trace(self.save_path, {
            "event": "evo_redundant_action_skipped",
            "action": action,
            "reason": intervention.reason,
        })
        if self._evo_skill.virtual_skip_feedback_enabled():
            aid = len(self.last_local_traj) // 2 + 1
            self.last_local_traj.append(f"[action {aid}] {action}")
            self.last_local_traj.append(
                f"[feedback {aid}] success (verified effect already satisfied)"
            )
        return True

    def get_ability_result(self, ability_name, args):
        if ability_name == "exploration_guidance":
            if args == self.last_to_find:
                pass
            elif self.last_to_find and args == self.last_to_find.split(" (hint:")[0]: 
                pass
            else:
                self.explored = []
            target_obj = args
            eg_backend = os.environ.get("ROBOAGENT_EG_BACKEND", "qwen").strip().lower()
            if eg_backend in ("explore_naive", "naive_explore"):
                place = explore_eg_naive(
                    target_obj, self.observed_objects_list, self.explored,
                    env_name=self.env_name, variant="naive")
                append_trace(self.save_path, {
                    "event": "ability_parsed", "role": "exploration_guidance",
                    "backend": eg_backend, "args": target_obj, "parsed": place,
                })
                return place
            if eg_backend in ("explore_aligned", "aligned_explore"):
                place = explore_eg_aligned(
                    target_obj, self.observed_objects_list, self.explored,
                    env_name=self.env_name, variant="aligned")
                append_trace(self.save_path, {
                    "event": "ability_parsed", "role": "exploration_guidance",
                    "backend": eg_backend, "args": target_obj, "parsed": place,
                })
                return place
            if eg_backend in ("explore_adapter", "adapter_explore", "explore"):
                place = explore_eg_adapter(
                    target_obj, self.observed_objects_list, self.explored,
                    env_name=self.env_name)
                append_trace(self.save_path, {
                    "event": "ability_parsed", "role": "exploration_guidance",
                    "backend": eg_backend, "args": target_obj, "parsed": place,
                })
                return place
            if eg_backend in ("eg_lora", "lora"):
                place = explore_eg_lora(
                    target_obj, self.observed_objects_list, self.explored,
                    env_name=self.env_name)
                _eg_trace_extra = {}
                if os.environ.get("ROBOAGENT_USR_CHANNEL", "0") == "1" and place:
                    try:
                        from agents.usr_sd_eg import eg_raw_to_usr
                        u = eg_raw_to_usr(place, observed_objects=self.observed_objects_list)
                        get_channel().publish("eg", u)
                        get_channel().log_decision("eg", place)
                        _eg_trace_extra["usr"] = u
                    except Exception:
                        pass
                _tr = {"event": "ability_parsed", "role": "exploration_guidance",
                       "backend": eg_backend, "args": target_obj, "parsed": place}
                _tr.update(_eg_trace_extra)
                append_trace(self.save_path, _tr)
                return place
            if eg_backend in ("validated_ft", "qwen25_7b", "qwen2.5-7b-instruct", "qwen25_7b_instruct"):
                _eg_prompt = self.prompt_eg.format(target_obj, self.observed_objects_list, self.explored)

                def _eg_qwen_infer(prompt_text: str, more_args=None):
                    kw = dict(
                        log_file=f"{self.save_path}/qwen_log.txt",
                        role="exploration_guidance",
                        save_path=self.save_path,
                    )
                    if more_args is not None:
                        kw["more_args"] = more_args
                    return qwen_inference(
                        self.vlm_processer, self.vlm,
                        [],
                        prompt_text,
                        **kw,
                    )

                if eg_backend == "validated_ft":
                    place = propose_eg_ft_qwen(
                        target_obj,
                        self.observed_objects_list,
                        self.explored,
                        _eg_prompt,
                        _eg_qwen_infer,
                        env_name=self.env_name,
                    )
                else:
                    place = propose_eg(
                        target_obj,
                        self.observed_objects_list,
                        self.explored,
                        _eg_prompt,
                        env_name=self.env_name,
                        qwen_infer=_eg_qwen_infer,
                    )
                append_trace(self.save_path, {
                    "event": "ability_parsed",
                    "role": "exploration_guidance",
                    "backend": eg_backend,
                    "args": target_obj,
                    "parsed": place,
                })
                return place

            if eg_backend == "usr":
                res = qwen_inference(
                    self.vlm_processer, self.vlm, 
                    [], 
                    self.prompt_eg.format(target_obj, self.observed_objects_list, self.explored),
                    log_file=f"{self.save_path}/qwen_log.txt",
                    role="exploration_guidance",
                    save_path=self.save_path,
                ).strip().replace("{", "").replace("}", "").replace("<", "").replace(">", "")
                egp = eg_usr_backend(res, observed_objects=self.observed_objects_list)
                get_channel().publish("eg", egp["usr"])
                get_channel().log_decision("eg", egp["text"] or "none")
                append_trace(self.save_path, {
                    "event": "ability_parsed",
                    "role": "exploration_guidance",
                    "backend": "usr",
                    "args": target_obj,
                    "parsed": egp["text"],
                    "usr": egp["usr"],
                })
                return egp["text"]
            res = qwen_inference(
                self.vlm_processer, self.vlm, 
                [], 
                self.prompt_eg.format(target_obj, self.observed_objects_list, self.explored),
                log_file=f"{self.save_path}/qwen_log.txt",
                role="exploration_guidance",
                save_path=self.save_path,
            ).strip().replace("{", "").replace("}", "").replace("<", "").replace(">", "")
            iii = 0
            while True:
                if self.env_name == "alfworld":
                    if not(res in self.explored or res.split(" ")[0] not in ["in", "on", "target"] or " ".join(res.split(" ")[1:]).lower() not in self.observed_objects_list):
                        break
                elif self.env_name == "eb-alfred":
                    if not (res in self.explored or res.split(" ")[0] not in ["in", "on", "target"] or " ".join(res.split(" ")[1:]) not in self.observed_objects_list):
                        break
                iii += 1
                more_args = {
                    "do_sample": True,
                    "temperature": 0.8+iii*0.1,
                    "top_k": 50, 
                    "top_p": 0.9,
                }
                res = qwen_inference(
                    self.vlm_processer, self.vlm, 
                    [], 
                    self.prompt_eg.format(target_obj, self.observed_objects_list, self.explored), more_args=more_args,
                    log_file=f"{self.save_path}/qwen_log.txt",
                    role="exploration_guidance",
                    save_path=self.save_path,
                ).strip().replace("{", "").replace("}", "")
                if iii > 10:
                    return None
                
            assert res not in self.explored, [self.prompt_eg.format(target_obj, self.observed_objects_list, self.explored), res]
            append_trace(self.save_path, {
                "event": "ability_parsed",
                "role": "exploration_guidance",
                "args": target_obj,
                "parsed": res,
                "retries": iii,
            })
            return res
        elif ability_name == "object_grounding":
            target_obj = args.split(" (hint")[0].split(" (except")[0]
            if self.last_goto == target_obj: # shortcut
                if not (self._evo_skill is not None and self._evo_skill.should_bypass_last_goto_shortcut()):
                    return [{"label": target_obj}]
            og_backend = os.environ.get("ROBOAGENT_OG_BACKEND", "qwen").lower()
            if og_backend == "llmdet":
                ret, meta = llmdet_ground(
                    self.cur_rgb_path,
                    target_obj,
                    last_goto=self.last_goto,
                    observed_objects=getattr(self, "observed_objects_list", None),
                )
                # Keep qwen_log format roughly compatible for offline tools.
                with open(f"{self.save_path}/qwen_log.txt", "a") as f:
                    f.write(self.cur_rgb_path + "\n")
                    f.write(self.prompt_og.format(target_obj) + "\n")
                    f.write("--------------------------------------------\n")
                    if ret is False:
                        f.write("no\n")
                    else:
                        f.write("```json\n" + str(ret) + "\n```\n")
                    f.write("\n\n=============================================\n\n")
                append_trace(self.save_path, {
                    "event": "ability_parsed",
                    "role": "object_grounding",
                    "backend": "llmdet",
                    "args": target_obj,
                    "raw_output": "no" if ret is False else ret,
                    "parsed_ok": ret is not False,
                    "parsed": ret,
                    "meta": meta,
                })
                if ret == False:
                    self.last_grounding_label = None
                else:
                    self.last_grounding_label = ret[0]["label"]
                return ret
            if og_backend in ("llmdet_qwen", "cascade"):
                base_prompt = self.prompt_og.format(target_obj)

                def _qwen_og(aug_prompt: str) -> str:
                    return qwen_inference(
                        self.vlm_processer, self.vlm,
                        [self.cur_rgb_path],
                        aug_prompt,
                        log_file=f"{self.save_path}/qwen_log.txt",
                        role="object_grounding",
                        save_path=self.save_path,
                    ).strip()

                ret, meta = llmdet_qwen_ground(
                    self.cur_rgb_path,
                    target_obj,
                    base_prompt,
                    _qwen_og,
                    last_goto=self.last_goto,
                    observed_objects=getattr(self, "observed_objects_list", None),
                )
                append_trace(self.save_path, {
                    "event": "ability_parsed",
                    "role": "object_grounding",
                    "backend": "llmdet_qwen",
                    "args": target_obj,
                    "raw_output": meta.get("qwen_raw", "no" if ret is False else ret),
                    "parsed_ok": ret is not False,
                    "parsed": ret,
                    "meta": meta,
                })
                if ret == False:
                    self.last_grounding_label = None
                else:
                    self.last_grounding_label = ret[0]["label"]
                return ret
            if og_backend in (
                "llmdet_qwen_gated",
                "gated_cascade",
                "llmdet_qwen_gated_v2",
                "gated_cascade_v2",
            ):
                base_prompt = self.prompt_og.format(target_obj)

                def _qwen_og_gated(aug_prompt: str) -> str:
                    return qwen_inference(
                        self.vlm_processer, self.vlm,
                        [self.cur_rgb_path],
                        aug_prompt,
                        log_file=f"{self.save_path}/qwen_log.txt",
                        role="object_grounding",
                        save_path=self.save_path,
                    ).strip()

                ret, meta = llmdet_qwen_gated_ground(
                    self.cur_rgb_path,
                    target_obj,
                    base_prompt,
                    _qwen_og_gated,
                    last_goto=self.last_goto,
                    observed_objects=getattr(self, "observed_objects_list", None),
                )
                append_trace(self.save_path, {
                    "event": "ability_parsed",
                    "role": "object_grounding",
                    "backend": og_backend,
                    "args": target_obj,
                    "raw_output": meta.get("qwen_raw", "no" if ret is False else ret),
                    "parsed_ok": ret is not False,
                    "parsed": ret,
                    "meta": meta,
                })
                if ret == False:
                    self.last_grounding_label = None
                else:
                    self.last_grounding_label = ret[0]["label"]
                return ret
            if og_backend in ("llmdet_qwen_gated_v2", "gated_cascade_v2"):
                base_prompt = self.prompt_og.format(target_obj)

                def _qwen_og_gated_v2(aug_prompt: str) -> str:
                    return qwen_inference(
                        self.vlm_processer, self.vlm,
                        [self.cur_rgb_path],
                        aug_prompt,
                        log_file=f"{self.save_path}/qwen_log.txt",
                        role="object_grounding",
                        save_path=self.save_path,
                    ).strip()

                ret, meta = llmdet_qwen_gated_v2_ground(
                    self.cur_rgb_path,
                    target_obj,
                    base_prompt,
                    _qwen_og_gated_v2,
                    last_goto=self.last_goto,
                    observed_objects=getattr(self, "observed_objects_list", None),
                )
                append_trace(self.save_path, {
                    "event": "ability_parsed",
                    "role": "object_grounding",
                    "backend": "llmdet_qwen_gated_v2",
                    "args": target_obj,
                    "raw_output": meta.get("qwen_raw", "no" if ret is False else ret),
                    "parsed_ok": ret is not False,
                    "parsed": ret,
                    "meta": meta,
                })
                if ret == False:
                    self.last_grounding_label = None
                else:
                    self.last_grounding_label = ret[0]["label"]
                return ret

            if og_backend in ("llmdet_qwen_usr", "usr"):
                base_prompt = self.prompt_og.format(target_obj)

                def _qwen_og_usr(aug_prompt: str) -> str:
                    return qwen_inference(
                        self.vlm_processer, self.vlm,
                        [self.cur_rgb_path],
                        aug_prompt,
                        log_file=f"{self.save_path}/qwen_log.txt",
                        role="object_grounding",
                        save_path=self.save_path,
                    ).strip()

                ret, meta = llmdet_qwen_usr_ground(
                    self.cur_rgb_path,
                    target_obj,
                    base_prompt,
                    _qwen_og_usr,
                    last_goto=self.last_goto,
                    observed_objects=getattr(self, "observed_objects_list", None),
                )
                append_trace(self.save_path, {
                    "event": "ability_parsed",
                    "role": "object_grounding",
                    "backend": og_backend,
                    "args": target_obj,
                    "raw_output": meta.get("qwen_raw", "no" if ret is False else ret),
                    "parsed_ok": ret is not False,
                    "parsed": ret,
                    "meta": meta,
                    "usr": meta.get("usr"),
                })
                if ret == False:
                    self.last_grounding_label = None
                else:
                    self.last_grounding_label = ret[0]["label"]
                if meta.get("usr"):
                    get_channel().publish("og", meta["usr"])
                    get_channel().log_decision("og", "grounded" if ret is not False else "not_found")
                return ret
            if og_backend in ("llmdet_qwen_remap", "remap_only", "llmdet_qwen_remap_only"):
                base_prompt = self.prompt_og.format(target_obj)

                def _qwen_og_remap(aug_prompt: str) -> str:
                    return qwen_inference(
                        self.vlm_processer, self.vlm,
                        [self.cur_rgb_path],
                        aug_prompt,
                        log_file=f"{self.save_path}/qwen_log.txt",
                        role="object_grounding",
                        save_path=self.save_path,
                    ).strip()

                ret, meta = llmdet_qwen_remap_ground(
                    self.cur_rgb_path,
                    target_obj,
                    base_prompt,
                    _qwen_og_remap,
                    last_goto=self.last_goto,
                    observed_objects=getattr(self, "observed_objects_list", None),
                )
                append_trace(self.save_path, {
                    "event": "ability_parsed",
                    "role": "object_grounding",
                    "backend": og_backend,
                    "args": target_obj,
                    "raw_output": meta.get("qwen_raw", "no" if ret is False else ret),
                    "parsed_ok": ret is not False,
                    "parsed": ret,
                    "meta": meta,
                    "usr": meta.get("usr"),
                })
                if ret == False:
                    self.last_grounding_label = None
                else:
                    self.last_grounding_label = ret[0]["label"]
                if os.environ.get("ROBOAGENT_USR_CHANNEL", "0") == "1":
                    try:
                        from agents.usr_og_backend import _found_usr, _no_det_usr
                        if ret is False:
                            u = _no_det_usr(meta)
                        else:
                            u = _found_usr(ret[0]["label"], meta)
                        get_channel().publish("og", u)
                        get_channel().log_decision("og", "grounded" if ret is not False else "not_found")
                        self.last_og_usr = u
                        meta["usr"] = u
                    except Exception:
                        pass
                return ret
            if og_backend in ("llmdet_qwen_aligned", "skill_alignment", "aligned"):
                base_prompt = self.prompt_og.format(target_obj)

                def _qwen_og_aligned(aug_prompt: str) -> str:
                    return qwen_inference(
                        self.vlm_processer, self.vlm,
                        [self.cur_rgb_path],
                        aug_prompt,
                        log_file=f"{self.save_path}/qwen_log.txt",
                        role="object_grounding",
                        save_path=self.save_path,
                    ).strip()

                ret, meta = llmdet_qwen_aligned_ground(
                    self.cur_rgb_path,
                    target_obj,
                    base_prompt,
                    _qwen_og_aligned,
                    last_goto=self.last_goto,
                    observed_objects=getattr(self, "observed_objects_list", None),
                    env_name=self.env_name,
                )
                append_trace(self.save_path, {
                    "event": "ability_parsed",
                    "role": "object_grounding",
                    "backend": og_backend,
                    "args": target_obj,
                    "raw_output": meta.get("qwen_raw", "no" if ret is False else ret),
                    "parsed_ok": ret is not False,
                    "parsed": ret,
                    "meta": meta,
                })
                if ret == False:
                    self.last_grounding_label = None
                else:
                    self.last_grounding_label = ret[0]["label"]
                return ret
            if og_backend in ("naive_detector", "naive"):
                ret, meta = naive_detector_ground(
                    self.cur_rgb_path,
                    target_obj,
                    last_goto=self.last_goto,
                    observed_objects=getattr(self, "observed_objects_list", None),
                    env_name=self.env_name,
                )
                append_trace(self.save_path, {
                    "event": "ability_parsed",
                    "role": "object_grounding",
                    "backend": og_backend,
                    "args": target_obj,
                    "raw_output": "no" if ret is False else ret,
                    "parsed_ok": ret is not False,
                    "parsed": ret,
                    "meta": meta,
                })
                if ret == False:
                    self.last_grounding_label = None
                else:
                    self.last_grounding_label = ret[0]["label"]
                return ret
            res = qwen_inference(
                self.vlm_processer, self.vlm, 
                [self.cur_rgb_path], 
                self.prompt_og.format(target_obj),
                log_file=f"{self.save_path}/qwen_log.txt",
                role="object_grounding",
                save_path=self.save_path,
            ).strip()
            assert (res.startswith("```json") and res.endswith("```")) or res.lower() == "no", res
            ret = parse_og_response(res)
            append_trace(self.save_path, {
                "event": "ability_parsed",
                "role": "object_grounding",
                "backend": "qwen",
                "args": target_obj,
                "raw_output": res,
                "parsed_ok": ret is not False,
                "parsed": ret,
            })
            if ret == False:
                self.last_grounding_label = None
            else:
                self.last_grounding_label = ret[0]["label"]
            return ret 
        elif ability_name == "exploration_planner":
            if self.exploration_subgoal is None:
                self.ability_buffer = []
                self.ability_buffer_idx = 0
                return ["pass"]
            res = qwen_inference(
                self.vlm_processer, self.vlm, 
                [], 
                self.prompt_lpe.format(self.exploration_subgoal),
                log_file=f"{self.save_path}/qwen_log.txt",
                role="exploration_planner",
                save_path=self.save_path,
            ).strip()
            assert res.startswith("[") and res.endswith("]"), res
            steps = res[1:-1].split(",")
            assert len(steps), res
            steps = [x.strip() for x in steps]
            append_trace(self.save_path, {
                "event": "ability_parsed",
                "role": "exploration_planner",
                "args": self.exploration_subgoal,
                "raw_output": res,
                "parsed": steps,
            })
            return steps
        elif ability_name == "manipulation_planner":
            manipulation_subgoal = args
            self.manipulation_subgoal = manipulation_subgoal
            res = qwen_inference(
                self.vlm_processer, self.vlm, 
                [], 
                self.prompt_lpm.format(self.invent, self.last_goto, self.scene_description, manipulation_subgoal),
                log_file=f"{self.save_path}/qwen_log.txt",
                role="manipulation_planner",
                save_path=self.save_path,
            ).strip()
            assert res.startswith("[") and res.endswith("]"), res
            steps = res[1:-1].split(",")
            assert len(steps), res
            steps = [x.strip() for x in steps]
            append_trace(self.save_path, {
                "event": "ability_parsed",
                "role": "manipulation_planner",
                "args": manipulation_subgoal,
                "raw_output": res,
                "parsed": steps,
            })
            return steps
        elif ability_name == "scene_description":
            if self.invent != "nothing":
                invent_des = f" Note that the agent is holding {self.invent}, which is shown at the bottom of the image. You can ignore it in your description. "
            else:
                invent_des = ""
            # assert self.last_grounding_label is not None
            if self.last_grounding_label is None:
                return ""
            sd_comp = os.environ.get("ROBOAGENT_SD_COMP", "").strip().lower()
            sd_target_label = self.last_grounding_label
            if sd_comp == "none":
                sd_target_label = None
            elif sd_comp == "raw":
                sd_target_label = getattr(self, "last_det_query", None) or self.last_grounding_label
            elif sd_comp == "usr":
                sd_target_label = getattr(self, "last_usr_class", None) or self.last_grounding_label
            elif sd_comp == "oracle":
                sd_target_label = self.last_grounding_label
            sd_backend = os.environ.get("ROBOAGENT_SD_BACKEND", "qwen").lower()
            if sd_backend == "usr":
                usr_ch = get_channel()
                if os.environ.get("ROBOAGENT_USR_CHANNEL", "0") == "1" and usr_ch.has("og"):
                    sd_target_label = usr_ch.get_field("og", "object.class") or sd_target_label
                if sd_target_label:
                    _sd_prompt = self.prompt_sd.format(sd_target_label) + invent_des
                else:
                    _sd_prompt = ("<image>\nThis is an egocentric image observed by a robotic "
                                  "household agent. Please describe the scene.") + invent_des
                res = qwen_inference(
                    self.vlm_processer, self.vlm, 
                    [self.cur_rgb_path], 
                    _sd_prompt,
                    max_new_tokens=512,
                    log_file=f"{self.save_path}/qwen_log.txt",
                    role="scene_description",
                    save_path=self.save_path,
                ).strip()
                sdp = sd_usr_backend(res, target=sd_target_label)
                usr_ch.publish("sd", sdp["usr"])
                usr_ch.log_decision("sd", "described")
                append_trace(self.save_path, {
                    "event": "ability_parsed",
                    "role": "scene_description",
                    "backend": "usr",
                    "comp_mode": sd_comp,
                    "args": sd_target_label,
                    "raw_output": res,
                    "parsed": sdp["text"],
                    "usr": sdp["usr"],
                })
                return sdp["text"]

            if sd_backend in ("florence2_qwen_verify", "florence2_verify", "florence2_gated"):
                def _sd_qwen_infer(prompt_text: str, more_args=None):
                    kw = dict(
                        log_file=f"{self.save_path}/qwen_log.txt",
                        role="scene_description",
                        save_path=self.save_path,
                    )
                    if more_args is not None:
                        kw["more_args"] = more_args
                    return qwen_inference(
                        self.vlm_processer, self.vlm,
                        [self.cur_rgb_path],
                        prompt_text,
                        **kw,
                    )
                desc = florence_qwen_verify_describe(
                    self.cur_rgb_path,
                    self.last_grounding_label,
                    self.invent != "nothing",
                    self.prompt_sd,
                    _sd_qwen_infer,
                    save_path=self.save_path,
                )
                append_trace(self.save_path, {
                    "event": "ability_parsed",
                    "role": "scene_description",
                    "backend": sd_backend,
                    "args": args,
                    "parsed": desc,
                })
                return desc
            if sd_backend in ("florence2_naive", "florence2"):
                res = describe_naive(
                    self.cur_rgb_path,
                    self.last_grounding_label,
                    invent=self.invent,
                )
                log_sd_res = res
            elif sd_backend in ("florence2_adapter", "florence2_cascade"):
                def _qwen_sd_infer(prompt):
                    return qwen_inference(
                        self.vlm_processer, self.vlm,
                        [self.cur_rgb_path],
                        prompt,
                        max_new_tokens=512,
                        log_file=f"{self.save_path}/qwen_log.txt",
                        role="scene_description",
                        save_path=self.save_path,
                    ).strip()
                res = describe_adapter(
                    self.cur_rgb_path,
                    self.last_grounding_label,
                    _qwen_sd_infer,
                    invent=self.invent,
                )
                log_sd_res = res
            else:
                if sd_target_label:
                    _sd_prompt = self.prompt_sd.format(sd_target_label) + invent_des
                else:
                    _sd_prompt = ("<image>\nThis is an egocentric image observed by a robotic "
                                  "household agent. Please describe the scene.") + invent_des
                res = qwen_inference(
                    self.vlm_processer, self.vlm, 
                    [self.cur_rgb_path], 
                    _sd_prompt,
                    max_new_tokens=512,
                    log_file=f"{self.save_path}/qwen_log.txt",
                    role="scene_description",
                    save_path=self.save_path,
                ).strip()
                log_sd_res = res


            append_trace(self.save_path, {
                "event": "ability_parsed",
                "role": "scene_description",
                "args": self.last_grounding_label,
                "raw_output": res,
                "parsed": res,
            })
            return res
        elif ability_name == "experience_summarization":
            assert len(self.last_local_traj)
            assert "none" not in "\n".join(self.last_local_traj), self.last_local_traj
            res = qwen_inference(
                self.vlm_processer, self.vlm, 
                [self.cur_rgb_path], 
                self.prompt_es.format(self.manipulation_subgoal, "\n".join(self.last_local_traj)),
                log_file=f"{self.save_path}/qwen_log.txt",
                role="experience_summarization",
                save_path=self.save_path,
            ).strip()
            append_trace(self.save_path, {
                "event": "ability_parsed",
                "role": "experience_summarization",
                "args": self.manipulation_subgoal,
                "raw_output": res,
                "parsed": res,
            })
            return res
        else:
            print(ability_name)
            raise NotImplementedError
        
