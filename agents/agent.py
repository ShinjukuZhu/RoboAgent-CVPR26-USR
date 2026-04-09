import torch
import numpy as np
import cv2
from collections import defaultdict

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from agents.qwen import inference as qwen_inference
from peft import PeftModel

from agents.prompt import *
import copy

SIM_CLS = {
    "alarm clock": "AlarmClock",
"apple": "Apple",
"sliced apple": "AppleSliced",
"armchair": "ArmChair",
"baseball bat": "BaseballBat",
"basketball": "BasketBall",
"bathtub": "Bathtub",
"bed": "Bed",
"book": "Book",
"bowl": "Bowl",
"box": "Box",
"bread": "Bread",
"sliced bread": "BreadSliced",
"butterknife": "ButterKnife",
"cd": "CD",
"cabinet": "Cabinet",
"candle": "Candle",
"cart": "Cart",
"cell phone": "CellPhone",
"cloth": "Cloth",
"coffee machine": "CoffeeMachine",
"coffee table": "CoffeeTable",
"countertop": "CounterTop",
"credit card": "CreditCard",
"cup": "Cup",
"desk": "Desk",
"desk lamp": "DeskLamp",
"dining table": "DiningTable",
"sponge": "DishSponge",
"drawer": "Drawer",
"dresser": "Dresser",
"egg": "Egg",
"floor lamp": "FloorLamp",
"fork": "Fork",
"fridge": "Fridge",
"garbage can": "GarbageCan",
"glass bottle": "Glassbottle",
"towel": "HandTowel",
"kettle": "Kettle",
"key": "KeyChain",
"knife": "Knife",
"ladle": "Ladle",
"laptop": "Laptop",
"lettuce": "Lettuce",
"sliced lettuce": "LettuceSliced",
"microwave": "Microwave",
"mug": "Mug",
"newspaper": "Newspaper",
"ottoman": "Ottoman",
"pan": "Pan",
"pen": "Pen",
"pencil": "Pencil",
"pepper shaker": "PepperShaker",
"pillow": "Pillow",
"plate": "Plate",
"plunger": "Plunger",
"pot": "Pot",
"potato": "Potato",
"sliced potato": "PotatoSliced",
"remote control": "RemoteControl",
"safe": "Safe",
"salt shaker": "SaltShaker",
"shelf": "Shelf",
"side table": "SideTable",
"sink": "Sink",
"soap bar": "SoapBar",
"soap bottle": "SoapBottle",
"sofa": "Sofa",
"spatula": "Spatula",
"spoon": "Spoon",
"spray bottle": "SprayBottle",
"statue": "Statue",
"stove burner": "StoveBurner",
"tennis racket": "TennisRacket",
"tissue box": "TissueBox",
"toilet": "Toilet",
"toilet paper": "ToiletPaper",
"toilet paper hanger": "ToiletPaperHanger",
"tomato": "Tomato",
"sliced tomato": "TomatoSliced",
"vase": "Vase",
"watch": "Watch",
"watering can": "WateringCan",
"wine bottle": "WineBottle",
}


class Agent(object):
    def __init__(self, vlm_model_path="CKPT/Qwen2.5VL_7B-Instruct", lora_path=None):
        self.vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(vlm_model_path, torch_dtype=torch.bfloat16, device_map="auto")
        self.vlm_processer = AutoProcessor.from_pretrained(vlm_model_path)
        if lora_path:
            self.vlm_ori = copy.deepcopy(self.vlm)
            self.vlm = PeftModel.from_pretrained(self.vlm, lora_path)
            print("Merging LoRA weights into base model...")
            self.vlm = self.vlm.merge_and_unload()
        else:
            self.vlm_ori = self.vlm
    
    def reset(self, save_path, obj_list):
        self.last_goto = None
        self.save_path = save_path
        self.save_i = 0
        
        self.observed_objects_list = [x for x in sorted(obj_list)]
        self.core_history = ""
        self.explored = []
        self.invent = "nothing"
        self.last_local_traj = []
        self.ability_buffer = []
        self.ability_buffer_idx = 0
        self.last_to_find = None
        
        with open(f"{self.save_path}/qwen_log.txt", "w") as f:
            f.write("BEGIN!!!\n")
        
    def process_observation(self, rgb, env_step_id):
        cv2.imwrite(f"{self.save_path}/step_{env_step_id}.png", rgb[:, :, ::-1])
        self.cur_rgb_path = f"{self.save_path}/step_{env_step_id}.png"
        
        
    def process_task(self, task_info, task_instruction):
        print("[TASK] ", task_instruction)
        self.task_instruction = task_instruction
        return
    
    def process_feedback(self, message, last_action):
        assert last_action not in ["examine", "pass", "do nothing"]
        assert last_action.split(" ")[0] in ["take", "open", "close", "put", "slice", "heat", "cool", "clean", "go", "use"], last_action
        self.last_action = last_action
        aid = len(self.last_local_traj) // 2 + 1
        self.last_local_traj.append(f"[action {aid}] {last_action}")
        self.last_local_traj.append(f"[feedback {aid}] {'success' if message else 'failure'}")
        
        self.scene_description = ""
        if message:
            if last_action.startswith("take "):
                assert self.invent == "nothing"
                self.invent = last_action.split("take ")[1].split(" from")[0]
            elif last_action.startswith("put "):
                assert self.invent != "nothing"
                self.invent = "nothing"
            elif last_action.startswith("go to"):
                self.last_goto = last_action.split("go to ")[1]
        return
     
    def get_qwen_action(self, ):
        if self.ability_buffer_idx >= len(self.ability_buffer):
            self.ability_buffer = []
            ret = self.get_core_result()
            if not ret:
                return ["fail"]
            assert len(self.ability_buffer)
            self.ability_buffer_idx = 0
        
        ability_name, ability_args = self.ability_buffer[self.ability_buffer_idx]
        ability_res = self.get_ability_result(ability_name, ability_args)
        self.ability_buffer_idx += 1
        if ability_name == "exploration_guidance":
            self.last_to_find = ability_args    
            
            place = ability_res
            if place is None:
                return ["fail"]
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
                ocls = ocls.replace("sliced ", "")
                assert self.last_goto is not None
                if not self.core_history.strip().endswith("Grounding feedback: the target object is not found"):
                    self.core_history += f"Grounding feedback: the target object ({ocls}) is found at {self.last_goto}\n"
                else:
                    self.core_history = self.core_history.strip()[:-len("Grounding feedback: the target object is not found")] + f"Grounding feedback: the target object ({ocls}) is found at {self.last_goto}\n"
        elif ability_name == "exploration_planner":
            steps = ability_res
            self.last_local_traj = []
            self.exploration_subgoal = None
            assert steps[0].startswith("go to "), steps
            return steps
        elif ability_name == "manipulation_planner":
            steps = ability_res
            self.last_local_traj = []
            return steps
        elif ability_name == "scene_description":
            self.scene_description = ability_res
        elif ability_name == "experience_summarization":
            self.core_history += f"Summarization feedback: {ability_res}\n"
        else:
            print(ability_name)
            raise NotImplementedError
        
        return ["pass"]
    
    def get_core_result(self,):
        res = qwen_inference(
            self.vlm_processer, self.vlm, 
            [], 
            prompt_ct.format(self.task_instruction, self.core_history),
            log_file=f"{self.save_path}/qwen_log.txt"
        )
        assert res.startswith("Think:"), res
        if "Query:" in res:
            think_text = res.split("Query:")[0].split("Think:")[1].strip()
            queries_text = res.split("Query:")[1].strip()
            
            if self.core_history.strip().endswith("Grounding feedback: the target object is not found") and self.core_history.strip()[:-len("Grounding feedback: the target object is not found")].strip().endswith(queries_text):
                pass
            else:
                self.core_history += "Query: " + queries_text + "\n"
            queries = queries_text.split("\n")
            queries = [q.split(". ")[1].strip() for q in queries]
            for iq, query in enumerate(queries):
                ability_name = query.split("(")[0]
                args = "(".join(query.split("(")[1:])
                assert args.endswith(")"), args
                args = args[:-1]
                self.ability_buffer.append([ability_name, args])
            return True
        else:
            if "Stop" not in res:
                # print("BAD! NO QUERY NO STOP!")
                return False
            assert "Stop" in res, res
            think_text = res.split("Stop")[0].split("Think:")[0].strip()
            return False
        
            
    def get_ability_result(self, ability_name, args):
        if ability_name == "exploration_guidance":
            if args == self.last_to_find:
                pass
            elif self.last_to_find and args == self.last_to_find.split(" (hint:")[0]: 
                pass
            else:
                self.explored = []
            target_obj = args
            res = qwen_inference(
                self.vlm_processer, self.vlm, 
                [], 
                prompt_eg.format(target_obj, self.observed_objects_list, self.explored),
                log_file=f"{self.save_path}/qwen_log.txt"
            ).strip().replace("{", "").replace("}", "").replace("<", "").replace(">", "")
            iii = 0
            while res in self.explored or res.split(" ")[0] not in ["in", "on", "target"] or " ".join(res.split(" ")[1:]).lower() not in self.observed_objects_list:
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
                    prompt_eg.format(target_obj, self.observed_objects_list, self.explored), more_args=more_args,
                    log_file=f"{self.save_path}/qwen_log.txt"
                ).strip().replace("{", "").replace("}", "")
                if iii > 10:
                    return None
                
            assert res not in self.explored, [prompt_eg.format(target_obj, self.observed_objects_list, self.explored), res]
            return res
        elif ability_name == "object_grounding":
            target_obj = args.split(" (hint")[0].split(" (except")[0]
            if self.last_goto == target_obj: # shortcut!
                return [{"label": target_obj}]
            res = qwen_inference(
                self.vlm_processer, self.vlm_ori if target_obj in SIM_CLS else self.vlm, 
                [self.cur_rgb_path], 
                prompt_og.format(target_obj),
                log_file=f"{self.save_path}/qwen_log.txt"
            ).strip()
            assert (res.startswith("```json") and res.endswith("```")) or res.lower() == "no", res
            ret = eval(res[8:-3].strip()) if (res.startswith("```json") and res.endswith("```")) else False
            if ret == False:
                self.last_grounding_label = None
            else:
                self.last_grounding_label = ret[0]["label"]
                assert self.last_grounding_label in SIM_CLS or self.last_grounding_label in ["remote", "ice cream ladle", "cushion", "soup spoon", "sprayer", "soap", "ink pen"], self.last_grounding_label
            return ret 
        elif ability_name == "exploration_planner":
            assert self.exploration_subgoal
            res = qwen_inference(
                self.vlm_processer, self.vlm, 
                [], 
                prompt_lpe.format(self.exploration_subgoal),
                log_file=f"{self.save_path}/qwen_log.txt"
            ).strip()
            assert res.startswith("[") and res.endswith("]"), res
            steps = res[1:-1].split(",")
            assert len(steps), res
            return [x.strip() for x in steps]
        elif ability_name == "manipulation_planner":
            manipulation_subgoal = args
            self.manipulation_subgoal = manipulation_subgoal
            res = qwen_inference(
                self.vlm_processer, self.vlm, 
                [], 
                prompt_lpm.format(self.invent, self.last_goto, self.scene_description, manipulation_subgoal),
                log_file=f"{self.save_path}/qwen_log.txt"
            ).strip()
            assert res.startswith("[") and res.endswith("]"), res
            steps = res[1:-1].split(",")
            assert len(steps), res
            return [x.strip() for x in steps]
        elif ability_name == "scene_description":
            if self.invent != "nothing":
                invent_des = f" Note that the agent is holding {self.invent}, which is shown at the bottom of the image. You can ignore it in your description. "
            else:
                invent_des = ""
            assert self.last_grounding_label is not None
            res = qwen_inference(
                self.vlm_processer, self.vlm, 
                [self.cur_rgb_path], 
                prompt_sd.format(self.last_grounding_label) + invent_des,
                max_new_tokens=512,
                log_file=f"{self.save_path}/qwen_log.txt"
            ).strip()
            return res
        elif ability_name == "experience_summarization":
            assert len(self.last_local_traj)
            assert "none" not in "\n".join(self.last_local_traj), self.last_local_traj
            res = qwen_inference(
                self.vlm_processer, self.vlm, 
                [self.cur_rgb_path], 
                prompt_es.format(self.manipulation_subgoal, "\n".join(self.last_local_traj)),
                log_file=f"{self.save_path}/qwen_log.txt"
            ).strip()
            return res
        else:
            print(ability_name)
            raise NotImplementedError
        
