import sys
import os

from agents.agent import Agent
from alfworld.agents.environment.alfred_thor_env import AlfredThorEnv

class AWRunner():
    def __init__(self, env: AlfredThorEnv, agent: Agent):
        self.env = env
        self.agent = agent
        self.env_step_id = 0
        
        self.last_recep = None
        self.last_info = {}
        
    def run(self, ):
        steps = 0
        max_steps = int(os.environ.get("ROBOAGENT_MAX_AW_STEPS", "0") or 0)
        hit_step_cap = False
        done = False
        while True:
            actions, _ = self.agent.get_qwen_action()
            
            for action in actions:
                sys.stdout.flush()
                if action == "fail":
                    if steps == 0:
                        gcr = 0
                        break
                    gcr = self.last_info["goal_condition_success_rate"][0]
                    break
                
                if action != "pass":
                    if self.agent.should_skip_evo_action(action):
                        continue
                    print(f"--------------------[ENV STEP {steps}]: {action}\n")
                    steps += 1
                    if max_steps and steps >= max_steps:
                        gcr = self.last_info.get("goal_condition_success_rate", [0])[0] if self.last_info else 0
                        hit_step_cap = True
                        break
                done, info = self.execute(action)
                if done:
                    gcr = info["goal_condition_success_rate"][0]
                    break
                if self.agent.consume_evo_interrupt():
                    break
            if done or action == "fail" or hit_step_cap:
                break
        return gcr
       
    
    def env_step(self, action, process_obs=True, img_only=False):
        _, scores, dones, infos = self.env.step([action.lower()])
        if process_obs:
            self.agent.process_observation(
                self.env.get_frames()[0][:, :, ::-1],
                self.env_step_id,
            )
        self.env_step_id += 1
        return _, scores, dones, infos
        
    def execute(self, action_str):
        if action_str in ["pass", "examine"]:
            action_str = "do nothing"
            _ = [""]
            done = False
            info = {'last_action_success': 1}
        else:
            _, scores, dones, info = self.env_step(action_str)
            done = dones[0]
            gcr = info.get("goal_condition_success_rate", [0])[0]
            self.agent.process_feedback(
                "Nothing happen" not in _[0],
                action_str.replace(" None", ""),
                gcr=gcr,
            )
            self.last_info = info
        return done, info
        