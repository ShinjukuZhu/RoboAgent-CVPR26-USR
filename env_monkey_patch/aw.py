import random
import os
import json
import traceback
from threading import Thread
from alfworld.agents.environment.alfred_thor_env import TASK_TYPES
from alfworld.env.thor_env import ThorEnv

import alfworld.agents
from alfworld.agents.utils.misc import get_templated_task_desc
from alfworld.env.thor_env import ThorEnv
from alfworld.agents.expert import HandCodedThorAgent, HandCodedAgentTimeout
from alfworld.agents.detector.mrcnn import load_pretrained_model
from alfworld.agents.controller import OracleAgent, OracleAStarAgent, MaskRCNNAgent, MaskRCNNAStarAgent

def get_env_paths_solvable(self):
    self.json_file_list = []

    if self.train_eval == "train":
        data_path = os.path.expandvars(self.config['dataset']['data_path'])
    elif self.train_eval == "eval_in_distribution":
        data_path = os.path.expandvars(self.config['dataset']['eval_id_data_path'])
    elif self.train_eval == "eval_out_of_distribution":
        data_path = os.path.expandvars(self.config['dataset']['eval_ood_data_path'])
    else:
        raise Exception("Invalid split. Must be either train or eval")

    # get task types
    assert len(self.config['env']['task_types']) > 0
    task_types = []
    for tt_id in self.config['env']['task_types']:
        if tt_id in TASK_TYPES:
            task_types.append(TASK_TYPES[tt_id])

    for root, dirs, files in os.walk(data_path, topdown=False):
        if 'traj_data.json' in files:
            # Skip movable and slice objects object tasks
            if 'movable' in root or 'Sliced' in root:
                continue

            # File paths
            json_path = os.path.join(root, 'traj_data.json')
            game_file_path = os.path.join(root, "game.tw-pddl")

            # Load trajectory file
            with open(json_path, 'r') as f:
                traj_data = json.load(f)

            # Check for any task_type constraints
            if not traj_data['task_type'] in task_types:
                continue

            # self.json_file_list.append(json_path)

            # # Only add solvable games
            if os.path.exists(game_file_path):
                with open(game_file_path, 'r') as f:
                    gamedata = json.load(f)
            
                if 'solvable' in gamedata and gamedata['solvable']:
                    self.json_file_list.append(json_path)

    print("Overall we have %s games..." % (str(len(self.json_file_list))))
    self.num_games = len(self.json_file_list)

    if self.train_eval == "train":
        num_train_games = self.config['dataset']['num_train_games'] if self.config['dataset']['num_train_games'] > 0 else len(self.json_file_list)
        self.json_file_list = self.json_file_list[:num_train_games]
        self.num_games = len(self.json_file_list)
        print("Training with %d games" % (len(self.json_file_list)))
    else:
        num_eval_games = self.config['dataset']['num_eval_games'] if self.config['dataset']['num_eval_games'] > 0 else len(self.json_file_list)
        self.json_file_list = self.json_file_list[:num_eval_games]
        self.num_games = len(self.json_file_list)
        print("Evaluating with %d games" % (len(self.json_file_list)))

def get_env_paths_allTrain(self):
    self.json_file_list = []

    if self.train_eval == "train":
        data_path = os.path.expandvars(self.config['dataset']['data_path'])
    elif self.train_eval == "eval_in_distribution":
        data_path = os.path.expandvars(self.config['dataset']['eval_id_data_path'])
    elif self.train_eval == "eval_out_of_distribution":
        data_path = os.path.expandvars(self.config['dataset']['eval_ood_data_path'])
    else:
        raise Exception("Invalid split. Must be either train or eval")

    # get task types
    assert len(self.config['env']['task_types']) > 0
    task_types = []
    for tt_id in self.config['env']['task_types']:
        if tt_id in TASK_TYPES:
            task_types.append(TASK_TYPES[tt_id])

    for root, dirs, files in os.walk(data_path, topdown=False):
        if 'traj_data.json' in files:
            # print("GOOD", root)
            # # Skip movable and slice objects object tasks
            # if 'movable' in root or 'Sliced' in root:
            #     continue

            # File paths
            json_path = os.path.join(root, 'traj_data.json')
            game_file_path = os.path.join(root, "game.tw-pddl")

            # Load trajectory file
            with open(json_path, 'r') as f:
                traj_data = json.load(f)

            # Check for any task_type constraints
            # if not traj_data['task_type'] in task_types:
            #     continue

            self.json_file_list.append(json_path)

            # # # Only add solvable games
            # if os.path.exists(game_file_path):
            #     with open(game_file_path, 'r') as f:
            #         gamedata = json.load(f)
            
            #     if 'solvable' in gamedata and gamedata['solvable']:
            #         self.json_file_list.append(json_path)
        else:
            if "trial" in root:
                print("BAD", root)

    print("Overall we have %s games..." % (str(len(self.json_file_list))))
    self.num_games = len(self.json_file_list)

    if self.train_eval == "train":
        num_train_games = self.config['dataset']['num_train_games'] if self.config['dataset']['num_train_games'] > 0 else len(self.json_file_list)
        self.json_file_list = self.json_file_list[:num_train_games]
        self.num_games = len(self.json_file_list)
        print("Training with %d games" % (len(self.json_file_list)))
    else:
        num_eval_games = self.config['dataset']['num_eval_games'] if self.config['dataset']['num_eval_games'] > 0 else len(self.json_file_list)
        self.json_file_list = self.json_file_list[:num_eval_games]
        self.num_games = len(self.json_file_list)
        print("Evaluating with %d games" % (len(self.json_file_list)))

def env_reset_with_idx(self, i=None):
    # set tasks
    batch_size = self.batch_size
    # reset envs
    
    if i is not None:
        assert batch_size == 1
        tasks = [self.json_file_list[i]]
    else:
        if self.train_eval == 'train':
            tasks = random.sample(self.json_file_list, k=batch_size)
        else:
            if len(self.json_file_list)-batch_size > batch_size:
                tasks = [self.json_file_list.pop(random.randrange(len(self.json_file_list))) for _ in range(batch_size)]
            else:
                tasks = random.sample(self.json_file_list, k=batch_size)
                self.get_env_paths()

    for n in range(batch_size):
        self.action_queues[n].put((None, True, tasks[n]))

    obs, dones, infos = self.wait_and_get_info()
    return obs, infos

def modified_oracle_step(self, action_str):
    event = None
    self.feedback = "Nothing happens."

    try:
        events = None
        cmd = self.parse_command(action_str)
        if cmd['action'] == self.Action.GOTO:
            target = cmd['tar']
            recep = self.get_object(target, self.receptacles)
            if recep and recep['num_id'] == self.curr_recep:
                return self.feedback, [""]
            self.curr_loc = recep['locs']
            event = self.navigate(self.curr_loc)
            self.curr_recep = recep['num_id']
            self.visible_objects, self.feedback = self.print_frame(recep, self.curr_loc)

            # feedback conditions
            loc_id = list(self.receptacles.keys()).index(recep['object_id'])
            loc_feedback = "You arrive at loc %s. " % loc_id
            state_feedback = "The {} is {}. ".format(self.curr_recep, "closed" if recep['closed'] else "open") if recep['closed'] is not None else ""
            loc_state_feedback = loc_feedback + state_feedback
            self.feedback = loc_state_feedback + self.feedback if "closed" not in state_feedback else loc_state_feedback
            self.frame_desc = str(self.feedback)

        elif cmd['action'] == self.Action.PICK:
            obj, rel, tar = cmd['obj'], cmd['rel'], cmd['tar']
            if obj in self.visible_objects:
                object = self.get_object(obj, self.objects)
                event = self.env.step({'action': "PickupObject",
                                        'objectId': object['object_id'],
                                        'forceAction': True})

                if event.metadata['lastActionSuccess']:
                    self.inventory.append(object['num_id'])
                    self.feedback = "You pick up the %s from the %s." % (obj, tar)

        elif cmd['action'] == self.Action.PUT:
            obj, rel, tar = cmd['obj'], cmd['rel'], cmd['tar']
            recep = self.get_object(tar, self.receptacles)
            if recep is None:
                recep = self.get_object(tar, self.objects)
            event = self.env.step({'action': "PutObject",
                                    'objectId': self.env.last_event.metadata['inventoryObjects'][0]['objectId'],
                                    'receptacleObjectId': recep['object_id'],
                                    'forceAction': True})
            if event.metadata['lastActionSuccess']:
                self.inventory.pop()
                self.feedback = "You put the %s %s the %s." % (obj, rel, tar)

        elif cmd['action'] == self.Action.OPEN:
            target = cmd['tar']
            recep = self.get_object(target, self.receptacles)
            event = self.env.step({'action': "OpenObject",
                                    'objectId': recep['object_id'],
                                    'forceAction': True})
            self.receptacles[recep['object_id']]['closed'] = False
            self.visible_objects, self.feedback = self.print_frame(recep, self.curr_loc)
            action_feedback = "You open the %s. The %s is open. " % (target, target)
            self.feedback = action_feedback + self.feedback.replace("On the %s" % target, "In it")
            self.frame_desc = str(self.feedback)

        elif cmd['action'] == self.Action.CLOSE:
            target = cmd['tar']
            recep = self.get_object(target, self.receptacles)
            event = self.env.step({'action': "CloseObject",
                                    'objectId': recep['object_id'],
                                    'forceAction': True})
            self.receptacles[recep['object_id']]['closed'] = True
            self.feedback = "You close the %s." % target

        elif cmd['action'] == self.Action.TOGGLE:
            target = cmd['tar']
            obj = self.get_object(target, self.objects)
            event = self.env.step({'action': "ToggleObjectOn",
                                    'objectId': obj['object_id'],
                                    'forceAction': True})
            self.feedback = "You turn on the %s." % target

        elif cmd['action'] == self.Action.HEAT:
            obj, rel, tar = cmd['obj'], cmd['rel'], cmd['tar']
            obj_id = self.env.last_event.metadata['inventoryObjects'][0]['objectId']
            recep = self.get_object(tar, self.receptacles)

            # open the microwave, heat the object, take the object, close the microwave
            events = []
            events.append(self.env.step({'action': 'OpenObject', 'objectId': recep['object_id'], 'forceAction': True}))
            events.append(self.env.step({'action': 'PutObject', 'objectId': obj_id, 'receptacleObjectId': recep['object_id'], 'forceAction': True}))
            events.append(self.env.step({'action': 'CloseObject', 'objectId': recep['object_id'], 'forceAction': True}))
            events.append(self.env.step({'action': 'ToggleObjectOn', 'objectId': recep['object_id'], 'forceAction': True}))
            events.append(self.env.step({'action': 'Pass'}))
            events.append(self.env.step({'action': 'ToggleObjectOff', 'objectId': recep['object_id'], 'forceAction': True}))
            events.append(self.env.step({'action': 'OpenObject', 'objectId': recep['object_id'], 'forceAction': True}))
            events.append(self.env.step({'action': 'PickupObject', 'objectId': obj_id, 'forceAction': True}))
            events.append(self.env.step({'action': 'CloseObject', 'objectId': recep['object_id'], 'forceAction': True}))

            if all(e.metadata['lastActionSuccess'] for e in events): #  and self.curr_recep == tar
                self.feedback = "You heat the %s using the %s." % (obj, tar)

        elif cmd['action'] == self.Action.CLEAN:
            obj, rel, tar = cmd['obj'], cmd['rel'], cmd['tar']
            object = self.env.last_event.metadata['inventoryObjects'][0]
            sink = self.get_obj_cls_from_metadata('BathtubBasin' if "bathtubbasin" in tar else "SinkBasin")
            faucet = self.get_obj_cls_from_metadata('Faucet')

            # put the object in the sink, turn on the faucet, turn off the faucet, pickup the object
            events = []
            events.append(self.env.step({'action': 'PutObject', 'objectId': object['objectId'], 'receptacleObjectId': sink['objectId'], 'forceAction': True}))
            events.append(self.env.step({'action': 'ToggleObjectOn', 'objectId': faucet['objectId'], 'forceAction': True}))
            events.append(self.env.step({'action': 'Pass'}))
            events.append(self.env.step({'action': 'ToggleObjectOff', 'objectId': faucet['objectId'], 'forceAction': True}))
            events.append(self.env.step({'action': 'PickupObject', 'objectId': object['objectId'], 'forceAction': True}))

            if all(e.metadata['lastActionSuccess'] for e in events): #  and self.curr_recep == tar
                self.feedback = "You clean the %s using the %s." % (obj, tar)

        elif cmd['action'] == self.Action.COOL:
            obj, rel, tar = cmd['obj'], cmd['rel'], cmd['tar']
            object = self.env.last_event.metadata['inventoryObjects'][0]
            fridge = self.get_obj_cls_from_metadata('Fridge')

            # open the fridge, put the object inside, close the fridge, open the fridge, pickup the object
            events = []
            events.append(self.env.step({'action': 'OpenObject', 'objectId': fridge['objectId'], 'forceAction': True}))
            events.append(self.env.step({'action': 'PutObject', 'objectId': object['objectId'], 'receptacleObjectId': fridge['objectId'], 'forceAction': True}))
            events.append(self.env.step({'action': 'CloseObject', 'objectId': fridge['objectId'], 'forceAction': True}))
            events.append(self.env.step({'action': 'Pass'}))
            events.append(self.env.step({'action': 'OpenObject', 'objectId': fridge['objectId'], 'forceAction': True}))
            events.append(self.env.step({'action': 'PickupObject', 'objectId': object['objectId'], 'forceAction': True}))
            events.append(self.env.step({'action': 'CloseObject', 'objectId': fridge['objectId'], 'forceAction': True}))

            # print([[e.metadata['errorMessage'], e.metadata["lastActionSuccess"], e.metadata['lastAction']] for e in events])
            
            if all(e.metadata['lastActionSuccess'] for e in events): #  and self.curr_recep == tar
                self.feedback = "You cool the %s using the %s." % (obj, tar)

        elif cmd['action'] == self.Action.SLICE:
            obj, rel, tar = cmd['obj'], cmd['rel'], cmd['tar']
            object = self.get_object(obj, self.objects)
            inventory_objects = self.env.last_event.metadata['inventoryObjects']
            if 'Knife' in inventory_objects[0]['objectType']:
                event = self.env.step({'action': "SliceObject",
                                        'objectId': object['object_id']})
            self.feedback = "You slice %s with the %s" % (obj, tar)

        elif cmd['action'] == self.Action.INVENTORY:
            if len(self.inventory) > 0:
                self.feedback = "You are carrying: a %s" % (self.inventory[0])
            else:
                self.feedback = "You are not carrying anything."

        elif cmd['action'] == self.Action.EXAMINE:
            target = cmd['tar']
            receptacle = self.get_object(target, self.receptacles)
            object = self.get_object(target, self.objects)

            if receptacle:
                self.visible_objects, self.feedback = self.print_frame(receptacle, self.curr_loc)
                self.frame_desc = str(self.feedback)
            elif object:
                self.feedback = self.print_object(object)

        elif cmd['action'] == self.Action.LOOK:
            if self.curr_recep == "nothing":
                self.feedback = "You are in the middle of a room. Looking quickly around you, you see nothing."
            else:
                self.feedback = "You are facing the %s. Next to it, you see nothing." % self.curr_recep

    except:
        if self.debug:
            print(traceback.format_exc())

    if event and not event.metadata['lastActionSuccess']:
        self.feedback = "Nothing happens."
        if self.debug:
            print(event.metadata['errorMessage'])

    if self.debug:
        print(self.feedback)
    return self.feedback, [e.metadata["errorMessage"] for e in events] if events else [event.metadata["errorMessage"] if event else ""]

def Thor_init_with_errors(self, queue, train_eval="train"):
    Thread.__init__(self)
    self.action_queue = queue
    self.mask_rcnn = None
    self.env =  None
    self.train_eval = train_eval
    self.controller_type = "oracle"
    
    self._errors = None

def Thor_step_with_errors(self, action):
    if not self._done:
        # take action
        self.prev_command = str(action)
        self._feedback, self._errors = self.controller.step(action)
        self._res = self.get_info()
        if self.env.save_frames_to_disk:
            self.record_action(action)
    self.steps += 1
    
def Thor_init_env(self, config):
    self.config = config

    screen_height = config['env']['thor']['screen_height']
    screen_width = config['env']['thor']['screen_width']
    smooth_nav = config['env']['thor']['smooth_nav']
    save_frames_to_disk = config['env']['thor']['save_frames_to_disk']

    if not self.env:
        self.env = ThorEnv(player_screen_height=screen_height,
                            player_screen_width=screen_width,
                            smooth_nav=smooth_nav,
                            save_frames_to_disk=save_frames_to_disk)
    self.controller_type = self.config['controller']['type']
    self._done = False
    self._res = ()
    self._feedback = ""
    # self.expert = HandCodedThorAgent(self.env, max_steps=200)
    self.prev_command = ""
    # self.load_mask_rcnn()

def Thor_reset(self, task_file):
    assert self.env
    assert self.controller_type

    self.set_task(task_file)

    # scene setup
    scene_num = self.traj_data['scene']['scene_num']
    object_poses = self.traj_data['scene']['object_poses']
    dirty_and_empty = self.traj_data['scene']['dirty_and_empty']
    object_toggles = self.traj_data['scene']['object_toggles']
    scene_name = 'FloorPlan%d' % scene_num
    self.env.reset(scene_name)
    self.env.restore_scene(object_poses, object_toggles, dirty_and_empty)

    # recording
    save_frames_path = self.config['env']['thor']['save_frames_path']
    self.env.save_frames_path = os.path.join(save_frames_path, self.traj_root.replace('../', ''))

    # initialize to start position
    self.env.step(dict(self.traj_data['scene']['init_action']))            # print goal instr
    task_desc = get_templated_task_desc(self.traj_data)
    print("Task: %s" % task_desc)
    # print("Task: %s" % (self.traj_data['turk_annotations']['anns'][0]['task_desc']))

    # setup task for reward
    class args: pass
    args.reward_config = os.path.join(alfworld.agents.__path__[0], 'config/rewards.json')
    self.env.set_task(self.traj_data, args, reward_type='dense')

    # set controller
    self.controller_type = self.config['controller']['type']
    self.goal_desc_human_anns_prob = self.config['env']['goal_desc_human_anns_prob']
    load_receps = self.config['controller']['load_receps']
    debug = self.config['controller']['debug']

    if self.controller_type == 'oracle':
        self.controller = OracleAgent(self.env, self.traj_data, self.traj_root,
                                        load_receps=load_receps, debug=debug,
                                        goal_desc_human_anns_prob=self.goal_desc_human_anns_prob)
    elif self.controller_type == 'oracle_astar':
        self.controller = OracleAStarAgent(self.env, self.traj_data, self.traj_root,
                                            load_receps=load_receps, debug=debug,
                                            goal_desc_human_anns_prob=self.goal_desc_human_anns_prob)
    elif self.controller_type == 'mrcnn':
        self.controller = MaskRCNNAgent(self.env, self.traj_data, self.traj_root,
                                        pretrained_model=self.mask_rcnn,
                                        load_receps=load_receps, debug=debug,
                                        goal_desc_human_anns_prob=self.goal_desc_human_anns_prob,
                                        save_detections_to_disk=self.env.save_frames_to_disk, save_detections_path=self.env.save_frames_path)
    elif self.controller_type == 'mrcnn_astar':
        self.controller = MaskRCNNAStarAgent(self.env, self.traj_data, self.traj_root,
                                                pretrained_model=self.mask_rcnn,
                                                load_receps=load_receps, debug=debug,
                                                goal_desc_human_anns_prob=self.goal_desc_human_anns_prob,
                                                save_detections_to_disk=self.env.save_frames_to_disk, save_detections_path=self.env.save_frames_path)
    else:
        raise NotImplementedError()

    # zero steps
    self.steps = 0

    # reset expert state
    # self.expert.reset(task_file)
    self.prev_command = ""

    # return intro text
    self._feedback = self.controller.feedback
    self._res = self.get_info()

    return self._feedback

def Thor_get_info(self):
        won = self.env.get_goal_satisfied()
        pcs = self.env.get_goal_conditions_met()
        goal_condition_success_rate = pcs[0] / float(pcs[1])
        acs = self.controller.get_admissible_commands()

        expert_actions = []

        training_method = self.config["general"]["training_method"]
        if training_method == "dqn":
            max_nb_steps_per_episode = self.config["rl"]["training"]["max_nb_steps_per_episode"]
        elif training_method == "dagger":
            max_nb_steps_per_episode = self.config["dagger"]["training"]["max_nb_steps_per_episode"]
        else:
            raise NotImplementedError
        self._done = won or self.steps > max_nb_steps_per_episode
        return (self._feedback, self._done, acs, won, goal_condition_success_rate, expert_actions)