from Search_Utils.Embedding import EmbeddingLayer
import random
import torch
import copy
# all imitate the gym
# gym.make(env_name);
# env.seed(0)
# env.observation_space.shape[0]
# action_dim = env.action_space.n
# state = env.reset()
# next_state, reward, done, _ = env.step(action)
class simple_sim:
    def __init__(self, time_budget):
        self.max_time = 20
        self.seed = 0
        random.seed(self.seed)
        self.env_name = "Simple"
        self.robot_num = 2
        self.target_num = 1000
        self.remain_target_num = self.target_num
        self.capture_one_step = 0
        self.start_position = 1
        self.total_position = 7
        self.position_embed = 2
        self.time_embed = 8
        self.action_dim = 4
        self.embedding_layer = EmbeddingLayer(self.total_position + 1, self.position_embed, 0)
        self.time_embedding = EmbeddingLayer(self.max_time + 1, self.time_embed, 0)
        self.robot_trajectories = None
        self.observations = None
        self.done = False
        self.team_reward = None
        self.action_nums = None
        self.remain_time_init = time_budget
        self.remain_time = self.remain_time_init
        self.target_start_positions = [3, 4, 7]
        self.target_start_positions_prob = [0.35,0.25,0.4]
        self.target_position_list = None
        self.capture_flag_list = None
        self.info_dict = {}
    def reset(self):
        self.robot_trajectories = [[self.start_position] for _ in range(self.robot_num)]
        self.action_nums = [self._actionNum_position(self.start_position) for _ in range(self.robot_num)]
        self._trajectories_to_observations()
        self.done = False
        self.capture_flag_list = False
        self.remain_target_num = self.target_num
        self.team_reward = 0
        self.remain_time = random.randint(1,self.max_time)
        self.remain_time = self.remain_time_init
        self.info_dict['action_nums'] = self.action_nums
        self.info_dict['remain_time'] = self.time_embedding(torch.tensor(self.remain_time))
        self.target_position_list = random.choices(self.target_start_positions, weights = self.target_start_positions_prob,
                                              k=self.target_num)
        self.capture_flag_list = [False for _ in range(self.target_num)]
        return copy.deepcopy(self.observations), self.info_dict['remain_time'], copy.copy(self.action_nums)

    def step(self, actions):
        for robot_label in range(self.robot_num):
            action = actions[robot_label]
            self._take_action(robot_label,action)
        self.remain_time -= 1
        self.info_dict['remain_time'] = self.time_embedding(torch.tensor(self.remain_time))
        self.team_reward = 0.
        self.capture_one_step = 0
        for robot_label in range(self.robot_num):
            self._judge_capture(robot_label)
        self.info_dict['capture_reward'] = self.team_reward
        self.team_reward = self.capture_one_step / self.remain_target_num
        self.remain_target_num -= self.capture_one_step
        self.info_dict['action_nums'] = self.action_nums
        self._trajectories_to_observations()
        return copy.deepcopy(self.observations), self.team_reward, self.done, copy.deepcopy(self.info_dict)
        # for our ProbVDN+MADDPG, we need next_observation, (reward = None), done, _ { is a dictionary} we add a capture flag in this dict
        # a list of observations, reward = None(team_reward), done, _ {capture_flag; }
        # action_max = 4, we assume it increase as the number of node increase

    def _trajectories_to_observations(self):
        self.observations = []
        for robot_label in range(self.robot_num):
            observation_embed = self._trajectory_to_observation(robot_label)
            self.observations.append(observation_embed)

    def _trajectory_to_observation(self, robot_label):
        return self.embedding_layer(torch.tensor(self.robot_trajectories[robot_label]))

    def _judge_capture(self,robot_label):
        robot_cur_position = self.robot_trajectories[robot_label][-1]
        for h in range(self.target_num):
            capture_flag = self.capture_flag_list[h]
            if not capture_flag:
                target_position = self.target_position_list[h]
                if target_position == robot_cur_position:
                    self.capture_one_step += 1
                    self.team_reward += (1./self.target_num)
                    self.capture_flag_list[h] = True
        if self.done == False and self.remain_time == 0 and robot_label == self.robot_num - 1:
            self.done = True


    def _take_action(self, robot_label, action):
        robot_cur_position = self.robot_trajectories[robot_label][-1]
        robot_next_position = robot_cur_position
        if robot_cur_position == 1:
            if action == 1:
                robot_next_position = 2
            elif action == 2:
                robot_next_position = 4
            elif action == 3:
                robot_next_position = 5
        elif robot_cur_position == 2:
            if action == 1:
                robot_next_position = 1
            elif action == 2:
                robot_next_position = 3
        elif robot_cur_position == 3:
            if action == 1:
                robot_next_position = 2
        elif robot_cur_position == 4:
            if action == 1:
                robot_next_position = 1
        elif robot_cur_position == 5:
            if action == 1:
                robot_next_position = 1
            elif action == 2:
                robot_next_position = 6
        elif robot_cur_position == 6:
            if action == 1:
                robot_next_position = 5
            elif action == 2:
                robot_next_position = 7
        else:
            if action == 1:
                robot_next_position = 6
        self.robot_trajectories[robot_label].append(robot_next_position)
        self.action_nums[robot_label] = self._actionNum_position(robot_next_position)


    def _actionNum_position(self, position):
        if position == 1:
            action_num = 4
        elif position == 2:
            action_num = 3
        elif position == 3:
            action_num = 2
        elif position == 4:
            action_num = 2
        elif position == 5:
            action_num = 3
        elif position == 6:
            action_num = 3
        else:
            action_num = 2
        return action_num