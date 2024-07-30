# totally same as IS-PF-MAAC, used for test different parameter, will delete after test
# totally new version
from classic_env import classic_sim as gym
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from Search_Utils import m_marl_utils
import copy
from torch.nn.utils.rnn import pad_sequence
import os
import pandas as pd
import random
import numpy as np
import time
import math

class PolicyNet(torch.nn.Module):
    def __init__(self, obs_dim, hidden_dim, time_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.gru = torch.nn.GRU(input_size=obs_dim, hidden_size=hidden_dim, batch_first=True)
        # # Freeze the GRU layer
        # for param in self.gru.parameters():
        #     param.requires_grad = False
        self.fc1 = torch.nn.Linear(hidden_dim+time_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x, tb, action_mask):
        lengths = torch.tensor([len(seq) for seq in x], dtype=torch.long)
        x = pad_sequence(x, batch_first=True)
        mask = torch.arange(x.size(1)).unsqueeze(0) < lengths.unsqueeze(1)
        mask = mask.to(x.device)
        x, _ = self.gru(x)
        x = x * mask.unsqueeze(2).float()
        x = x[torch.arange(x.size(0)), lengths - 1]
        x_cat = torch.cat((x, tb), dim=1)
        x = F.relu(self.fc1(x_cat))
        logits = self.fc2(x)
        logits_masked = logits.masked_fill(action_mask, float('-inf'))
        return F.softmax(logits_masked, dim=1)

class Qnet(torch.nn.Module):
    ''' 只有一层隐藏层的Q网络 '''
    def __init__(self, obs_dim, hidden_dim, time_dim, action_dim, num_agents):
        super(Qnet, self).__init__()
        self.gru = torch.nn.GRU(input_size=obs_dim * num_agents, hidden_size=hidden_dim, batch_first=True)
        # # Freeze the GRU layer
        # for param in self.gru.parameters():
        #     param.requires_grad = False
        self.fc1 = torch.nn.Linear(hidden_dim + time_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x, tb):
        lengths = torch.tensor([len(seq) for seq in x], dtype=torch.long)
        x = pad_sequence(x, batch_first=True)
        mask = torch.arange(x.size(1)).unsqueeze(0) < lengths.unsqueeze(1)
        mask = mask.to(x.device)
        x, _ = self.gru(x)
        x = x * mask.unsqueeze(2).float()
        # print("x before lengths - 1:", x)
        x = x[torch.arange(x.size(0)), lengths - 1]
        # print("x after gru:", x)
        # print("x:",x,", tb:",tb)
        x_cat = torch.cat((x, tb), dim=1)
        # print("x_cat:",x_cat)
        x = F.relu(self.fc1(x_cat))
        x = F.sigmoid(self.fc2(x))
        delta = 1e-6
        x = torch.clamp(x, min=delta, max=1 - delta)
        return x

class ActorCritic:
    def __init__(self, obs_dim, hidden_dim, time_dim, action_dim, actor_lr, critic_lr, device, num_agents):
        # super(ActorCritic, self).__init__()
        self.action_dim = action_dim
        self.actor = PolicyNet(obs_dim, hidden_dim, time_dim, action_dim).to(device)
        self.critic = Qnet(obs_dim, hidden_dim, time_dim, action_dim, num_agents).to(device)  # 价值网络
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)  # 价值网络优化器
        self.device = device

    def take_action(self, obs, remain_time, action_num):
        obs = [obs]
        tb = remain_time.unsqueeze(0)
        action_mask = self.create_action_mask(self.action_dim, action_num)
        action_mask = torch.tensor(action_mask, dtype=torch.bool).to(self.device)
        # print("action_num:",action_num)
        # print("obs:",obs,", remain_time:",remain_time, ", action_mask:", action_mask)
        probs = self.actor(obs, tb, action_mask)
        # print("action_mask:",action_mask)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        prob = probs[0][action].item()
        return action.item(),prob

    def create_action_mask(self, total_actions, valid_actions):
        action_mask = [False] * valid_actions + [True] * (total_actions - valid_actions)
        return action_mask

    def create_action_masks(self, total_actions, valid_actions_list, device):
        action_masks = []
        for valid_actions in valid_actions_list:
            action_mask = [False] * valid_actions + [True] * (total_actions - valid_actions)
            action_masks.append(action_mask)
        action_masks_tensor = torch.tensor(action_masks, dtype=torch.bool).to(device)
        return action_masks_tensor

class PF_MAAC:
    def __init__(self, num_agents, obs_dim, hidden_dim, time_dim, action_dim, actor_lr, critic_lr, device):
        self.acs = [ActorCritic(obs_dim, hidden_dim, time_dim,action_dim, actor_lr, critic_lr, device, num_agents) for _ in range(num_agents)]
        # ****employ target_ac to make training more stable*****
        self.target_acs = [ActorCritic(obs_dim, hidden_dim, time_dim,action_dim, actor_lr, critic_lr, device, num_agents) for _ in range(num_agents)]
        for q in range(len(self.acs)):
            ac = self.acs[q]
            target_ac = self.target_acs[q]
            target_ac.critic.load_state_dict(ac.critic.state_dict())
            target_ac.actor.load_state_dict(ac.actor.state_dict())
        self.tau = 0.05
        #********************************************************
        self.num_agents = num_agents
        self.device = device
        # ******** increase exploration *************
        # self.update_count = 0
        # self.epsilon = 0.1
        self.counter = 0

    def take_actions(self, observations, remain_time, action_nums):
        actions = []
        probs = []
        for agent_id, agent in enumerate(self.acs):
            obs = observations[agent_id]
            action_num = action_nums[agent_id]

            action,prob = agent.take_action(obs, remain_time, action_num)
            actions.append(action)
            probs.append(prob)
        return actions,probs

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def update(self, transitions):
        pessimistic = 0.1 *(max(self.counter - 1/2 * num_episodes,0))/num_episodes
        pessimistic = 0
        self.counter += 1
        if self.counter % 1000 == 0:
            print("counter:", self.counter)
        trajec_length = len(transitions['observations'])
        all_current_Q_values = []
        all_next_Q_values = []
        all_is_radios = []
        rewards = transitions['rewards']
        dones = transitions['dones']
        tb = transitions['remain_time']  # Remain time for capture
        next_tb = transitions['next_remain_time']
        weights = transitions['weights']
        # sample_probs = transitions['sample_probs']
        # next_tb = copy.deepcopy(tb[1:])
        # next_tb.append(copy.deepcopy(tb[-1]))
        # sample_probs = torch.tensor(sample_probs, dtype=torch.float).view(-1, 1).to(self.device)
        # print("sample_probs:",sample_probs)
        rewards = torch.tensor(rewards, dtype=torch.float).view(-1, 1).to(self.device)
        weights = torch.tensor(weights, dtype=torch.float).view(-1, 1).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float).view(-1, 1).to(self.device)
        tb = torch.stack(tb).to(self.device)
        next_tb = torch.stack(next_tb).to(self.device)
        observation_lists = []
        next_observation_lists = []
        for agent_id in range(self.num_agents):
            observations = [transitions['observations'][_][agent_id] for _ in range(trajec_length)]
            next_observations = [transitions['next_observations'][_][agent_id] for _ in range(trajec_length)]
            observations = [torch.tensor(obs, dtype=torch.float).to(self.device) for obs in observations]
            next_observations = [torch.tensor(next_obs, dtype=torch.float).to(self.device) for next_obs in
                                 next_observations]
            observation_lists.append(observations)
            next_observation_lists.append(next_observations)
        global_observations = [torch.cat(tensors, dim=1) for tensors in zip(*observation_lists)]
        global_next_observations = [torch.cat(tensors, dim=1) for tensors in zip(*next_observation_lists)]
        for agent_id, agent in enumerate(self.acs):
            agent.critic_optimizer.zero_grad()
            agent.actor_optimizer.zero_grad()
            observations = [transitions['observations'][_][agent_id] for _ in range(trajec_length)]
            actions = [transitions['actions'][_][agent_id] for _ in range(trajec_length)]
            sample_probs = [transitions['sample_probs'][_][agent_id] for _ in range(trajec_length)]
            next_observations = [transitions['next_observations'][_][agent_id] for _ in range(trajec_length)]
            action_nums = [transitions['action_nums'][_][agent_id] for _ in range(trajec_length)]
            next_action_nums = [transitions['next_action_nums'][_][agent_id] for _ in range(trajec_length)]
            # next_actions = copy.copy(actions[1:])
            # next_actions.append(0)
            observations = [torch.tensor(obs, dtype=torch.float).to(self.device) for obs in observations]
            next_observations = [torch.tensor(next_obs, dtype=torch.float).to(self.device) for next_obs in next_observations]
            actions = torch.tensor(actions, dtype=torch.long).view(-1, 1).to(self.device)
            sample_probs = torch.tensor(sample_probs, dtype=torch.float).view(-1, 1).to(self.device)
            # next_actions = torch.tensor(next_actions, dtype=torch.long).view(-1, 1).to(self.device)
            action_masks = agent.create_action_masks(agent.action_dim, action_nums,self.device)
            next_action_masks = agent.create_action_masks(agent.action_dim, next_action_nums,self.device)
            current_Q_all_actions = agent.critic(global_observations, tb)
            current_Q_values = current_Q_all_actions.gather(1, actions)
            log_current_1minusQ = torch.log(1-current_Q_values)

            # importance sampling code:
            cur_probs = agent.actor(observations, tb, action_masks)
            cur_probs = cur_probs.gather(1, actions).detach()
            division_probs = (cur_probs/sample_probs).squeeze()
            p_is_ratios = division_probs
            t_is_ratios = torch.ones_like(division_probs)
            # Compute IS ratios
            cumulative_product = 1.0
            for i in range(len(division_probs)):
                cumulative_product *= division_probs[i]
                t_is_ratios[i] = cumulative_product
                if dones[i] == 1:
                    cumulative_product = 1.0
            t_is_ratios = t_is_ratios.view(-1,1)
            # ***********next_Q_values calculate from target_critic and target_actor*************
            target_ac = self.target_acs[agent_id]
            target_actor = target_ac.actor
            target_critic = target_ac.critic
            next_probs = target_actor(next_observations, next_tb, next_action_masks).detach()
            next_actions = self._actions_from_probs(next_probs)
            next_actions = torch.tensor(next_actions, dtype=torch.long).view(-1, 1).to(self.device)
            next_Q_values = target_critic(global_next_observations, next_tb).gather(1,next_actions).detach()

            #*************************************************************************************
            log_next_1minusQ = torch.log(1-next_Q_values)
            all_current_Q_values.append(current_Q_values)
            all_next_Q_values.append(next_Q_values)
            all_is_radios.append(p_is_ratios)
            # Compute actor loss
            probs = agent.actor(observations, tb, action_masks)
            # print("current_Q_all_actions:", current_Q_all_actions)
            # print("probs:", probs)
            # print("Q_value:",current_Q_values)
            V_value = torch.sum(current_Q_all_actions.detach() * probs.detach(), dim=1, keepdim=True)
            # print(V_value)
            prob = probs.gather(1, actions)
            action_log_probs = torch.log(prob)
            # print("action_log_probs:",action_log_probs)
            actor_loss = -((current_Q_values.detach()-V_value-pessimistic) * action_log_probs * t_is_ratios * weights).mean()
            # actor_loss = -((current_Q_values.detach() - V_value - pessimistic) * action_log_probs * weights).mean()
            # Update the actor
            actor_loss.backward()
            agent.actor_optimizer.step()
            # print("agent_id "+ str(agent_id)+ " policy net update finished")
        # Centralized processing of Q_tot

        log_current_1minusQ = 0
        log_next_1minusQ = 0
        is_ratios = 1.0
        for agent_id in range(len(self.acs)):
            log_current_1minusQ += torch.log(1-all_current_Q_values[agent_id])
            log_next_1minusQ += torch.log(1-all_next_Q_values[agent_id])
            is_ratios *= all_is_radios[agent_id]
        total_current_Q = 1 - torch.exp(log_current_1minusQ)
        total_next_Q = 1 - torch.exp(log_next_1minusQ)
        # print("dones:",dones)
        # print("rewards:",rewards)
        total_expected_Q_values = 1 - (1-(total_next_Q * (1-dones))) * (1-rewards)
        # total_critic_loss = F.mse_loss(total_current_Q, total_expected_Q_values)
        normalized_is_ratios = is_ratios / is_ratios.sum()
        squared_diffs = normalized_is_ratios * (total_current_Q - total_expected_Q_values) ** 2
        # squared_diffs = (total_current_Q - total_expected_Q_values) ** 2
        total_critic_loss = squared_diffs.mean()
        total_critic_loss.backward()
        for agent_id, agent in enumerate(self.acs):
            agent.critic_optimizer.step()
        # print("critic update finish")
        for agent_id, agent in enumerate(self.acs):
            target_ac = self.target_acs[agent_id]
            target_actor = target_ac.actor
            target_critic = target_ac.critic
            self.soft_update(agent.actor, target_actor)
            self.soft_update(agent.critic, target_critic)

    def _actions_from_probs(self, action_probs):
        chosen_actions = []
        # Iterate over each set of probabilities and sample an action
        for prob in action_probs:
            # Create a categorical distribution and sample an action
            action_distribution = torch.distributions.Categorical(prob)
            action = action_distribution.sample()
            chosen_actions.append(action.item())
        return chosen_actions

def add_component_to_csv(file_path, key, mean_values, variance_values, mean_objective1=None, variance_objective1=None):
    new_component = pd.DataFrame({
        'Key': [key],
        'MeanValues': [mean_values],
        'VarianceValues': [variance_values],
        'MeanObjectiveI': [mean_objective1],
        'VarianceObjectiveI': [variance_objective1]
    })

    # Check if the file exists
    if os.path.exists(file_path):
        # File exists, append the new component without header
        existing_data = pd.read_csv(file_path)
        if key in existing_data['Key'].values:
            # Key exists, replace the row
            existing_data = existing_data[existing_data['Key'] != key]  # Remove old row
            existing_data = pd.concat([existing_data, new_component], ignore_index=True)  # Add new row
        else:
            existing_data = pd.concat([existing_data, new_component], ignore_index=True)
        existing_data.to_csv(file_path, index=False)
    else:
        # File does not exist, create it and add the component with header
        new_component.to_csv(file_path, mode='w', header=True, index=False)

def add_or_update_column_based_on_key(file_path, key, training_time):
    # log time
    training_time *= 10
    training_time = math.log10(training_time)
    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No such file: '{file_path}'")

    # Load existing data
    existing_data = pd.read_csv(file_path)

    # Check if the 'Training Time/log(s)' column exists, if not add it with default values
    if 'Training Time/log(s)' not in existing_data.columns:
        existing_data['Training Time/log(s)'] = None

    # Check if the key exists in the DataFrame
    if key in existing_data['Key'].values:
        # Key exists, update the column for the specific key
        existing_data.loc[existing_data['Key'] == key, 'Training Time/log(s)'] = training_time
    else:
        # Optionally, handle the case where the key does not exist
        print(f"Key '{key}' not found. No data updated.")

    # Save the updated data back to the CSV file
    existing_data.to_csv(file_path, index=False)

if __name__ == "__main__":
    Algorithm = "PF-MAAC"
    time_budget = 10
    robot_num = 3
    env_name = "Grid_20"
    #currently time_budget = 6 for OFFICE time_budget, and time_budget = 7 for grid_10 time_budget = 7 for MUSEUM

    actor_lr = 1e-4
    critic_lr = 1e-3
    num_episodes = 20000
    hidden_dim = 16
    buffer_size = 200
    minimal_size = 10
    batch_size = 8
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
        "cpu")
    env = gym(env_name, robot_num, time_budget)
    env.seed = 0
    # print("env.robot_num", env.robot_num)
    # env.robot_num = robot_num
    # print("env.robot_num",env.robot_num)
    torch.manual_seed(0)
    obs_dim = env.position_embed
    action_dim = env.action_dim
    time_dim = env.time_embed
    return_list_set = []
    # replay_buffer = m_marl_utils.ReplayBuffer(buffer_size)
    replay_buffer = m_marl_utils.ImportanceSamplingReplayBuffer(buffer_size)
    start_time = time.time()
    for p in range(3):
        agents = PF_MAAC(robot_num, obs_dim, hidden_dim, time_dim, action_dim, actor_lr, critic_lr,
                            device)
        return_list = m_marl_utils.train_importance_sampling(env, agents, num_episodes, replay_buffer, minimal_size, batch_size)
        return_list_set.append(copy.copy(return_list))
    end_time = time.time()
    training_duration = end_time - start_time

    return_list = [sum(col) / len(col) for col in zip(*return_list_set)]
    return_variance = [sum((xi - mu) ** 2 for xi in col) / len(col) for col, mu in
                       zip(zip(*return_list_set), return_list)]
    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Detection Probability')
    plt.title('IS-Off-Policy PF-MAAC on {} with R = {}'.format(env_name,robot_num))
    plt.show()

    mv_return = m_marl_utils.moving_average(return_list, 9)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Detection Probability')
    plt.title('IS-Off-Policy PF-MAAC on {} with R = {}'.format(env_name,robot_num))
    plt.show()

    selected_mv_return = mv_return.tolist()[::9]
    selected_return_variance = return_variance[::9]
    selected_episodes_list = episodes_list[::9]
    file_path = './{}_R{}.csv'
    file_path = file_path.format(env_name, robot_num)
    add_component_to_csv(file_path, Algorithm, selected_mv_return, selected_return_variance)
    # save_to_csv(selected_mv_return,Algorithm,file_path)
    plt.plot(selected_episodes_list, selected_mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Detection Probability')
    plt.title('IS-Off-Policy PF-MAAC on {} with R = {}'.format(env_name,robot_num))
    plt.show()