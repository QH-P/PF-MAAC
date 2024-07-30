from tqdm import tqdm
import numpy as np
import torch
import collections
import random

def train_on_policy_agent(env, agents, num_episodes):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                # individual: observations, actions, next_observations, action_num
                # team: remain_time, next_remain_time, rewards, capture_flag, dones
                transition_dict = {'observations': [],'remain_time': [], 'actions': [], 'action_nums': [] ,
                      'next_observations': [], 'rewards': [], 'dones': []}
                # print("trajectories:",env.robot_trajectories)
                observations, remain_time, action_nums = env.reset()
                done = False
                while not done:
                    actions = agents.take_actions(observations, remain_time, action_nums)
                    next_observations, reward, done, info = env.step(actions)
                    transition_dict['observations'].append(observations)
                    transition_dict['remain_time'].append(remain_time)
                    transition_dict['actions'].append(actions)
                    transition_dict['action_nums'].append(action_nums)
                    transition_dict['next_observations'].append(next_observations)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    observations = next_observations
                    action_nums = info['action_nums']
                    remain_time = info['remain_time']
                    episode_return += reward
                return_list.append(episode_return)
                agents.update(transition_dict)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                                      'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
                if i_episode % (int(num_episodes / 30)) == 0 and i_episode != 0:
                    print("trajectories:",env.robot_trajectories)
    return return_list

def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size - 1, 2)
    begin = np.cumsum(a[:window_size - 1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, observations, remain_time, actions, action_nums, reward, next_observations,
            next_remain_time, next_action_nums, done):
        self.buffer.append((observations,remain_time, actions, action_nums, reward,
                            next_observations, next_remain_time, next_action_nums, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        observations, remain_time, actions, action_nums, reward, next_observations, next_remain_time, \
        next_action_nums, done = zip(*transitions)
        return observations, remain_time, actions, action_nums, reward, next_observations, next_remain_time, \
        next_action_nums, done

    def size(self):
        return len(self.buffer)

class WeightedReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = collections.deque(maxlen=capacity)
        self.freqs = collections.defaultdict(int)

    def add(self, observations, remain_time, actions, action_nums, reward, next_observations,
            next_remain_time, next_action_nums, done):
        # Convert observations and actions to a hashable format for tracking
        state_action = (tuple(observations), tuple(actions))
        self.freqs[state_action] += 1
        if len(self.buffer) == self.capacity:
            old_obs, _, old_actions, *_ = self.buffer[0]
            old_state_action = (tuple(old_obs), tuple(old_actions))
            self.freqs[old_state_action] -= 1
            if self.freqs[old_state_action] == 0:
                del self.freqs[old_state_action]
        self.buffer.append((observations, remain_time, actions, action_nums, reward,
                            next_observations, next_remain_time, next_action_nums, done))

    def sample(self, batch_size):
        # Calculate weights: inversely proportional to frequencies
        weights = [1.0 / self.freqs[(tuple(item[0]), tuple(item[2]))] for item in self.buffer]
        total_weight = sum(weights)
        probabilities = [w / total_weight for w in weights]

        # Weighted random sampling
        transitions = random.choices(self.buffer, weights=probabilities, k=batch_size)
        observations, remain_time, actions, action_nums, reward, next_observations, next_remain_time, \
        next_action_nums, done = zip(*transitions)
        return observations, remain_time, actions, action_nums, reward, next_observations, next_remain_time, \
               next_action_nums, done

    def size(self):
        return len(self.buffer)

def train_off_policy_agent(env, agents, num_episodes, replay_buffer, minimal_size, batch_size):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                observations, remain_time, action_nums = env.reset()
                done = False
                while not done:
                    actions = agents.take_actions(observations, remain_time, action_nums)
                    next_observations, reward, done, info = env.step(actions)
                    next_remain_time = info['remain_time']
                    next_action_nums = info['action_nums']
                    replay_buffer.add(observations, remain_time, actions, action_nums, reward, next_observations,
                                      next_remain_time, next_action_nums, done)
                    observations = next_observations
                    action_nums = next_action_nums
                    remain_time = next_remain_time
                    episode_return += reward
                    if replay_buffer.size() > minimal_size:
                        b_o, b_t, b_a, b_an, b_r, b_no, b_nt, b_nan, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {'observations': b_o, 'remain_time': b_t, 'actions': b_a, 'action_nums':b_an,
                                           'rewards': b_r, 'next_observations': b_no, 'next_remain_time': b_nt,
                                           'next_action_nums':b_nan,'dones': b_d}
                        agents.update(transition_dict)
                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                                      'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
                if i_episode % (int(num_episodes / 30)) == 0 and i_episode != 0:
                    print("trajectories:",env.robot_trajectories)
    return return_list

# class ImportanceSamplingReplayBuffer:
#     def __init__(self, capacity):
#         self.buffer = collections.deque(maxlen=capacity)
#         self.current_trajectory = []  # Temporary storage for the current trajectory
#
#     def start_trajectory(self):
#         self.current_trajectory = []  # Reset the current trajectory
#
#     def add(self, observations, remain_time, actions, action_nums, reward, next_observations,
#             next_remain_time, next_action_nums, done, probs):
#         # Add experience to the current trajectory
#         self.current_trajectory.append((observations, remain_time, actions, action_nums, reward,
#                                         next_observations, next_remain_time, next_action_nums, done, probs))
#         # If 'done' is True, it means the trajectory has ended. Save the trajectory to the buffer and reset.
#         if done:
#             self.buffer.append(list(self.current_trajectory))  # Store as a list to keep the trajectory together
#             self.start_trajectory()  # Prepare for a new trajectory
#
#     def sample(self, num_trajectories):
#         # Make sure we don't sample more trajectories than we have
#         num_trajectories = min(num_trajectories, len(self.buffer))
#
#         sampled_trajectories = random.sample(self.buffer, num_trajectories)
#         # Flatten the list of sampled trajectories if you need a continuous list
#         flattened_samples = [experience for trajectory in sampled_trajectories for experience in trajectory]
#         observations, remain_time, actions, action_nums, reward, next_observations, next_remain_time, \
#         next_action_nums, done, probs = zip(*flattened_samples)
#         return observations, remain_time, actions, action_nums, reward, next_observations, next_remain_time, \
#                next_action_nums, done, probs
#
#     def size(self):
#         return len(self.buffer)
class ImportanceSamplingReplayBuffer4:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)
        self.current_trajectory = []  # Temporary storage for the current trajectory
        self.state_frequencies = collections.defaultdict(int)

    def start_trajectory(self):
        self.current_trajectory = []  # Reset the current trajectory

    def add(self, observations, remain_time, actions, action_nums, reward, next_observations,
            next_remain_time, next_action_nums, done, probs):
        # Add experience to the current trajectory
        self.state_frequencies[str(observations)] += 1
        self.current_trajectory.append((observations, remain_time, actions, action_nums, reward,
                                        next_observations, next_remain_time, next_action_nums, done, probs))
        # If 'done' is True, it means the trajectory has ended. Save the trajectory to the buffer and reset.
        if done:
            self.buffer.append(list(self.current_trajectory))  # Store as a list to keep the trajectory together
            self.start_trajectory()  # Prepare for a new trajectory

    def sample(self, num_trajectories):
        # Make sure we don't sample more trajectories than we have
        num_trajectories = min(num_trajectories, len(self.buffer))

        sampled_trajectories = random.sample(self.buffer, num_trajectories)
        # Flatten the list of sampled trajectories if you need a continuous list
        flattened_samples = [experience for trajectory in sampled_trajectories for experience in trajectory]
        observations, remain_time, actions, action_nums, reward, next_observations, next_remain_time, \
        next_action_nums, done, probs = zip(*flattened_samples)

        frequencies = [self.state_frequencies[str(obs)] for obs in observations]
        max_frequency = max(self.state_frequencies.values(), default=1)
        normalized_frequencies = [self.state_frequencies[str(obs)] / max_frequency for obs in observations]
        weights = [1.0 / freq for freq in normalized_frequencies]
        # c, d = 0.02, 1.
        # # if np.max(weights)/np.min(weights) > d/c:
        # a, b = np.min(weights), np.max(weights)
        # scaled_weights = c + (weights - a) * (d - c) / (b - a + 0.000001)
        max_weight = max(weights)
        # weight_sum = sum(weights)
        scaled_weights = [w / max_weight for w in weights]
        # else:
        #     scaled_num = np.min(weights)/c
        #     scaled_weights = np.array(weights)/scaled_num
        return observations, remain_time, actions, action_nums, reward, next_observations, next_remain_time, \
               next_action_nums, done, probs, scaled_weights

    def size(self):
        return len(self.buffer)

class DequeWithHook(collections.deque):
    def __init__(self, maxlen, remove_callback):
        super().__init__(maxlen=maxlen)
        self.remove_callback = remove_callback

    def append(self, item):
        if len(self) == self.maxlen:
            # Call the hook before the oldest item is removed
            self.remove_callback(self[0])
        super().append(item)

class ImportanceSamplingReplayBuffer3:
    def __init__(self, capacity):
        print("buffer initial")
        self.buffer = collections.deque(maxlen=capacity)
        self.buffer = DequeWithHook(maxlen=capacity, remove_callback=self.on_remove)
        self.current_trajectory = []  # Temporary storage for the current trajectory
        self.state_frequencies = collections.defaultdict(int)

    def on_remove(self, trajectory):
        # Decrease frequency counts for each observation in the removed trajectory
        for observations, _, _, _, _, _, _, _, _, _ in trajectory:
            for obs in observations:
                obs_key = str(obs)
                if self.state_frequencies[obs_key] > 0:
                    self.state_frequencies[obs_key] -= 1
                    if self.state_frequencies[obs_key] == 0:
                        del self.state_frequencies[obs_key]  # Optional: clean up zero entries

    def start_trajectory(self):
        self.current_trajectory = []  # Reset the current trajectory

    def add(self, observations, remain_time, actions, action_nums, reward, next_observations,
            next_remain_time, next_action_nums, done, probs):
        # Add experience to the current trajectory
        for p in range(len(observations)):
            self.state_frequencies[str(observations[p])] += 1
        # self.state_frequencies[str(observations)] += 1
        self.current_trajectory.append((observations, remain_time, actions, action_nums, reward,
                                        next_observations, next_remain_time, next_action_nums, done, probs))
        # If 'done' is True, it means the trajectory has ended. Save the trajectory to the buffer and reset.
        if done:
            self.buffer.append(list(self.current_trajectory))  # Store as a list to keep the trajectory together
            self.start_trajectory()  # Prepare for a new trajectory

    def sample(self, num_trajectories):
        # Make sure we don't sample more trajectories than we have
        num_trajectories = min(num_trajectories, len(self.buffer))

        sampled_trajectories = random.sample(self.buffer, num_trajectories)
        # Flatten the list of sampled trajectories if you need a continuous list
        flattened_samples = [experience for trajectory in sampled_trajectories for experience in trajectory]
        observations, remain_time, actions, action_nums, reward, next_observations, next_remain_time, \
        next_action_nums, done, probs = zip(*flattened_samples)

        flat_observations = [obs for sublist in observations for obs in sublist]  # Flatten all observations

        weights = [1.0/ self.state_frequencies[str(obs)] for obs in flat_observations]
        # weight_sum = sum(weights)
        # scaled_weights = [w / weight_sum for w in weights]

        c, d = 0.02, 1.
        # if np.max(weights)/np.min(weights) > d/c:
        a, b = np.min(weights), np.max(weights)
        scaled_weights = c + (weights - a) * (d - c) / (b - a + 0.000001)

        index = 0
        reshaped_weights = []
        for p in range(len(observations)):
            group_weights = []
            for q in range(len(observations[p])):
                group_weights.append(scaled_weights[index])
                index += 1
            reshaped_weights.append(group_weights)
        # for obs_group in observations:
        #     group_weights = []
        #     for _ in obs_group:
        #         group_weights.append(scaled_weights[index])
        #         index += 1
        #     reshaped_weights.append(group_weights)
        return observations, remain_time, actions, action_nums, reward, next_observations, next_remain_time, \
               next_action_nums, done, probs, reshaped_weights

    def size(self):
        return len(self.buffer)

class ImportanceSamplingReplayBuffer2:
    # when trajectory pop, frequency change
    def __init__(self, capacity):
        print("buffer initial")
        self.buffer = collections.deque(maxlen=capacity)
        self.buffer = DequeWithHook(maxlen=capacity, remove_callback=self.on_remove)
        self.current_trajectory = []  # Temporary storage for the current trajectory
        self.state_frequencies = collections.defaultdict(int)

    def on_remove(self, trajectory):
        # Decrease frequency counts for each observation in the removed trajectory
        for observations, _, _, _, _, _, _, _, _, _ in trajectory:
            obs_key = str(observations)
            if self.state_frequencies[obs_key] > 0:
                self.state_frequencies[obs_key] -= 1
                if self.state_frequencies[obs_key] == 0:
                    del self.state_frequencies[obs_key]  # Optional: clean up zero entries

    def start_trajectory(self):
        self.current_trajectory = []  # Reset the current trajectory

    def add(self, observations, remain_time, actions, action_nums, reward, next_observations,
            next_remain_time, next_action_nums, done, probs):
        # Add experience to the current trajectory
        self.state_frequencies[str(observations)] += 1
        self.current_trajectory.append((observations, remain_time, actions, action_nums, reward,
                                        next_observations, next_remain_time, next_action_nums, done, probs))
        # If 'done' is True, it means the trajectory has ended. Save the trajectory to the buffer and reset.
        if done:
            self.buffer.append(list(self.current_trajectory))  # Store as a list to keep the trajectory together
            self.start_trajectory()  # Prepare for a new trajectory

    def sample(self, num_trajectories):
        # Make sure we don't sample more trajectories than we have
        num_trajectories = min(num_trajectories, len(self.buffer))

        sampled_trajectories = random.sample(self.buffer, num_trajectories)
        # Flatten the list of sampled trajectories if you need a continuous list
        flattened_samples = [experience for trajectory in sampled_trajectories for experience in trajectory]
        observations, remain_time, actions, action_nums, reward, next_observations, next_remain_time, \
        next_action_nums, done, probs = zip(*flattened_samples)

        frequencies = [self.state_frequencies[str(obs)] for obs in observations]
        max_frequency = max(self.state_frequencies.values(), default=1)
        normalized_frequencies = [self.state_frequencies[str(obs)] / max_frequency for obs in observations]
        weights = [1.0 / freq for freq in normalized_frequencies]
        c, d = 0.02, 1.
        # if np.max(weights)/np.min(weights) > d/c:
        a, b = np.min(weights), np.max(weights)
        scaled_weights = c + (weights - a) * (d - c) / (b - a + 0.000001)
        # else:
        #     scaled_num = np.min(weights)/c
        #     scaled_weights = np.array(weights)/scaled_num
        # weight_sum = sum(weights)
        # scaled_weights = [w / weight_sum for w in weights]
        return observations, remain_time, actions, action_nums, reward, next_observations, next_remain_time, \
               next_action_nums, done, probs, scaled_weights

    def size(self):
        return len(self.buffer)

class ImportanceSamplingReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)
        self.current_trajectory = []  # Temporary storage for the current trajectory
        self.state_frequencies = collections.defaultdict(int)

    def start_trajectory(self):
        self.current_trajectory = []  # Reset the current trajectory

    def add(self, observations, remain_time, actions, action_nums, reward, next_observations,
            next_remain_time, next_action_nums, done, probs):
        # Add experience to the current trajectory
        self.state_frequencies[str(observations)] += 1
        self.current_trajectory.append((observations, remain_time, actions, action_nums, reward,
                                        next_observations, next_remain_time, next_action_nums, done, probs))
        # If 'done' is True, it means the trajectory has ended. Save the trajectory to the buffer and reset.
        if done:
            self.buffer.append(list(self.current_trajectory))  # Store as a list to keep the trajectory together
            self.start_trajectory()  # Prepare for a new trajectory

    def sample(self, num_trajectories):
        # Make sure we don't sample more trajectories than we have
        num_trajectories = min(num_trajectories, len(self.buffer))

        sampled_trajectories = random.sample(self.buffer, num_trajectories)
        # Flatten the list of sampled trajectories if you need a continuous list
        flattened_samples = [experience for trajectory in sampled_trajectories for experience in trajectory]
        observations, remain_time, actions, action_nums, reward, next_observations, next_remain_time, \
        next_action_nums, done, probs = zip(*flattened_samples)

        frequencies = [self.state_frequencies[str(obs)] for obs in observations]
        max_frequency = max(self.state_frequencies.values(), default=1)
        normalized_frequencies = [self.state_frequencies[str(obs)] / max_frequency for obs in observations]
        weights = [1.0 / freq for freq in normalized_frequencies]
        c, d = 0.02, 1.
        # if np.max(weights)/np.min(weights) > d/c:
        a, b = np.min(weights), np.max(weights)
        scaled_weights = c + (weights - a) * (d - c) / (b - a + 0.000001)
        # else:
        #     scaled_num = np.min(weights)/c
        #     scaled_weights = np.array(weights)/scaled_num
        return observations, remain_time, actions, action_nums, reward, next_observations, next_remain_time, \
               next_action_nums, done, probs, scaled_weights

    def size(self):
        return len(self.buffer)


def train_importance_sampling(env, agents, num_episodes, replay_buffer, minimal_size, batch_size):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                observations, remain_time, action_nums = env.reset()
                done = False
                while not done:
                    actions, probs = agents.take_actions(observations, remain_time, action_nums)
                    next_observations, reward, done, info = env.step(actions)
                    next_remain_time = info['remain_time']
                    next_action_nums = info['action_nums']
                    replay_buffer.add(observations, remain_time, actions, action_nums, reward, next_observations,
                                      next_remain_time, next_action_nums, done, probs)
                    observations = next_observations
                    action_nums = next_action_nums
                    remain_time = next_remain_time
                    if 'capture_reward' in info.keys():
                        episode_return += info['capture_reward']
                    else:
                        episode_return += reward
                if replay_buffer.size() >= minimal_size:
                    b_o, b_t, b_a, b_an, b_r, b_no, b_nt, b_nan, b_d, b_p, b_sw= replay_buffer.sample(batch_size)
                    transition_dict = {'observations': b_o, 'remain_time': b_t, 'actions': b_a, 'action_nums': b_an,
                                       'rewards': b_r, 'next_observations': b_no, 'next_remain_time': b_nt,
                                       'next_action_nums': b_nan, 'dones': b_d, 'sample_probs': b_p, 'weights':b_sw}
                    agents.update(transition_dict)
                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                                      'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
                if i_episode % (int(num_episodes / 30)) == 0 and i_episode != 0:
                    print("trajectories:", env.robot_trajectories)
    return return_list

def train_without_importance_sampling(env, agents, num_episodes, replay_buffer, minimal_size, batch_size):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                for _ in range(batch_size):
                    episode_return = 0
                    observations, remain_time, action_nums = env.reset()
                    done = False
                    while not done:
                        actions, probs = agents.take_actions(observations, remain_time, action_nums)
                        next_observations, reward, done, info = env.step(actions)
                        next_remain_time = info['remain_time']
                        next_action_nums = info['action_nums']
                        replay_buffer.add(observations, remain_time, actions, action_nums, reward, next_observations,
                                          next_remain_time, next_action_nums, done, probs)
                        observations = next_observations
                        action_nums = next_action_nums
                        remain_time = next_remain_time
                        if 'capture_reward' in info.keys():
                            episode_return += info['capture_reward']
                        else:
                            episode_return += reward
                for _ in range(env.remain_time_init):
                    b_o, b_t, b_a, b_an, b_r, b_no, b_nt, b_nan, b_d, b_p= replay_buffer.sample(batch_size)
                    transition_dict = {'observations': b_o, 'remain_time': b_t, 'actions': b_a, 'action_nums': b_an,
                                       'rewards': b_r, 'next_observations': b_no, 'next_remain_time': b_nt,
                                       'next_action_nums': b_nan, 'dones': b_d, 'sample_probs': b_p}
                    agents.update(transition_dict)
                replay_buffer.buffer.clear()
                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                                      'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
                if i_episode % (int(num_episodes / 30)) == 0 and i_episode != 0:
                    print("trajectories:", env.robot_trajectories)
    return return_list


def update_episode_return(episode_return, reward):
    # Check if the reward is a float
    if isinstance(reward, float):
        episode_return += reward
    # Check if the reward is a list
    elif isinstance(reward, list):
        # Calculate the sum of the list and add it to episode_return
        episode_return += sum(reward)
    return episode_return

# below is used to process the MuRESI problem.

def train_on_policy_mures1(env, agents, num_episodes, sample_size, compare_T):
    return_list = []
    compare_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                compare_return = 0
                episode_return = 0
                # individual: observations, actions, next_observations, action_num
                # team: remain_time, next_remain_time, rewards, capture_flag, dones
                transition_dict = {'observations': [],'cur_positions': [], 'possible_next_positions': [], 'actions': [],
                                   'action_nums': [] , 'next_observations': [], 'next_action_nums': [], 'rewards': [], 'dones': []}
                # print("trajectories:",env.robot_trajectories)
                for _ in range(sample_size):
                    time_step = 0
                    observations, info = env.reset()
                    action_nums = info['action_nums']
                    cur_positions = info['current_positions']
                    possible_next_positions = info['next_all_positions']
                    done = False
                    while not done:
                        actions, probs = agents.take_actions(observations, action_nums)
                        next_observations, reward, done, info = env.step(actions)
                        transition_dict['observations'].append(observations)
                        transition_dict['cur_positions'].append(cur_positions)
                        transition_dict['possible_next_positions'].append(possible_next_positions)
                        transition_dict['actions'].append(actions)
                        transition_dict['action_nums'].append(action_nums)
                        transition_dict['next_observations'].append(next_observations)
                        transition_dict['rewards'].append(reward)
                        transition_dict['dones'].append(done)
                        observations = next_observations
                        action_nums = info['action_nums']
                        transition_dict['next_action_nums'].append(action_nums)
                        cur_positions = info['current_positions']
                        possible_next_positions = info['next_all_positions']
                        episode_return = update_episode_return(episode_return, reward)
                        time_step += 1
                        if time_step <= compare_T:
                            compare_return = update_episode_return(compare_return, reward)
                return_list.append(episode_return/sample_size)
                compare_list.append(compare_return/sample_size)
                agents.update(transition_dict)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                                      'total_return': '%.3f' % np.mean(return_list[-10:]),
                                      'return': '%.3f' % np.mean(compare_list[-10:])})
                pbar.update(1)
                if i_episode % (int(num_episodes / 30)) == 0 and i_episode != 0:
                    print("trajectories:",env.robot_trajectories)
    return return_list, compare_list

class ReplayBufferMuRESI:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)
        self.current_trajectory = []  # Temporary storage for the current trajectory

    def start_trajectory(self):
        self.current_trajectory = []  # Reset the current trajectory

    def add(self, observations, actions, action_nums, reward, next_observations, next_action_nums, done):
        self.current_trajectory.append((observations, actions, action_nums, reward, next_observations,
                                        next_action_nums, done))
        if done:
            self.buffer.append(list(self.current_trajectory))  # Store as a list to keep the trajectory together
            self.start_trajectory()  # Prepare for a new trajectory

    def sample(self, num_trajectories):
        num_trajectories = min(num_trajectories, len(self.buffer))
        sampled_trajectories = random.sample(self.buffer, num_trajectories)
        flattened_samples = [experience for trajectory in sampled_trajectories for experience in trajectory]
        observations, actions, action_nums, reward, next_observations, \
        next_action_nums, done = zip(*flattened_samples)
        return observations, actions, action_nums, reward, next_observations, next_action_nums, done

    def size(self):
        return len(self.buffer)

def train_off_policy_mures1(env, agents, num_episodes, replay_buffer, minimal_size, batch_size, compare_T):
    compare_list = []
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                time_step = 0
                compare_return = 0
                episode_return = 0
                observations, info = env.reset()
                action_nums = info['action_nums']
                done = False
                while not done:
                    actions = agents.take_actions(observations, action_nums)
                    next_observations, reward, done, info = env.step(actions)
                    next_action_nums = info['action_nums']
                    replay_buffer.add(observations, actions, action_nums, reward, next_observations,
                                      next_action_nums, done)
                    observations = next_observations
                    action_nums = next_action_nums
                    episode_return = update_episode_return(episode_return, reward)
                # calculate compare_return:
                observations, info = env.reset()
                action_nums = info['action_nums']
                while time_step < compare_T:
                    actions = agents.take_max_actions(observations, action_nums, compare_T-time_step)
                    next_observations, reward, done, info = env.step(actions)
                    next_action_nums = info['action_nums']
                    observations = next_observations
                    action_nums = next_action_nums
                    compare_return = update_episode_return(compare_return, reward)
                    time_step += 1

                if replay_buffer.size() >= minimal_size:
                    b_o, b_a, b_an, b_r, b_no, b_nan, b_d= replay_buffer.sample(batch_size)
                    transition_dict = {'observations': b_o, 'actions': b_a, 'action_nums': b_an, 'rewards': b_r,
                                       'next_observations': b_no, 'next_action_nums': b_nan, 'dones': b_d}
                    agents.update(transition_dict)
                return_list.append(episode_return)
                compare_list.append(compare_return)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                                      'total_return': '%.3f' % np.mean(return_list[-10:]),
                                      'return': '%.3f' % np.mean(compare_list[-10:])})
                pbar.update(1)
                if i_episode % (int(num_episodes / 30)) == 0 and i_episode != 0:
                    print("trajectories:", env.robot_trajectories)
    return return_list, compare_list

def train_off_policy_mures2(env, agents, num_episodes, replay_buffer, minimal_size, batch_size, compare_T):
    compare_list = []
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                time_step = 0
                compare_return = 0
                episode_return = 0
                observations, info = env.reset()
                action_nums = info['action_nums']
                done = False
                while not done:
                    remain_capture_time = max(compare_T-time_step, 0)
                    actions = agents.take_actions(observations, action_nums, remain_capture_time)
                    next_observations, reward, done, info = env.step(actions)
                    next_action_nums = info['action_nums']
                    replay_buffer.add(observations, actions, action_nums, reward, next_observations,
                                      next_action_nums, done)
                    observations = next_observations
                    action_nums = next_action_nums
                    episode_return = update_episode_return(episode_return, reward)
                    time_step += 1
                # calculate compare_return:
                observations, info = env.reset()
                action_nums = info['action_nums']
                time_step = 0
                while time_step < compare_T:
                    actions = agents.take_max_actions(observations, action_nums, compare_T-time_step)
                    next_observations, reward, done, info = env.step(actions)
                    next_action_nums = info['action_nums']
                    observations = next_observations
                    action_nums = next_action_nums
                    compare_return = update_episode_return(compare_return, reward)
                    time_step += 1

                if replay_buffer.size() >= minimal_size:
                    b_o, b_a, b_an, b_r, b_no, b_nan, b_d= replay_buffer.sample(batch_size)
                    transition_dict = {'observations': b_o, 'actions': b_a, 'action_nums': b_an, 'rewards': b_r,
                                       'next_observations': b_no, 'next_action_nums': b_nan, 'dones': b_d}
                    agents.update(transition_dict)
                return_list.append(episode_return)
                compare_list.append(compare_return)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                                      'total_return': '%.3f' % np.mean(return_list[-10:]),
                                      'return': '%.3f' % np.mean(compare_list[-10:])})
                pbar.update(1)
                if i_episode % (int(num_episodes / 30)) == 0 and i_episode != 0:
                    print("trajectories:", env.robot_trajectories)
    return return_list, compare_list