from tqdm import tqdm
import numpy as np
import torch
import collections
import random

def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size - 1, 2)
    begin = np.cumsum(a[:window_size - 1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))

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
        c, d = 0.1, 1.
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