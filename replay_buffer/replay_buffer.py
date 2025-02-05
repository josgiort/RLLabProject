import torch
import numpy as np
from collections import deque
from replay_buffer.SumTree import SumTree

class ReplayBuffer:
    def __init__(self, capacity, n_step=1, gamma=0.99, alpha=0.6):
        self.tree = SumTree(capacity)
        self.alpha = alpha  # Prioritization exponent
        self.capacity = capacity
        self.n_step = n_step
        self.gamma = gamma

        # Multi-step storage buffer
        self.n_step_buffer = deque(maxlen=n_step)

    def _get_n_step_transition(self):
        """
        Compute the n-step return from the stored transitions in the temporal n_step buffer
        """
        obs, action, _, _, _ = self.n_step_buffer[0]  # First transition's state and action
        total_reward = 0
        next_obs_n = self.n_step_buffer[-1][3]  # Last next_state
        done_flag = self.n_step_buffer[-1][4]  # Last done

        for i, (_, _, reward, state_look_ahead, done_look_ahead) in enumerate(self.n_step_buffer):
            total_reward += (self.gamma ** i) * reward
            if done_look_ahead:
                next_obs_n = state_look_ahead
                done_flag =  torch.tensor(True)
                break
            else :
                next_obs_n = state_look_ahead
                done_flag = torch.tensor(False)
        return (obs, action, total_reward, next_obs_n, done_flag)

    def store(self, obs: torch.Tensor, action: torch.Tensor, reward: torch.Tensor, next_obs: torch.Tensor, terminated: torch.Tensor):
        obs = torch.tensor(obs)
        action = torch.tensor(action)
        reward = torch.tensor(reward)
        next_obs = torch.tensor(next_obs)
        terminated = torch.tensor(terminated)

        self.n_step_buffer.append((obs, action, reward, next_obs, terminated))

        # Only store n_step transition in the buffer and sume tree if there are enough
        # transitions in the temporal nstep buffer
        if len(self.n_step_buffer) == self.n_step:
            n_step_transition = self._get_n_step_transition()
            init_priority = np.max(self.tree.tree[self.tree.capacity - 1:self.tree.capacity - 1 + self.tree.size]) if self.tree.size > 0 else 1.0
            self.tree.add(init_priority, n_step_transition)

    def sample(self, batch_size, beta=0.4):
        batch = []
        indices = []
        priorities = []
        segment = self.tree.total_priority / batch_size

        # Segment the priority space and sample each batch element from this segments
        # This is done to avoid not sampling from the whole priority space
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            value = np.random.uniform(a, b)

            index, priority, data = self.tree.get_leaf(value)

            batch.append(data)
            indices.append(index)
            priorities.append(priority)

        #
        sampling_probabilities = np.array(priorities) / self.tree.total_priority
        weights = (1 / (len(self.tree.data) * sampling_probabilities)) ** beta
        weights /= weights.max()

        grouped = list(zip(*batch))

        obs_batch = torch.stack(grouped[0], dim=0)
        action_batch = torch.stack(grouped[1], dim=0)
        reward_batch = torch.stack(grouped[2], dim=0)
        next_obs_batch = torch.stack(grouped[3], dim=0)
        terminated_batch = torch.stack(grouped[4], dim=0)

        return (obs_batch, action_batch, reward_batch, next_obs_batch, terminated_batch), indices, torch.tensor(weights, dtype=torch.float32)

    def update_priorities(self, indices, priorities):
        for index, priority in zip(indices, priorities):
            self.tree.update(index, (priority + 1e-5) ** self.alpha)