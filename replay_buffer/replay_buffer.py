import torch
import numpy as np
from replay_buffer.SumTree import SumTree

class ReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        """
        Prioritized Replay Buffer with Sum-Tree for efficient sampling.
        :param capacity: Maximum number of transitions.
        :param alpha: Priority exponent (controls how much prioritization is used).
        """
        self.tree = SumTree(capacity)
        self.alpha = alpha  # Prioritization exponent

    def store(self, obs: torch.Tensor, action: torch.Tensor, reward: torch.Tensor, next_obs: torch.Tensor,
              terminated: torch.Tensor):
        """
        Add a new transition to the replay buffer with maximum priority.
        :param transition: (state, action, reward, next_state, done)
        """
        obs = torch.tensor(obs)
        action = torch.tensor(action)
        reward = torch.tensor(reward)
        next_obs = torch.tensor(next_obs)
        terminated = torch.tensor(terminated)

        transition = (obs, action, reward, next_obs, terminated)
        # Max priority is the highest among the sum tree leaves
        init_priority = np.max(self.tree.tree[self.tree.capacity - 1:self.tree.capacity - 1 + self.tree.size]) if self.tree.size > 0 else 1.0  # Initial priority (1.0) when tree is empty
        self.tree.add(init_priority, transition)

    def sample(self, batch_size, beta=0.4):
        """
        Sample a batch of transitions.
        :param batch_size: Number of samples to return.
        :param beta: Importance-sampling exponent.
        :return: (batch, indices, weights)
        """
        batch = []
        indices = []
        priorities = []
        segment = self.tree.total_priority / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            value = np.random.uniform(a, b)

            index, priority, data = self.tree.get_leaf(value)
            batch.append(data)
            indices.append(index)
            priorities.append(priority)

        # A priority spans a portion in the cumulative priority for a transition
        # Therefore this priority divided by the cumulative priority would represent
        # the portion for a transition probability wrt total probability
        sampling_probabilities = np.array(priorities) / self.tree.total_priority
        # Importance sampling weights
        weights = (1 / (len(self.tree.data) * sampling_probabilities)) ** beta
        weights /= weights.max()

        grouped = list(zip(*batch))

        obs_batch = torch.stack(grouped[0], dim=0)
        action_batch = torch.stack(grouped[1], dim=0)
        reward_batch = torch.stack(grouped[2], dim=0)
        next_obs_batch = torch.stack(grouped[3], dim=0)
        terminated_batch = torch.stack(grouped[4], dim=0)

        return (obs_batch, action_batch, reward_batch, next_obs_batch, terminated_batch), indices, weights

    def update_priorities(self, indices, priorities):
        """
        Update the priorities of sampled transitions.
        :param indices: Indices of transitions to update.
        :param priorities: New priority values.
        """
        for index, priority in zip(indices, priorities):
            self.tree.update(index, (priority + 1e-5) ** self.alpha)