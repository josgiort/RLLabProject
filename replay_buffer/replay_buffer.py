import random

import torch

class ReplayBuffer:
    def __init__(self, max_size: int):
        """
        Create the replay buffer.

        :param max_size: Maximum number of transitions in the buffer.
        """
        self.data = []
        self.max_size = max_size
        self.position = 0

    def __len__(self) -> int:
        """Returns how many transitions are currently in the buffer."""
        # TODO: Your code here
        return len(self.data)

    def store(self, obs: torch.Tensor, action: torch.Tensor, reward: torch.Tensor, next_obs: torch.Tensor,
              terminated: torch.Tensor):
        """
        Adds a new transition to the buffer. When the buffer is full, overwrite the oldest transition.

        :param obs: The current observation.
        :param action: The action.
        :param reward: The reward.
        :param next_obs: The next observation.
        :param terminated: Whether the episode terminated.
        """
        # TODO: Your code here
        obs = torch.tensor(obs)
        action = torch.tensor(action)
        reward = torch.tensor(reward)
        next_obs = torch.tensor(next_obs)
        terminated = torch.tensor(terminated)

        # If buffer is not full then just append the transitions
        if self.__len__() < self.max_size:
            self.data.append((obs, action, reward, next_obs, terminated))
        # Else don't append but look for the oldest transition and replace it with the new one
        else:
            # self.position marks the oldest transition and, since appending always sends new transitions to
            # right end of list then the oldest one is initially at position 0
            if self.position == self.max_size:
                self.position = 0
            # As we replace the oldest one, the new oldest one is just at the right index from self.position
            # So we update it by adding to this pointer, except when we reach max possible index, then we go back to 0
            self.data[self.position] = (obs, action, reward, next_obs, terminated)
            self.position += 1

    def sample(self, batch_size: int) -> torch.Tensor:
        """
        Sample a batch of transitions uniformly and with replacement. The respective elements e.g. states, actions, rewards etc. are stacked

        :param batch_size: The batch size.
        :returns: A tuple of tensors (obs_batch, action_batch, reward_batch, next_obs_batch, terminated_batch), where each tensors is stacked.
        """
        # TODO: Your code here
        # Sample uniformly with replacement for a batch size
        samples = random.choices(self.data, k=batch_size)
        grouped = list(zip(*samples))

        obs_batch = torch.stack(grouped[0], dim=0)
        action_batch = torch.stack(grouped[1], dim=0)
        reward_batch = torch.stack(grouped[2], dim=0)
        next_obs_batch = torch.stack(grouped[3], dim=0)
        terminated_batch = torch.stack(grouped[4], dim=0)

        return (obs_batch, action_batch, reward_batch, next_obs_batch, terminated_batch)