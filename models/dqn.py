import torch.nn.functional as F
import torch
from torch import nn

class DQN(nn.Module):
    def __init__(self, obs_shape: torch.Size, num_actions: int):

        """
        Initialize the Dueling DQN network.

        :param obs_shape: Shape of the observation space
        :param num_actions: Number of actions
        """
        super(DQN, self).__init__()


        # Convolutional feature extractor (same as your DQN)
        self.conv1 = nn.Conv2d(obs_shape[-1], 16, 5)
        self.conv2 = nn.Conv2d(16, 32, 3)

        # Compute the flattened size after convolutions
        self.flattened_size = 32 * 4 * 4  # Derived from MinAtar Breakout obs shape

        # Shared fully connected layer before splitting streams
        self.fc1 = nn.Linear(self.flattened_size, 128)

        # Value stream (outputs a single value V(s))
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Single output for state value
        )

        # Advantage stream (outputs values for each action A(s, a))
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions)  # One output per action
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Adjust input shape for PyTorch conv layers
        x = x.permute(0, 3, 1, 2)  # (batch, channels, height, width)

        # Feature extraction
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, 1)  # Flatten all dimensions except batch
        x = F.relu(self.fc1(x))

        # Compute value and advantage separately
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)

        # Combine streams: Q(s, a) = V(s) + A(s, a) - mean(A(s, a))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))

        return q_values


"""
class DQN(nn.Module):
    def __init__(self, obs_shape: torch.Size, num_actions: int):
        
        Initialize the DQN network.

        :param obs_shape: Shape of the observation space
        :param num_actions: Number of actions
        
        # obs_shape is the shape of a single observation -> use this information to define the dimensions of the layers
        super(DQN, self).__init__()
        # Get number of channels from obs_shape
        self.conv1 = nn.Conv2d(obs_shape[-1], 16, 5)
        # Output size 6*6 according to output dimensions DL formula
        self.conv2 = nn.Conv2d(16, 32, 3)
        # Output size 4*4
        self.fc1 = nn.Linear(32 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Your code here
        # Make input tensor in right shape for pytorch conv nets
        x = x.permute(0, 3, 1, 2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, 1)  # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
"""