import torch.nn.functional as F
import torch
from torch import nn

# Dueling network
class DQN(nn.Module):
    def __init__(self, obs_shape: torch.Size, num_actions: int):

        super(DQN, self).__init__()

        # Convolutional feature extractor
        self.conv1 = nn.Conv2d(obs_shape[-1], 16, 5)
        self.conv2 = nn.Conv2d(16, 32, 3)

        self.flattened_size = 32 * 4 * 4  # Derived from MinAtar Breakout obs shape

        self.fc1 = nn.Linear(self.flattened_size, 128)

        # Value stream of dueling
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Single output for state value
        )

        # Advantage stream of dueling
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions)  # One output per action
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Adjust input shape for PyTorch conv layers
        x = x.permute(0, 3, 1, 2)

        # Feature extraction
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))

        value = self.value_stream(x)
        advantage = self.advantage_stream(x)

        # Q(s, a) = V(s) + A(s, a) - mean(A(s, a))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))

        return q_values