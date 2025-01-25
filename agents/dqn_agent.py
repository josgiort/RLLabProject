import torch.nn.functional as F
import itertools

import numpy as np
import torch
import torch.optim as optim
import copy
from collections import namedtuple

from torch import nn

from models.dqn import DQN
from replay_buffer.replay_buffer import ReplayBuffer


def make_epsilon_greedy_policy(Q: nn.Module, num_actions: int):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon. Taken from last exercise with changes.

    :param Q: The DQN network.
    :param num_actions: Number of actions in the environment.

    :returns: A function that takes the observation as an argument and returns the greedy action in form of an int.
    """

    def policy_fn(obs: torch.Tensor, epsilon: float = 0.0):
        """This function takes in the observation and returns an action."""
        if np.random.uniform() < epsilon:
            return np.random.randint(0, num_actions)

        # For action selection, we do not need a gradient and so we call ".detach()"
        return Q(obs).argmax().detach().numpy()

    return policy_fn

def linear_epsilon_decay(eps_start: float, eps_end: float, current_timestep: int, duration: int) -> float:
    """
    Linear decay of epsilon.

    :param eps_start: The initial epsilon value.
    :param eps_end: The final epsilon value.
    :param current_timestep: The current timestep.
    :param duration: The duration of the schedule (in timesteps). So when schedule_duration == current_timestep, eps_end should be reached

    :returns: The current epsilon.
    """

    # TODO: Your code here
    if current_timestep <= duration:
        return eps_start - ((abs(eps_end - eps_start) / duration) * current_timestep)
    return eps_start - abs(eps_end - eps_start)


def update_dqn(
        q: nn.Module,
        q_target: nn.Module,
        optimizer: optim.Optimizer,
        gamma: float,
        obs: torch.Tensor,
        act: torch.Tensor,
        rew: torch.Tensor,
        next_obs: torch.Tensor,
        tm: torch.Tensor,
        weights: torch.Tensor
):
    """
    Update the DQN network for one optimizer step.

    :param weights:
    :param q: The DQN network.
    :param q_target: The target DQN network.
    :param optimizer: The optimizer.
    :param gamma: The discount factor.
    :param obs: Batch of current observations.
    :param act: Batch of actions.
    :param rew: Batch of rewards.
    :param next_obs: Batch of next observations.
    :param tm: Batch of termination flags.

    """
    # TODO: Zero out the gradient
    optimizer.zero_grad()  # zero the gradient buffers

    # TODO: Calculate the TD-Target
    with torch.no_grad():
        td_target = rew + gamma * q_target(next_obs).max(dim=1)[0] * (1 - tm.float())

    # TODO: Calculate the TD errors
    predicted_q_values = q(obs).gather(1, act.unsqueeze(1)).squeeze(1)  # Shape: [batch_size]
    td_errors = td_target - predicted_q_values  # Shape: [batch_size]

    # TODO: Calculate the loss with importance sampling weights
    loss = F.mse_loss(predicted_q_values, td_target, reduction='none')  # Per-sample loss
    weighted_loss = (loss * weights).mean()  # Apply importance weights

    # TODO: Backpropagate the weighted loss and step the optimizer
    weighted_loss.backward()
    optimizer.step()

    # Return the TD errors to update priorities later
    return td_errors.abs().detach()  # Use absolute TD errors for priorities


EpisodeStats = namedtuple("Stats", ["episode_lengths", "episode_rewards"])


class DQNAgent:
    def __init__(self,
                 env,
                 gamma=0.99,
                 lr=0.001,
                 batch_size=64,
                 eps_start=1.0,
                 eps_end=0.1,
                 schedule_duration=10_000,
                 update_freq=100,
                 maxlen=100_000,
                 ):
        """
        Initialize the DQN agent.

        :param env: The environment.
        :param gamma: The discount factor.
        :param lr: The learning rate.
        :param batch_size: Mini batch size.
        :param eps_start: The initial epsilon value.
        :param eps_end: The final epsilon value.
        :param schedule_duration: The duration of the schedule (in timesteps).
        :param update_freq: How often to update the Q target.
        :param max_size: Maximum number of transitions in the buffer.
        """

        self.env = env
        self.gamma = gamma
        self.batch_size = batch_size
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.schedule_duration = schedule_duration
        self.update_freq = update_freq

        # TODO: Initialize the Replay Buffer
        self.r_buffer = ReplayBuffer(maxlen)

        # TODO: Initialize the Deep Q-Network. Hint: Remember observation_space and action_space
        self.q = DQN(self.env.observation_space.shape, self.env.action_space.n)

        # TODO: Initialize the second Q-Network, the target network. Load the parameters of the first one into the second
        self.q_target = DQN(self.env.observation_space.shape, self.env.action_space.n)

        params_dqn = copy.deepcopy(self.q.state_dict())
        self.q_target.load_state_dict(params_dqn)

        # TODO: Create an ADAM optimizer for the Q-network
        self.optimizer = optim.Adam(self.q.parameters(), lr)

        self.policy = make_epsilon_greedy_policy(self.q, env.action_space.n)

    def train(self, num_episodes: int) -> EpisodeStats:
        """
        Train the DQN agent.

        :param num_episodes: Number of episodes to train.
        :returns: The episode statistics.
        """
        # Keeps track of useful statistics
        stats = EpisodeStats(
            episode_lengths=np.zeros(num_episodes),
            episode_rewards=np.zeros(num_episodes),
        )
        current_timestep = 0
        epsilon = self.eps_start

        for i_episode in range(num_episodes):
            # Print out which episode we're on, useful for debugging.
            if (i_episode + 1) % 100 == 0:
                print(
                    f'Episode {i_episode + 1} of {num_episodes}  Time Step: {current_timestep}  Epsilon: {epsilon:.3f}')

            # Reset the environment and get initial observation
            obs, _ = self.env.reset()

            for episode_time in itertools.count():
                # TODO: Get current epsilon value
                epsilon = linear_epsilon_decay(self.eps_start, self.eps_end, current_timestep, self.schedule_duration)

                # Choose action and execute
                action = self.policy(torch.as_tensor(obs).unsqueeze(0).float(), epsilon=epsilon)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)

                # Update statistics
                stats.episode_rewards[i_episode] += reward
                stats.episode_lengths[i_episode] += 1

                # TODO: Store sample in the replay buffer
                self.r_buffer.store(obs, action, reward, next_obs, terminated)

                # TODO: Sample a mini batch from the replay buffer
                sampled_batch, indices, weights  = self.r_buffer.sample(self.batch_size)
                obs_batch, act_batch, rew_batch, next_obs_batch, tm_batch = sampled_batch


                weights = torch.from_numpy(weights)

                # Update the Q network
                td_errors = update_dqn(
                    self.q,
                    self.q_target,
                    self.optimizer,
                    self.gamma,
                    obs_batch.float(),
                    act_batch,
                    rew_batch.float(),
                    next_obs_batch.float(),
                    tm_batch,
                    weights
                )
                # Just for readability
                updated_priorities = td_errors
                self.r_buffer.update_priorities(indices, updated_priorities.numpy())

                # Update the current Q target
                if current_timestep % self.update_freq == 0:
                    self.q_target.load_state_dict(self.q.state_dict())
                current_timestep += 1

                # Check whether the episode is finished
                if terminated or truncated or episode_time >= 500:
                    break
                obs = next_obs

        return stats