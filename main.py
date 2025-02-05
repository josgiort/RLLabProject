import json

import torch

from agents.dqn_agent import DQNAgent
from utils.utils import display_performance
import gymnasium as gym

env = gym.make('MinAtar/Breakout-v1', render_mode="rgb_array")

# Print observation and action space infos
print(f"Training on {env.spec.id}")
print(f"Observation space: {env.observation_space}")
print(f"Action space: {env.action_space}\n")

# Hyperparameters, Hint: Change as you see fit
LR = 0.001
BATCH_SIZE = 8
REPLAY_BUFFER_SIZE = 100_000
UPDATE_FREQ = 100
EPS_START = 0.5
EPS_END = 0.05
SCHEDULE_DURATION = 125_000
NUM_EPISODES = 5_000
DISCOUNT_FACTOR = 0.99
"""
# Train DQN
agent = DQNAgent(
    env,
    gamma=DISCOUNT_FACTOR,
    lr=LR,
    batch_size=BATCH_SIZE,
    eps_start=EPS_START,
    eps_end=EPS_END,
    schedule_duration=SCHEDULE_DURATION,
    update_freq=UPDATE_FREQ,
    maxlen=REPLAY_BUFFER_SIZE,
)
stats = agent.train(NUM_EPISODES)

display_performance(stats)"""

import numpy as np
import os

num_runs = 5  # Run the experiment 5 times
all_stats = []

if not os.path.exists("results_integrated"):
    os.makedirs("results_integrated")
if not os.path.exists("models_integrated"):
    os.makedirs("models_integrated")

for run in range(num_runs):
    print(f"Starting run {run + 1}/{num_runs}...")

    # Create a new agent for each run
    agent = DQNAgent(
        env,
        gamma=DISCOUNT_FACTOR,
        lr=LR,
        batch_size=BATCH_SIZE,
        eps_start=EPS_START,
        eps_end=EPS_END,
        schedule_duration=SCHEDULE_DURATION,
        update_freq=UPDATE_FREQ,
        maxlen=REPLAY_BUFFER_SIZE,
    )

    stats = agent.train(NUM_EPISODES)

    all_stats.append(stats)

    # Save stats for this run
    with open(f"results_integrated/training_stats_run{run}.json", "w") as f:
        json.dump(stats, f)

    # Save trained model for this run
    torch.save(agent.q.state_dict(), f"models_integrated/dqn_trained_run{run}.pth")

# Save all results into a single file
np.save("results_integrated/all_training_stats.npy", all_stats)

print("All runs completed and saved!")
