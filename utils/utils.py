from matplotlib import pyplot as plt
import pandas as pd
"""
def display_performance(stats) :
    smoothing_window = 20
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), tight_layout=True)

    # Plot the episode length over time
    ax = axes[0]
    ax.plot(stats.episode_lengths)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Length")
    ax.set_title("Episode Length over Time")

    # Plot the episode reward over time
    ax = axes[1]
    rewards_smoothed = pd.Series(stats.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    ax.plot(rewards_smoothed)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Reward (Smoothed)")
    ax.set_title(f"Episode Reward over Time\n(Smoothed over window size {smoothing_window})")
    plt.show()
    """

from matplotlib import pyplot as plt
import pandas as pd


def display_performance(mean_lengths, std_lengths, mean_rewards, std_rewards):
    smoothing_window = 20
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), tight_layout=True)

    # Plot Episode Length with std deviation
    ax = axes[0]
    ax.plot(mean_lengths, label="Mean Episode Length")
    ax.fill_between(range(len(mean_lengths)), mean_lengths - std_lengths, mean_lengths + std_lengths, alpha=0.2)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Length")
    ax.set_title("Average Episode Length over Time")
    ax.legend()

    # Plot Smoothed Episode Reward with std deviation
    ax = axes[1]
    rewards_smoothed = pd.Series(mean_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    std_smoothed = pd.Series(std_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()

    ax.plot(rewards_smoothed, label="Mean Reward (Smoothed)")
    ax.fill_between(range(len(rewards_smoothed)), rewards_smoothed - std_smoothed, rewards_smoothed + std_smoothed,
                    alpha=0.2)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Reward (Smoothed)")
    ax.set_title(f"Average Episode Reward over Time\n(Smoothed over {smoothing_window}-episode window)")
    ax.legend()

    plt.show()
