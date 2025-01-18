from matplotlib import pyplot as plt
import pandas as pd

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