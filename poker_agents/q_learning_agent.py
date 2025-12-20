"""Implement q-learning agent for Texas Hold em

Code adapted from https://gymnasium.farama.org/introduction/train_agent/#about-the-environment-blackjack
to work with Texas Hold em
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import gymnasium as gym
from collections import defaultdict
from tqdm import tqdm 
from matplotlib import pyplot as plt
import gymnasium as gym
import numpy as np
import random

from poker_agents import Action, TexasHoldemEnv


class RandomAgent:
    def get_action(self, obs: np.ndarray) -> int:
        """Return random action."""
        return random.choice([Action.FOLD, Action.CALL, Action.RAISE])


class PokerAgent:
    def __init__(
            self,
            env: gym.Env,
            learning_rate: float,
            initial_epsilon: float,
            epsilon_decay: float,
            final_epsilon: float,
            discount_factor: float = 0.95,
    ):
        """Initialize a Q-Learning agent.

        Args:
            env: The training environment
            learning_rate: How quickly to update Q-values (0-1)
            initial_epsilon: Starting exploration rate (usually 1.0)
            epsilon_decay: How much to reduce epsilon each episode
            final_epsilon: Minimum exploration rate (usually 0.1)
            discount_factor: How much to value future rewards (0-1)
        """
        self.env = env

        # Q-table: maps (state, action) to expected reward
        # defaultdict automatically creates entries with zeros for new states
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        self.lr = learning_rate
        self.discount_factor = discount_factor 

        # Exploration parameters
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        # Track learning progress
        self.training_error = []
    
    def get_action(self, obs: np.ndarray) -> int: 
        """Choose an action using epsilon-greedy strategy.

        Returns:
            action: 0 (fold) 1 (call) 2 (raise)
        """
        # With probability epsilon: explore (random action)
        if np.random.random() < self.epsilon: # np.random.random returns [0.0, 1.0)
            return self.env.action_space.sample()
        
        # With probability (1-epsilon): exploit (best known action)
        else:
            state = (obs[1], obs[2], obs[8]) # player_hand_rank, player_hand_high, is_preflop
            return int(np.argmax(self.q_values[state]))
        
    def update(
            self,
            obs: np.ndarray,
            action: int,
            reward: float,
            terminated: bool,
            next_obs: np.ndarray
    ):
        """Update Q-value based on experience.
        
        This is the heart of Q-learning: learn from (state, action, reward, next_state)

        Args:
            obs: state
            action: action
            reward: reward from the given action at the given state
            terminated: whether the hand is over
            next_obs: next state after taking the given action
        """
        # What's the best we could do from the next state?
        # (Zero if episode terminated - no future rewards possible)
        next_state = (next_obs[1], next_obs[2], next_obs[8])
        future_q_value = (not terminated) * np.max(self.q_values[next_state])

        # What should the Q-value be? (Bellman equation)
        target = reward + self.discount_factor * future_q_value

        # How wrong was our current estimate?
        state = (obs[1], obs[2], obs[8])
        temporal_difference = target - self.q_values[state][action]

        # Update our estimate in the direction of the error
        # Learning rate controls how big steps we take
        self.q_values[state][action] = (
            self.q_values[state][action] + self.lr * temporal_difference
        )

        # Track learning progress (useful for debugging)
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        """Reduce exploration rate after each episode."""
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)
        

def train():   
    # Training hyperparameters
    learning_rate = 0.01
    n_episodes = 100_000    # Number of hands to practice
    start_epsilon = 1.0     # Start with 100% random actions
    epsilon_decay = start_epsilon / (n_episodes / 2)
    final_epsilon = 0.1     # Always keep some exploration

    # Create environment
    env = TexasHoldemEnv(
        num_players=2,
        initial_stack=40,
        small_blind=10,
        big_blind=20,
        raise_amount=20,
        seed=42,
        render_mode="ansi",
    )
    env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

    # Create agent
    agent = PokerAgent(
        env=env,
        learning_rate=learning_rate,
        initial_epsilon=start_epsilon,
        epsilon_decay=epsilon_decay,
        final_epsilon=final_epsilon,
    )

    # Implement the training loop
    for episode in tqdm(range(n_episodes)):
        # Start a new hand
        obs, info = env.reset()
        done = False

        # Play one complete hand
        while not done:
            # Agent chooses action (initially random, gradually more intelligent)
            action = agent.get_action(obs)

            # Take action and observe result
            next_obs, reward, terminated, truncated, info = env.step(action)

            # Learn from this experience
            agent.update(obs, action, reward, terminated, next_obs)

            done = terminated or truncated
            obs = next_obs
        # Reduce exploration rate (agent becomes less random over time)
        agent.decay_epsilon()
    return env, agent


def get_moving_avgs(arr, window, convolution_mode):
    """Compute moving average to smooth noisy data."""
    return np.convolve(
        np.array(arr).flatten(),
        np.ones(window),
        mode=convolution_mode
    ) / window


def analyze_training(env, agent):
    # smooth over a 500-episode window
    rolling_length = 500
    fig, axs = plt.subplots(ncols=3, figsize=(12,5))

    # Episode rewards (win/loss performance)
    axs[0].set_title("Episode rewards")
    reward_moving_average = get_moving_avgs(
    env.return_queue,
    rolling_length,
    "valid"
    )
    axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
    axs[0].set_ylabel("Average Reward")
    axs[0].set_xlabel("Episode")

    # Episode lengths (how many actions per hand)
    axs[1].set_title("Episode lengths")
    length_moving_average = get_moving_avgs(
        env.length_queue,
        rolling_length,
        "valid"
    )
    axs[1].plot(range(len(length_moving_average)), length_moving_average)
    axs[1].set_ylabel("Average Episode Length")
    axs[1].set_xlabel("Episode")

    # Training error (how much we're still learning)
    axs[2].set_title("Training Error")
    training_error_moving_average = get_moving_avgs(
        agent.training_error,
        rolling_length,
        "same"
    )
    axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)
    axs[2].set_ylabel("Temporal Difference Error")
    axs[2].set_xlabel("Step")

    plt.tight_layout()
    plt.savefig("training_analysis.png")


# Test the trained agent
def test_agent(env, agent, num_episodes=1000):
    """Test agent performance without learning or exploration."""
    total_rewards = []

    # Temporarily disable exploration for testing
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0  # Pure exploitation
    max_steps = 500

    for _ in tqdm(range(num_episodes)):
        obs, info = env.reset()
        episode_reward = 0
        done = False

        steps = 0

        while not done:
            # get agent action
            action = agent.get_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            done = terminated or truncated or (steps >= max_steps)


        total_rewards.append(episode_reward)

    # Restore original epsilon
    agent.epsilon = old_epsilon

    win_rate = np.mean(np.array(total_rewards) > 0)
    average_reward = np.mean(total_rewards)

    print(f"Test Results over {num_episodes} episodes:")
    print(f"Win Rate: {win_rate:.1%}")
    print(f"Average Reward: {average_reward:.3f}")
    print(f"Standard Deviation: {np.std(total_rewards):.3f}")


def main():
    env, agent = train()
    analyze_training(env, agent)
    test_agent(env, agent)


if __name__ == "__main__":
    main()