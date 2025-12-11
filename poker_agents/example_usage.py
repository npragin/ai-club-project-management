"""Example usage of the Texas Hold'em environment."""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import gymnasium as gym

from poker_agents import Action, TexasHoldemEnv


def random_agent(obs: gym.Space) -> int:
    """Simple random agent for demonstration."""
    # Randomly choose an action
    import random

    return random.choice([Action.FOLD, Action.CALL, Action.RAISE])


def main() -> None:
    """Run a simple example game."""
    # Create environment
    env = TexasHoldemEnv(
        num_players=4,
        initial_stack=1000,
        small_blind=10,
        big_blind=20,
        raise_amount=20,
        seed=42,
        render_mode="ansi",
    )

    # Reset environment
    obs, info = env.reset()

    print("Starting Texas Hold'em Tournament")
    print("=" * 50)

    step_count = 0
    max_steps = 1000

    while step_count < max_steps:
        # Get action from agent for the current player
        # In self-play mode, the agent controls all players
        # In a real scenario, you'd use your trained model here
        # action = random_agent(obs)
        print(env.render())
        action = input(
            f"Player {info['current_player']}, enter action (FOLD=0, CALL=1, RAISE=2): "
        )
        action = int(action)

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)

        print(f"Reward: {reward}")

        step_count += 1

        if terminated or truncated:
            print("\nTournament ended!")
            print(f"Final stacks: {info['player_stacks']}")
            print(f"Current player when ended: {info['current_player']}")
            break

    print(f"\nTotal steps: {step_count}")


if __name__ == "__main__":
    main()
