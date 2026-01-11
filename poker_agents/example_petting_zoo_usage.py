import numpy as np 
from petting_zoo_texas_holdem_env import TexasHoldEm


class RandomAgent:
    def act(self, observation, env):
        """Returns a random index where the mask is 1"""
        action_mask = observation["action_mask"]
        valid_actions = np.flatnonzero(action_mask)     
        return np.random.choice(valid_actions)
    
class AlwaysFoldAgent:
    def act(self, observation, env: TexasHoldEm):
        """
        In PettingZoo Texas Hold'em
        action mask: [Call, Raise, Fold, Check]
        """
        action_mask = observation["action_mask"]
        valid_actions = np.flatnonzero(action_mask)  
        # Always try to Fold if it's legal.
        # If not, pick the first available legal action   
        if env.FOLD in valid_actions:
            return env.FOLD
        return valid_actions[0]
    
    
env = TexasHoldEm(num_players=2, render_mode="ansi")
env.reset(seed=42)

# define policies for each agent
# Petting Zoo uses the "player_idx" to determine each agent
policies = {
    "player_0": RandomAgent(),
    "player_1": AlwaysFoldAgent()
}

for agent in env.agent_iter:
    print(env.render())
    observation, reward, termination, truncation, info = env.last()
    print(observation)

    if termination or truncation:
        action = None
    else:
        policy = policies[agent]

        action = policy.act(observation, env)
        print(f"Agent {agent} chose to {env.ACTION_NAMES[action]}")

    env.step(action)
env.close()