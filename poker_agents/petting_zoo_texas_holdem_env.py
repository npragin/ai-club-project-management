from pettingzoo.classic import texas_holdem_v4
import numpy as np


"""
Wrap the texas_holdem_v4 environment to include an ansi
option for rendering and make ___ changes to the state representation?

TODO: update observation and/or info to be more human readable and 
easy to understand for others
"""
    

class TexasHoldEm():
    """
    Wrapper of texas_holdem_v4 which is a wrapper of RLCardGame
    Structure: AECEnv -> texas_holdem_v4 -> RLCardGame
    """
    # Define readable constants following RLCard's implementation
    CALL = 0
    RAISE = 1
    FOLD = 2
    CHECK = 3

    ACTION_NAMES = {CALL: "Call", RAISE: "Raise", FOLD: "Fold", CHECK: "Check"}

    def __init__(self, num_players=2, render_mode="ansi"):
        self.env = texas_holdem_v4.env(num_players=num_players, render_mode=render_mode)
        self.env.reset()

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
    
    def step(self, action):
        return self.env.step(action)
    
    def close(self):
        return self.env.close()
    
    @property
    def agent_iter(self):
        return self.env.agent_iter()
    
    @property
    def action_space(self):
        return self.env.action_space

    def last(self):
        observation, reward, termination, truncation, info = self.env.last()

        # Access the internal RLCard game state
        internal_game = self.env.unwrapped.env.game
        state = internal_game.get_state(self.env.unwrapped.env.game.get_player_id())

        info["hand_cards"] = state["hand"]
        
        return observation, reward, termination, truncation, info

    def render(self):
        if self.env.render_mode == "ansi":
            game_engine = self.env.unwrapped.env.game
            player_id = game_engine.get_player_id()
            return game_engine.get_state(player_id)
        return self.env.render()
