"""Implement Proximal Policy Optimization agent for Texas Hold em
using skrl library
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np

from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, CategoricalMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

from poker_agents import Action, TexasHoldemEnv


# seed for reproducibility
set_seed()  # e.g. `set_seed(42)` for fixed seed


# define models (stochastic and deterministic models) using mixins
class Policy(CategoricalMixin, Model):
    def __init__(self, observation_space, action_space, device=None, unnormalized_log_prob=True):
        Model.__init__(self, observation_space, action_space, device)
        CategoricalMixin.__init__(self, unnormalized_log_prob)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 64),
                                 nn.ReLU(),
                                 nn.Linear(64, 64),
                                 nn.ReLU(),
                                 nn.Linear(64, self.num_actions))
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role): 
        """Return the action the policy learned to take"""
        return self.net(inputs["states"]), {}

class Value(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 64),
                                 nn.ReLU(),
                                 nn.Linear(64, 64),
                                 nn.ReLU(),
                                 nn.Linear(64, 1))
        
    def compute(self, inputs, role):
        """Return the estimated value for the given state"""
        return self.net(inputs["states"]), {}

def main():
    # load and wrap Poker Environment
    env = TexasHoldemEnv(
        num_players=2,
        initial_stack=1000,
        small_blind=10,
        big_blind=20,
        raise_amount=20,
        seed=42,
        render_mode="ansi",
    )
    env = wrap_env(env)
    device = env.device

    # instantiate a memory as rollout buffer (any memory can be used for this)
    memory = RandomMemory(memory_size=1024, num_envs=env.num_envs, device=device)

    # instantiate the agent's models (function approximators)
    models = {}
    models["policy"] = Policy(env.observation_space, env.action_space, device)
    models["value"] = Value(env.observation_space, env.action_space, device)


    # configure and instantiate the agent
    cfg = PPO_DEFAULT_CONFIG.copy()
    cfg["rollouts"] = 1024  # memory_size
    cfg["learning_epochs"] = 10
    cfg["mini_batches"] = 32
    cfg["discount_factor"] = 0.9
    cfg["lambda"] = 0.95
    cfg["learning_rate"] = 1e-3
    cfg["learning_rate_scheduler"] = KLAdaptiveRL
    cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.008}
    cfg["grad_norm_clip"] = 0.5
    cfg["ratio_clip"] = 0.2
    cfg["value_clip"] = 0.2
    cfg["clip_predicted_values"] = False
    cfg["entropy_loss_scale"] = 0.0
    cfg["value_loss_scale"] = 0.5
    cfg["kl_threshold"] = 0
    cfg["mixed_precision"] = True
    cfg["state_preprocessor"] = RunningStandardScaler
    cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
    cfg["value_preprocessor"] = RunningStandardScaler
    cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}
    # logging to TensorBoard and write checkpoints (in timesteps)
    cfg["experiment"]["write_interval"] = 500
    cfg["experiment"]["checkpoint_interval"] = 5000
    cfg["experiment"]["directory"] = "runs/torch/Pendulum"

    agent = PPO(models=models,
                memory=memory,
                cfg=cfg,
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=device)


    # configure and instantiate the RL trainer
    cfg_trainer = {"timesteps": 1000, "headless": True}
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=[agent])

    # start training
    trainer.train()


if __name__ == "__main__":
    main()