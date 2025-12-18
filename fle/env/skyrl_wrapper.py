"""
SkyRL-Gym compatible environment for Factorio Learning Environment.

This wrapper bridges FLE's FactorioGymEnv with SkyrL's BaseTextEnv interface.
"""
from typing import Dict, Any, Tuple, List

# SkyrL imports
try:
    from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput
except ImportError:
    # Fallback for development without skyrl-gym
    from dataclasses import dataclass
    
    @dataclass
    class BaseTextEnvStepOutput:
        observations: List[Dict[str, str]]
        reward: float
        done: bool
        metadata: Dict[str, Any] = None
    
    class BaseTextEnv:
        def __init__(self): pass
        def step(self, action: str) -> BaseTextEnvStepOutput: raise NotImplementedError
        def init(self, prompt) -> Tuple[List[Dict[str, str]], Dict[str, Any]]: return prompt, {}
        def close(self): pass

import gym
from fle.env.gym_env.action import Action
from fle.env.gym_env.observation_formatter import BasicObservationFormatter


class FactorioEnv(BaseTextEnv):
    """
    SkyrL-compatible wrapper for Factorio.
    
    Uses FLE's existing FactorioGymEnv and BasicObservationFormatter.
    """
    
    def __init__(self, env_id: str = "iron_ore_throughput", max_turns: int = 10, **kwargs):
        super().__init__()
        self.env_id = env_id
        self.max_turns = max_turns
        self.gym_env = None
        self.formatter = BasicObservationFormatter(include_research=False)
        self.turns = 0
    
    def init(self, prompt: List[Dict[str, str]]) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
        """Initialize environment with a prompt."""
        self.gym_env = gym.make(self.env_id)
        self.gym_env.reset()
        self.turns = 0
        return prompt, {}
    
    def step(self, action: str) -> BaseTextEnvStepOutput:
        """Execute action (Python code) and return observation."""
        self.turns += 1
        
        # Execute via FLE
        obs, reward, terminated, truncated, info = self.gym_env.step(
            Action(agent_idx=0, code=action)
        )
        
        # Format observation using FLE's formatter
        formatted = self.formatter.format(obs)
        
        done = terminated or truncated or self.turns >= self.max_turns
        
        return BaseTextEnvStepOutput(
            observations=[{"role": "user", "content": formatted.raw_str}] if not done else [],
            reward=float(reward),
            done=done,
            metadata={"info": info, "turns": self.turns}
        )
    
    def close(self):
        """Clean up resources."""
        if self.gym_env:
            self.gym_env.close()
            self.gym_env = None
