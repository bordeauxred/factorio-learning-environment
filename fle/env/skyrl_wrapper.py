import gym
import numpy as np
from typing import Optional, Tuple, Dict, Any

from fle.env.gym_env.environment import FactorioGymEnv
from fle.env.instance import FactorioInstance
from fle.env.gym_env.action import Action
from fle.commons.models.game_state import GameState

class SkyRLFactorioWrapper(gym.Env):
    """
    A wrapper for FactorioGymEnv that adapts it for SkyrL (text-in/text-out).
    
    This wrapper:
    1. Flattens the dictionary observation into a single text prompt.
    2. Accepts text actions (Python code) and converts them to FLE Actions.
    """
    
    def __init__(
        self, 
        host: str = "localhost", 
        port: int = 27000, 
        agent_idx: int = 0,
        task_config: Optional[str] = None,
        # Add other FLE args as needed
    ):
        self.host = host
        self.port = port
        self.agent_idx = agent_idx
        
        # Initialize FLE instance connecting to the (potentially remote) server
        self.instance = FactorioInstance(
            address=host,
            tcp_port=port,
            num_agents=1, # Assuming single agent per wrapper for now
            # Other params might be needed depending on setup
        )
        
        # Initialize the base gym environment
        # Note: Task loading logic might need to be added here if task_config is provided
        self.env = FactorioGymEnv(self.instance)
        
        # Define Action Space: Text (Python Code)
        # SkyrL likely expects a Discrete or Text space, but for now we'll assume it handles raw text or we define a dummy space.
        # If SkyrL uses standard Gym spaces, we might need a custom Text space.
        # For now, we'll use a generic Box or Discrete as placeholder if Text isn't standard in the SkyrL version we target,
        # but based on "text-in/text-out", we'll assume the agent sends strings.
        # However, gym.Env requires action_space to be defined.
        self.action_space = gym.spaces.Text(max_length=10000)
        
        # Define Observation Space: Text (Prompt)
        self.observation_space = gym.spaces.Text(max_length=100000)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[str, Dict[str, Any]]:
        obs_dict = self.env.reset(options=options, seed=seed)
        text_obs = self._observation_to_text(obs_dict)
        return text_obs, {}

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        """
        Args:
            action (str): Python code to execute.
        """
        # Create FLE Action
        # We assume the agent just sends the code. 
        # Game state management (resetting to specific states) might be complex here, 
        # so we default to current state unless we want to expose that to SkyrL.
        fle_action = Action(
            agent_idx=self.agent_idx,
            code=action,
            game_state=None # We don't reset state every step usually in this flow
        )
        
        obs_dict, reward, terminated, truncated, info = self.env.step(fle_action)
        
        text_obs = self._observation_to_text(obs_dict)
        
        return text_obs, reward, terminated, truncated, info

    def _observation_to_text(self, obs: Dict[str, Any]) -> str:
        """
        Converts the FLE dictionary observation into a structured text prompt.
        """
        # Extract components
        task_info = obs.get("task_info", {})
        goal = task_info.get("goal_description", "No goal specified.") if task_info else "No goal specified."
        
        inventory = obs.get("inventory", [])
        inventory_str = ", ".join([f"{item['quantity']} {item['type']}" for item in inventory])
        if not inventory_str:
            inventory_str = "Empty"
            
        # Entities (simplified for text)
        entities = obs.get("entities", [])
        # entities is a list of repr strings, so we can join them or summarize
        # If too long, we might need to truncate
        entities_str = "\n".join(entities[:50]) # Limit to 50 for now to avoid context overflow
        if len(entities) > 50:
            entities_str += f"\n... ({len(entities) - 50} more entities)"
            
        last_output = obs.get("raw_text", "")
        
        # Construct the prompt
        prompt = f"""
# Goal
{goal}

# Inventory
{inventory_str}

# Nearby Entities
{entities_str}

# Last Output
{last_output}

# Instructions
Write Python code to advance the goal.
"""
        return prompt.strip()

    def close(self):
        self.env.close()
