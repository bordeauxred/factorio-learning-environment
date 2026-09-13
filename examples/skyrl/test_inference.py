import os
import sys
from typing import List, Dict

# Ensure skyrl_gym is available
try:
    import skyrl_gym
    from skyrl_gym.envs.registration import register
except ImportError:
    print("Error: skyrl-gym not found. Please install it first.")
    sys.exit(1)

# Ensure FLE is available
try:
    import fle
except ImportError:
    print("Error: factorio-learning-environment not found. Please install it in editable mode.")
    sys.exit(1)

# 1. Manually register if not already in skyrl-gym
try:
    register(
        id="factorio",
        entry_point="fle.env.skyrl_wrapper:FactorioEnv",
    )
    print("Registered 'factorio' environment.")
except Exception as e:
    print(f"Note: Environment might already be registered: {e}")

def test_inference():
    print("\n--- Starting Factorio Inference Test ---")
    
    # 2. Create environment
    try:
        env = skyrl_gym.make("factorio", env_id="iron_ore_throughput", max_turns=5)
        print("Environment created successfully.")
    except Exception as e:
        print(f"Error creating environment: {e}")
        print("Make sure Factorio server is running on port 27000.")
        return

    # 3. Initialize
    prompt = [{"role": "system", "content": "You are a Factorio expert. Produce Python code to build a miner."}]
    obs, info = env.init(prompt)
    print(f"Initial observation received. Length: {len(obs)}")
    print(f"Last message: {obs[-1]['content'][:100]}...")

    # 4. Step
    sample_action = "game.player.insert{name='iron-plate', count=10}" # Simple test action
    print(f"\nTaking step with action: {sample_action}")
    
    try:
        output = env.step(sample_action)
        print(f"Step successful!")
        print(f"Reward: {output.reward}")
        print(f"Done: {output.done}")
        if output.observations:
            print(f"New observation: {output.observations[-1]['content'][:100]}...")
    except Exception as e:
        print(f"Error during step: {e}")

    # 5. Close
    env.close()
    print("\n--- Test Complete ---")

if __name__ == "__main__":
    test_inference()
