import gym
from fle.env.skyrl_wrapper import SkyRLFactorioWrapper

# This is a placeholder for the actual SkyrL training loop.
# Since SkyrL library details are not fully known (no installed package to check),
# we provide a generic Gym-compatible training loop structure.

def main():
    # Configuration
    host = "localhost"
    port = 27000
    
    print(f"Connecting to Factorio at {host}:{port}...")
    
    # Initialize the environment
    env = SkyRLFactorioWrapper(host=host, port=port)
    
    print("Environment initialized.")
    
    # Reset environment
    obs, info = env.reset()
    print("Initial Observation:")
    print("-" * 20)
    print(obs)
    print("-" * 20)
    
    # Example interaction loop
    for i in range(5):
        # In a real scenario, the agent would generate this code based on the observation
        action = 'print("Hello from SkyrL Agent step {}")'.format(i)
        
        print(f"\nStep {i+1} Action: {action}")
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"Reward: {reward}")
        print(f"Terminated: {terminated}")
        print(f"Truncated: {truncated}")
        print("Observation:")
        print("-" * 20)
        print(obs[:500] + "..." if len(obs) > 500 else obs) # Truncate for display
        print("-" * 20)
        
        if terminated or truncated:
            print("Episode finished.")
            break
            
    env.close()

if __name__ == "__main__":
    main()
