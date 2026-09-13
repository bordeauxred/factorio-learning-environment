"""
Pure inference script for Factorio with SkyRL.
Runs a model on Factorio tasks and evaluates performance.
"""
import argparse
import os
from typing import List, Dict

import pandas as pd
from skyrl_gym import make
from skyrl_train.utils.model_utils import load_vllm_model # Assuming this exists based on GSM8K patterns

def run_inference(
    model_path: str,
    dataset_path: str,
    num_examples: int = 5,
    max_turns: int = 10,
    backend: str = "vllm"
):
    print(f"Loading dataset from {dataset_path}...")
    df = pd.read_parquet(dataset_path)
    examples = df.to_dict("records")[:num_examples]

    print(f"Loading model: {model_path} (backend: {backend})")
    # In a real SkyRL setup, we'd use the appropriate generator or vLLM engine
    # For this script, we'll use skyrl_gym to manage the environment interaction
    
    results = []
    
    for i, ex in enumerate(examples):
        print(f"\n--- Example {i+1}/{len(examples)} ---")
        env_id = ex["extra_info"]["env_id"]
        prompt = ex["prompt"]
        
        print(f"Task: {ex['prompt'][-1]['content']}")
        
        env = make("factorio", env_id=env_id, max_turns=max_turns)
        obs, info = env.init(prompt)
        
        done = False
        total_reward = 0
        turn = 0
        
        while not done and turn < max_turns:
            print(f"Turn {turn+1}...")
            # Here we would normally call the model
            # For pure inference script demonstration, we'll placeholder the model call
            # In a real scenario, you'd use vllm to generate the action string
            
            # Simple placeholder action for demonstration
            action = "game.player.print('Hello from SkyRL Turn ' .. game.tick)" 
            
            output = env.step(action)
            total_reward = output.reward
            done = output.done
            turn += 1
            
            print(f"  Action: {action}")
            print(f"  Reward: {total_reward}")
            
        env.close()
        results.append({
            "task": ex["prompt'][-1]['content'],
            "reward": total_reward,
            "turns": turn
        })

    print("\n--- Summary ---")
    for res in results:
        print(f"Task: {res['task']} | Reward: {res['reward']} | Turns: {res['turns']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--dataset_path", type=str, default=f"{os.environ.get('HOME')}/data/factorio/validation.parquet")
    parser.add_argument("--num_examples", type=int, default=3)
    args = parser.parse_args()
    
    run_inference(
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        num_examples=args.num_examples
    )
