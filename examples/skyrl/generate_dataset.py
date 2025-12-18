"""
Dataset generator for Factorio RL training with SkyrL.
"""
import argparse
from pathlib import Path
import pandas as pd


SYSTEM_PROMPT = """You are an expert Factorio player and Python programmer.

You control a Factorio factory by writing Python code. Available functions include:
- place_entity(entity, direction, position) - Place buildings
- harvest_resource(position, count) - Mine resources
- craft_item(item, count) - Craft items
- insert_item(entity, item, count) - Insert items into machines
- move_to(position) - Move player
- get_entities() - Get nearby entities
- inspect_inventory() - Check inventory

Write Python code to accomplish the task. Your code will be executed directly."""


TASKS = [
    {"task": "Mine 50 iron ore", "env_id": "iron_ore_throughput", "quota": 50},
    {"task": "Set up iron plate smelting", "env_id": "iron_plate_throughput", "quota": 20},
    {"task": "Automate copper mining", "env_id": "copper_ore_throughput", "quota": 50},
    {"task": "Build a coal-powered furnace setup", "env_id": "iron_plate_throughput", "quota": 30},
]


def generate_examples(num_examples: int, split: str) -> list:
    examples = []
    for i in range(num_examples):
        task = TASKS[i % len(TASKS)]
        example = {
            "data_source": "factorio",
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Task: {task['task']}\nGoal: Achieve {task['quota']} throughput."},
            ],
            "env_class": "factorio",
            "reward_spec": {"method": "rule", "ground_truth": task["quota"]},
            "extra_info": {"env_id": task["env_id"], "split": split},
        }
        examples.append(example)
    return examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--train_size", type=int, default=100)
    parser.add_argument("--val_size", type=int, default=20)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train = generate_examples(args.train_size, "train")
    val = generate_examples(args.val_size, "validation")

    pd.DataFrame(train).to_parquet(output_dir / "train.parquet")
    pd.DataFrame(val).to_parquet(output_dir / "validation.parquet")

    print(f"Generated {len(train)} train, {len(val)} val examples to {output_dir}")


if __name__ == "__main__":
    main()
