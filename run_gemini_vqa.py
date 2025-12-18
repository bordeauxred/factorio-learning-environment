#!/usr/bin/env python3
"""Run VQA evaluation with Gemini"""

from inspect_ai import eval
from data.vqa.tasks import entity_name_task, position_task
from data.vqa.hook import VQAPairsHook

# Run evaluation with Gemini
results = eval(
    tasks=[
        entity_name_task(questions_per_blueprint=5),
        position_task(questions_per_blueprint=5),
    ],
    model=["google/gemini-2.5-flash-light"],  # Your model choice
    limit=2,  # Just 2 samples per task for testing
    log_dir="./logs",
    hooks=[VQAPairsHook()]
)

print("\n✓ Evaluation complete!")
print(f"Results: {results}")