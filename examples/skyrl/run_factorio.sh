#!/bin/bash
# Factorio RL Training with SkyrL
set -euo pipefail

DATA_DIR="${HOME}/data/factorio"

# Generate dataset if not exists
if [ ! -f "$DATA_DIR/train.parquet" ]; then
    echo "Generating dataset..."
    python examples/skyrl/generate_dataset.py --output_dir "$DATA_DIR"
fi

# Check Factorio cluster
if ! ss -lntp 2>/dev/null | grep -q ":27000"; then
    echo "WARNING: Factorio server not running on port 27000"
    echo "Start with: cd fle/cluster && ./run-envs.sh -n 1"
    # exit 1 # Optional: exit if not running
fi

# Run training
python -m skyrl_train.entrypoints.main_base \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  environment.env_class=factorio \
  trainer.policy.model.path="Qwen/Qwen2.5-1.5B-Instruct" \
  trainer.critic.model.path="Qwen/Qwen2.5-1.5B-Instruct" \
  trainer.placement.colocate_all=true \
  trainer.placement.policy_num_gpus_per_node=1 \
  generator.num_inference_engines=1 \
  generator.backend=vllm \
  generator.run_engines_locally=true \
  generator.async_engine=true \
  generator.batched=false \
  trainer.logger=console \
  trainer.epochs=5 \
  "$@"
