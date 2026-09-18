#!/bin/zsh
# Regenerate plots with descriptive arm names and rebuild the dashboard. Used by the scheduled checks.
cd /Users/robertmueller/Desktop/agents/factorio-learning-environment
.venv/bin/python tests/benchmarks/plot_first_curves.py --out docs/rl/results/overnight_v1 --window 10 --block 10 \
  "rainbow-macro=runs/rainbow-optimisticper-n5-macro/env_27003.jsonl,runs/rainbow-optimisticper-n5-macro/env_27004.jsonl,runs/rainbow-optimisticper-n5-macro/env_27005.jsonl" \
  $( [ -f runs/rainbow-optimisticper-n5-bare/env_27000.jsonl ] && echo "rainbow-bare=runs/rainbow-optimisticper-n5-bare/env_27000.jsonl,runs/rainbow-optimisticper-n5-bare/env_27001.jsonl,runs/rainbow-optimisticper-n5-bare/env_27002.jsonl" ) \
  2>&1 | grep -v Pydantic | grep "| all\|drill" | cut -c1-110
.venv/bin/python tests/benchmarks/build_dashboard.py --out docs/rl/results/dashboard.html --plots docs/rl/results/overnight_v1 2>&1 | grep -v Pydantic | tail -1
