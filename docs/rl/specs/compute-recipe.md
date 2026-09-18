# Compute recipe: FLE deep RL on a Google Cloud VM

Written 2026-09-17 for the GFS Cloud Program credits, revised the same
day with measured numbers from `docs/rl/results/vm_throughput.md`.
Everything below is CPU-only.

The old framing said the Factorio server is the bottleneck and the
learner trains fine on CPU. The first half holds. The second is now the
constraint: the learner drains 16 steps/s and 128 servers produce 238.
See "The learner is the ceiling" below.

## Machine

Measured 2026-09-17, see `docs/rl/results/vm_throughput.md`. The numbers
below are from that sweep, not estimates.

| item | choice | why |
|---|---|---|
| type | `h4d-standard-192` (AMD Turin, 192 physical cores, no SMT) | cheapest per step of everything quota allows: 6.44 CHF per million steps |
| servers | **128** (sizing rule: 0.7 x vCPUs) | the measured knee; 238 steps/s |
| fallback | `n4d-highcpu-96` at 64 servers | 8.45 CHF/M steps; use if H4D has no capacity |
| disk | 200 GB **`hyperdisk-balanced`** | H4D rejects `pd-balanced` outright |
| image | Ubuntu 24.04 LTS x86_64 | Docker packages current |
| provisioning | on-demand | `PREEMPTIBLE_CPUS` quota is 0 in this project, so Spot is not an option until that is raised |
| zone | `us-central1-b` | `us-central1-a` had no H4D capacity |

h4d-standard-192 is 6.316 CHF/h, so 128 servers at 238 steps/s is 7.38
CHF per million steps and a full day is about 152 CHF. The credits are
about 24,863 USD, which is roughly 150 days of continuous running.

**Do not size one server per vCPU.** Each environment demands about 1.4
vCPU: the Docker template pins the server to 1 CPU, but the actor process
beside it wants another third. The knee landed at 0.7 x vCPUs on all
three machines measured (128/192, 64/96, 12/16).

**Quota, as found.** `CPUS_PER_VM_FAMILY` is dimensioned by region and
family and does not show up in `gcloud compute regions describe`. C3D,
C4D, C4 and C4A are capped at 24 vCPUs per region, which is why the
earlier `c3d-highcpu-60` plan in this file was unbuildable. H4D is 192,
N4D and N4 are 200, H3 is 88. General `CPUS` is 200 and binds across
families within a region.

```
gcloud compute instances create fle-rl-1 --project=aganthos-rag \
  --zone=us-central1-b --machine-type=h4d-standard-192 \
  --image-family=ubuntu-2404-lts-amd64 --image-project=ubuntu-os-cloud \
  --boot-disk-size=200GB --boot-disk-type=hyperdisk-balanced
gcloud compute ssh fle-rl-1 --zone=us-central1-b
```

## Setup on the VM (about 15 minutes)

```
sudo apt-get update && sudo apt-get install -y docker.io docker-compose-v2 git tmux htop
sudo usermod -aG docker $USER && newgrp docker
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.local/bin/env
git clone <repo> factorio-learning-environment && cd factorio-learning-environment
git checkout feat/pufferlib-rl
uv venv .venv --python 3.12 && uv pip install -e . --python .venv/bin/python
.venv/bin/pytest -q tests/rl          # 156 tests, offline, 5 s
```

Servers: `fle cluster start -n 128 -s open_world` generates a compose
file with `factoriotools/factorio:2.0.73`, `platform linux/amd64` (native
here, emulated on the Mac), 1 CPU and 1 GB per server, RCON on ports
27000 to 27127. First start pulls the image and generates 128 maps.

The 1-to-33 instance limit lives only in `fle/cluster/run-envs.sh`. The
Python `ClusterManager` that `fle cluster start` and the sweep both use
has no cap; 192 servers were started this way without trouble.

## Hour one: confirm, do not re-measure

The sweep already ran. Expect about **1.9 steps/s per server** at 128
servers and **238 steps/s aggregate**, with the game advancing about 311
ticks/s per server. Confirm with one short run and move on:

```
.venv/bin/python -m fle.rl.env --port 27000 --steps 200
```

Under 0.6 s per macro step is normal on this machine. If it is above 1 s,
check server UPS before anything else. Re-run the full sweep only if the
action regime or the observation size changes:

```
.venv/bin/python tests/benchmarks/vm_throughput_sweep.py \
  --servers 16,32,64,96,128,160,192 --seconds 300 --warmup 30 \
  --machine h4d-standard-192 --price-per-hour 6.316 --vcpus 192 \
  --out ~/sweep --ready-timeout 2400
```

**Shipping the code.** `fle/rl/` is untracked, so cloning the remote does
not give you the RL package. Build the tarball on the Mac with
`COPYFILE_DISABLE=1 tar -czf fle.tar.gz pyproject.toml uv.lock README.md
fle tests data`. Without that variable macOS writes AppleDouble resource
forks (`._*.lua`) into the archive, and `get_tools_to_load()` globs
`*.lua` and dies with `UnicodeDecodeError` on byte 0xa3.

**torch is not a base dependency.** The learner needs
`uv pip install torch --python .venv/bin/python --index-url
https://download.pytorch.org/whl/cpu`.

## The learner is the ceiling, not the environment

Measured on the H4D with the Rainbow flags below, against `--fake`:

| torch threads | updates/s |
|---:|---:|
| 1 | 12.45 |
| 2 | **16.15** |
| 4 | 15.58 |

Use two threads. Four is slower than two.

At `--update-every 1` a learner consumes one env step per update, so
16.15 / 1.86 is **8.7 servers per learner**. Six servers per learner, the
layout this file already used, is slightly conservative and holds.

This does **not** mean Rainbow cannot keep up. It means one learner
matches about nine servers, so the box is sized by how many runs you want
in parallel, not by one run's appetite.

Balanced unit: **1 learner + 9 servers = 14.6 cores** (9 x 1.4 for the
environments, 2 for the learner). On 192 cores that is:

| layout | learners | servers | cores | aggregate steps/s |
|---|---:|---:|---:|---:|
| fill the box | 13 | 117 | 190 | 211 |
| the planned experiment (2 regimes x 3 seeds) | 6 | 54 | 88 | 97 |

The planned experiment needs 88 cores, not 192. An `h4d-standard-192`
either runs it with 104 cores spare, or runs 13 seeds and ablations at
once instead of 6. Choose the machine by how many parallel runs you want.

Running 128 servers behind a single learner is the one layout that does
not work: the learner would drop 93 % of what the servers produce. If you
want one run to consume more than 16 steps/s, raise `--update-every`,
which changes the replay ratio and is a research decision, not a config
one. Not decided here.

## The runs

Two learners per regime, three seeds each, six servers per learner. On
128 servers that is 21 learners, which runs into the ceiling described
above; settle `--update-every` first and size the layout from the answer.
Rainbow flags are the ones in `docs/rl/results/overnight_v1.md`, section
"The program as of 16:23". Set `--torch-threads 2`, which the probe
measured as the best setting.

```
mkdir -p runs/rainbow-macro-s24
nohup .venv/bin/python -m fle.rl.dqn --algo dqn --rainbow --lr 1e-4 --per-alpha 0.6 \
  --per-optimism 3 --per-episode-return-bonus 2 --n-step 5 --replay-size 100000 \
  --batch-size 64 --update-every 1 --device cpu --log-actions --explore epsilon --eps-floor 0.02 \
  --archive-frac 0.25 --archive-mode uniform --action-regime macro \
  --ports 27000,27001,27002,27003,27004,27005 --total-steps 1000000 --seed 24 \
  --run-name rainbow-macro-s24 --out runs/rainbow-macro-s24 > runs/rainbow-macro-s24/train.log 2>&1 &
```

Repeat with `--action-regime bare` on the next six ports, then seeds 25
and 26 for whichever regime is ahead at 50k steps. Each learner writes
`env_<port>.jsonl`, `actions_<port>.jsonl`, `metrics.jsonl` and
checkpoints every 2,000 steps into its run directory; resume with
`--resume runs/<name>/checkpoint-<n>.pt`.

Monitoring from the repo: `tests/benchmarks/night_status.py`,
`tests/benchmarks/refresh_plots.sh` (edit the run list at the top), and
the dashboard it writes to `docs/rl/results/dashboard.html`. Copy it back
with `gcloud compute scp`.

## Things learned the hard way, so they do not repeat

- Prune checkpoints; a full disk killed three learners on the Mac.
- Which episode cap binds depends on the hardware. On the Mac episodes
  ended on the one-game-hour tick cap at a median of 115 decisions. On
  this machine they end on the 256-step cap every time, because a native
  step is about 4x faster while the game is not proportionally faster:
  166 ticks per step here against 480 to 1,100 on the Mac. An episode
  here therefore contains about 2.2x more decisions than the same config
  produced on the Mac, and results from before the move are not directly
  comparable. Keep reporting decisions per episode.
- Never edit the environment under a running comparison; restart from a
  checkpoint and note the version boundary in the log.
- Read traces, not only curves. Every wall so far was found in traces.
