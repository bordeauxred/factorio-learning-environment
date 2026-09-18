# VM throughput and cost study

Measured 2026-09-17 on Google Cloud, project `aganthos-rag`, paid from the
GFS credits. Purpose: replace the guessed machine sizing in
`docs/rl/specs/compute-recipe.md` with measured numbers, before any
multi-day run. Harness: `tests/benchmarks/vm_throughput_sweep.py`, spec at
`docs/rl/specs/throughput-profile.md`.

All prices are CHF per hour, on-demand, from the Cloud Billing Catalog
API, which is the currency the billing account settles in. Spot was not
available: `PREEMPTIBLE_CPUS` is 0 in this project.

## Bottom line

Put the work on **h4d-standard-192**, and run **128 Factorio servers** on
it, not 24 and not 192.

- 128 servers is the knee. Marginal return per added vCPU holds near 1.9
  to 2.1 steps/s through 128, then falls to 0.81 at 160 and 0.27 at 192.
- If the box is already rented, 192 servers is still the cheapest per
  step (155,225 steps per CHF against 135,526 at 128), because the VM
  price does not change. Choose 128 when you care about latency per
  env, 192 when you only care about total steps per franc.
- Budget **1.4 vCPU per environment**, not 1. This is the single most
  useful number in the study and it reproduced on all three machines.
- SMT does carry a Factorio server, about a third of a core's worth.
  Do not size one server per physical core. H4D still wins on cost
  because its cores are cheap, not because hyperthreads are useless.

## What was measured

Each point starts N Factorio headless servers with the same code path
`fle cluster start` uses, waits for RCON on every port, then runs one
worker process per server driving a random valid macro policy for 300 s
after a 30 s warmup. Episodes use the production configuration (256
steps, 216,000 ticks). The per-step JSONL that training writes is written
here too, so disk cost is inside the measurement.

## h4d-standard-192, 192 physical cores, no SMT, 6.316 CHF/h

Complete.

| servers | steps/s | per server | ticks/s/srv | CPU % | vCPU/env | CHF/1k steps | steps/CHF | marginal steps/s per vCPU |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 33.74 | 2.109 | 349 | 10.3 | 1.24 | 0.0520 | 19,233 | |
| 32 | 64.55 | 2.017 | 330 | 21.3 | 1.28 | 0.0272 | 36,794 | 1.926 |
| 64 | 123.71 | 1.933 | 317 | 45.8 | 1.37 | 0.0142 | 70,514 | 1.849 |
| 96 | 170.32 | 1.774 | 294 | 74.7 | 1.49 | 0.0103 | 97,079 | 1.456 |
| **128** | **237.77** | **1.858** | **311** | **88.5** | **1.33** | **0.0074** | **135,526** | **2.108** |
| 160 | 263.64 | 1.648 | 272 | 93.1 | 1.12 | 0.0067 | 150,272 | 0.808 |
| 192 | 272.33 | 1.418 | 235 | 95.1 | 0.95 | 0.0064 | 155,225 | 0.272 |

272 steps/s is about 980,000 steps per hour, or 23.5 M steps per day from
one machine.

## n4d-highcpu-96, 96 threads on 48 physical cores, 2.758 CHF/h

Complete. This machine exists to answer whether SMT threads carry a
Factorio server. The crossing point is 48.

| servers | steps/s | per server | ticks/s/srv | CPU % | vCPU/env | steps/CHF | marginal steps/s per vCPU |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 17.23 | 2.153 | 349 | 11.0 | 1.32 | 22,486 | |
| 16 | 29.20 | 1.825 | 311 | 22.7 | 1.36 | 38,119 | 1.497 |
| 24 | 39.91 | 1.663 | 278 | 35.6 | 1.43 | 52,090 | 1.338 |
| 32 | 49.72 | 1.554 | 257 | 49.1 | 1.47 | 64,899 | 1.227 |
| 48 | 65.36 | 1.362 | 217 | 80.0 | 1.60 | 85,314 | 0.978 |
| **64** | **81.36** | **1.271** | **204** | **91.9** | **1.38** | **106,199** | **1.000** |
| 80 | 88.24 | 1.103 | 182 | 95.3 | 1.14 | 115,175 | 0.430 |
| 96 | 90.71 | 0.945 | 156 | 97.4 | 0.97 | 118,408 | 0.155 |

Knee at 64 servers, which is 16 past the 48 physical cores. Throughput
keeps climbing past 48 and only flattens near 96, so hyperthreads do
carry real work.

## c3d-highcpu-16, 16 threads on 8 physical cores, 0.482 CHF/h

Complete. Control for the family `compute-recipe.md` assumed.

| servers | steps/s | per server | ticks/s/srv | CPU % | vCPU/env | steps/CHF | marginal steps/s per vCPU |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 6.13 | 1.532 | 288 | 38.1 | 1.53 | 45,759 | |
| 8 | 10.96 | 1.370 | 238 | 82.5 | 1.65 | 81,859 | 1.208 |
| **12** | **13.74** | **1.145** | **197** | **93.8** | **1.25** | **102,598** | **0.694** |
| 16 | 14.66 | 0.916 | 165 | 96.6 | 0.97 | 109,519 | 0.232 |

Knee at 12 of 16 threads. Single-environment speed on this machine is
1.532 steps/s against 2.109 on the H4D and 2.153 on the N4D, so the
small C3D is the slowest per environment of the three even when it is
not saturated.

## Does SMT carry a Factorio server

Yes, by about a third. This contradicts the hypothesis the study was
built to test.

| machine | steps/s per physical core at its best | physical cores |
|---|---:|---:|
| n4d-highcpu-96 | 1.890 | 48 |
| c3d-highcpu-16 | 1.833 | 8 |
| h4d-standard-192 | 1.418 | 192 |

The two SMT machines extract about 33 % more throughput per physical
core than the no-SMT H4D. A Factorio server blocks on the RCON round
trip often enough that a second thread on the same core has real work to
do. Sizing one server per physical core leaves throughput unused.

H4D still wins on cost, for a different reason than expected: its
physical cores are 1.75x cheaper (0.0329 CHF per core-hour against
0.0575 for N4D). SMT gives N4D 1.33x more work per core, H4D cores cost
1.75x less, so H4D wins by 1.75 / 1.33 = 1.31x. That is exactly the
measured ratio of 155,225 to 118,408 steps per CHF.

## The knee is the same fraction on every machine

| machine | knee | vCPUs | knee / vCPUs |
|---|---:|---:|---:|
| h4d-standard-192 | 128 | 192 | 0.67 |
| n4d-highcpu-96 | 64 | 96 | 0.67 |
| c3d-highcpu-16 | 12 | 16 | 0.75 |

Two thirds of the vCPU count, which is the reciprocal of the 1.4 vCPU
per environment demand. Sizing rule: **servers = 0.7 x vCPUs**, whatever
the machine.

## Cost per million steps

| machine | servers | steps/s | CHF per 1 M steps |
|---|---:|---:|---:|
| h4d-standard-192 | 192 | 272.33 | **6.44** |
| h4d-standard-192 | 128 (knee) | 237.77 | 7.38 |
| n4d-highcpu-96 | 96 | 90.71 | 8.45 |
| c3d-highcpu-16 | 16 | 14.66 | 9.13 |
| n4d-highcpu-96 | 64 (knee) | 81.36 | 9.42 |

One H4D at 192 servers produces 23.5 M steps per day for about 152 CHF
per day. The 24,863 USD of credits buys roughly 150 days of that.

## Learner

Rainbow configuration from `compute-recipe.md`, against `--fake`, 2,000
updates, on the H4D:

| torch threads | updates/s | RSS GB |
|---:|---:|---:|
| 1 | 12.45 | 1.25 |
| 2 | 16.15 | 1.25 |
| 4 | 15.58 | 1.24 |

Two threads is the setting. Four is slower than two, so do not give a
learner more.

At `--update-every 1` a learner consumes one env step per update, so
16.15 updates/s divided by 1.86 steps/s per server is **8.7 servers per
learner**. The recipe's six servers per learner is close and slightly
conservative, so that assumption survives.

The useful way to read this: **one learner plus nine servers is 14.6
cores**, and that unit is what a machine holds. A 192-core box holds 13
of them, which is 117 servers and 211 steps/s, or it holds the planned
six-learner experiment in 88 cores with room to spare. Size the machine
by how many runs you want in parallel, not by a single run's appetite.

The layout that does not work is 128 servers behind one learner: it would
discard 93 % of what the servers produce. Raising `--update-every` is the
only way to make a single run consume more, and that changes the replay
ratio, so it is a research decision.

## Corrections to compute-recipe.md

Three of its assumptions are now measured. One is wrong, one has
stopped being true, and one holds.

**Wrong: one vCPU per server.** The Docker template pins each server to
1 CPU, but the actor process next to it wants another third of a core.
Measured demand is 1.3 to 1.65 vCPU per environment on every machine.
The recipe's "24 servers plus 4 learners on a 60-vCPU box" is sized
against the wrong unit.

**Changed, not wrong: which cap binds.** The recipe says episodes end on
the one-game-hour tick cap and that decisions per episode rise with
server speed. That was true when it was written. The Mac runs recorded in
`overnight_v1_log.md` end on the tick cap at a median of 115 decisions,
and the 512-step explorer episodes truncated at 67 to 310 steps.

On native hardware it flips. Measured decisions per episode is exactly
256 at every server count on all three machines: episodes end on the step
cap and the tick cap never binds.

The mechanism is ticks per step. Under box64 the Mac ran about 480 UPS
with 1 to 2.3 s per macro step, so each step advanced 480 to 1,100 ticks
and 256 steps needed 123k to 283k ticks against a 216k cap. On the H4D
the game advances 349 ticks/s while steps take 0.47 s, which is 166 ticks
per step, so 256 steps need only 42k ticks. Native hardware made the step
about 4x faster without making the game proportionally faster, so the
binding constraint moved.

This matters beyond sizing: **an episode on the VM is not the same
experiment as an episode on the Mac.** Same config, different amount of
game time per episode, roughly 2.2x more decisions. Results from before
the move are not directly comparable to results after it.

**Right: the native gain.** The recipe predicted under 0.5 s per macro
step against 1 to 2.3 s on the Mac under box64. Measured 0.47 s per step
at light load, so about 4x.

## CSCS Alps was evaluated and ruled out

Probed 2026-09-17 on `clariden.alps.cscs.ch`, account `abs12`.

**Factorio cannot run on Clariden.** The GH200 nodes run a 64 KB page
kernel (`6.4.0-150600.23.125-64kb`, `getconf PAGESIZE` = 65536) on both
login and compute nodes. x86-64 binaries assume 4 KB pages, and box64
refuses to start:

```
Error: PageSize configuration is wrong: configured with 4096, but got 65536
```

This is architectural, not configuration. There is no flag that fixes it
and no 4 KB-page partition on this vCluster.

Everything else had already been made to work before hitting this, so it
is worth recording in case an x86 vCluster becomes available:

- The `factoriotools/factorio:2.0.73` image has an arm64 variant carrying
  `/bin/box64` next to the x86-64 Factorio ELF, and `run_envs.py:96`
  already selects `/bin/box64` on aarch64.
- Podman fails on both the NFS home and `$SCRATCH` with
  `lsetxattr: operation not supported`. Use
  `--storage-driver vfs` with the store on `/dev/shm` (334 GB tmpfs).
- There is **no compose on Alps**: no `docker compose` plugin, no
  `podman-compose`, no `docker-compose`. `/usr/bin/docker` is a podman
  shim. `ClusterManager` shells out to `docker compose`, so real use
  needs a `podman run` path per container.
- `/opt/factorio/scenarios` and `/opt/factorio/saves` are symlinks into
  `/factorio`, which is a declared VOLUME. Binding to the
  `/opt/factorio/*` paths works under Docker but crun rejects it with
  `creating /opt/factorio/scenarios: No such file or directory`. Bind to
  `/factorio/scenarios` and `/factorio/mods` instead.

**No x86 compute is reachable from this account.** Every Clariden
partition (`normal`, `debug`, `low`, `highprio`, `preemptable`) is the
same 288-core `gh` node type; the only 128-CPU nodes are two `xfer` data
transfer nodes. `eiger`, `daint`, `santis` and `bristen` all refuse the
key. The 1,024-node AMD Rome partition in the CSCS node table belongs to
a vCluster this account cannot reach.

Conclusion: the GCP H4D plan stands. Revisit only if an x86 Alps
allocation appears.

## Caveats

`--speed 40` asks the server for 40x real time, which is 2,400 ticks/s.
Measured game speed is 349 ticks/s at 16 servers on an idle 192-core
machine. The game is not reaching requested speed even when nothing is
competing for CPU, which points at the RCON round trip rather than
simulation as the binding constraint at low server counts. This is worth
its own investigation and it is not answered here.

The policy is random, not trained. A trained policy issues different
operations with different costs, so absolute steps/s will move. The
shape of the curves and the vCPU-per-env number should hold.

## Reproduce

```
gcloud compute instances create fle-sweep --project=aganthos-rag \
  --zone=us-central1-b --machine-type=h4d-standard-192 \
  --image-family=ubuntu-2404-lts-amd64 --image-project=ubuntu-os-cloud \
  --boot-disk-size=200GB --boot-disk-type=hyperdisk-balanced
# H4D needs hyperdisk; pd-balanced is rejected. us-central1-a had no H4D
# capacity, us-central1-b did.

COPYFILE_DISABLE=1 tar -czf fle.tar.gz pyproject.toml uv.lock README.md fle tests data
# Without COPYFILE_DISABLE the macOS resource forks (._*.lua) ship too and
# FLE's get_tools_to_load() tries to parse them as Lua. It fails with
# UnicodeDecodeError on byte 0xa3.

.venv/bin/python tests/benchmarks/vm_throughput_sweep.py \
  --servers 16,32,64,96,128,160,192 --seconds 300 --warmup 30 \
  --machine h4d-standard-192 --price-per-hour 6.316 --vcpus 192 \
  --out ~/sweep --ready-timeout 2400
# torch is not a base dependency; the learner probe needs
# uv pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Quota, as found

Measured, not assumed. `CPUS_PER_VM_FAMILY` is dimensioned by region and
family and does not appear in `gcloud compute regions describe`.

| family | limit per region | note |
|---|---:|---|
| C3D, C4D, C4, C4A | 24 | too small for this work |
| H4D | 192 | no SMT, cheapest physical core here |
| N4D, N4, N4A, E4, E4A | 200 | SMT |
| H3 | 88 | no SMT, Intel |
| C4N | 384 | 0.0547 CHF/vCPU-h, expensive |
| general `CPUS` | 200 | binds across families in a region |
| `PREEMPTIBLE_CPUS` | 0 | no Spot at all |

Spot C3D is 4.3x cheaper than on-demand, so raising `PREEMPTIBLE_CPUS` is
worth real money before any multi-day run.
