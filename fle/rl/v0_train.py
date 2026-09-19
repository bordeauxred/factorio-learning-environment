"""Offline-capable V0 masked autoregressive Double-DQN trainer."""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from fle.rl.v0_nets import V0DQN, behaviour_actions, double_dqn_loss
from fle.rl.v0_replay import (
    V0NStepAccumulator,
    V0ReplayBuffer,
    full_masks,
)
from fle.rl.v0_schema import (
    BUILD_CHANNELS,
    BUILD_SIDE,
    DIRECTIONS,
    HEADS,
    OBS_SPEC,
    PLACE_MAX_AGE_TICKS,
    PLACEABLE_NAMES,
    SMDP_HALF_LIFE_SECONDS,
    VERB_HEADS,
    VERB_INDEX,
    VERBS,
    cell_to_world,
)

WANDB_ENTITY = "2robert-mueller-none"
WANDB_PROJECT = "fle-sm-arq"


def select_device(requested: str = "auto") -> torch.device:
    """Prefer CUDA, then a successfully probed MPS device, then CPU."""
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        try:
            probe = torch.ones(1, device="mps")
            if float((probe + 1).cpu()[0]) == 2.0:
                return torch.device("mps")
        except (RuntimeError, OSError):
            pass
    return torch.device("cpu")


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _git_value(*args: str) -> str:
    try:
        return subprocess.check_output(
            ("git", *args), text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _vocab_hash(values: Sequence[str]) -> str:
    return hashlib.sha256("\0".join(values).encode()).hexdigest()


def run_config(args: argparse.Namespace, network: V0DQN, replay_size: int) -> dict:
    from fle.rl.v0_schema import ITEM_NAMES, PLACEABLE_NAMES, RECIPE_NAMES, TECH_NAMES

    return {
        "git_sha": _git_value("rev-parse", "HEAD"),
        "git_branch": _git_value("branch", "--show-current"),
        "factorio_version": os.environ.get(
            "FACTORIO_VERSION", "offline-fake" if args.fake else "unknown"
        ),
        "seed": args.seed,
        "observation_shapes": {
            key: list(shape) for key, (shape, _) in OBS_SPEC.items()
        },
        "observation_dtypes": {key: str(dtype) for key, (_, dtype) in OBS_SPEC.items()},
        "vocab_hashes": {
            "items": _vocab_hash(ITEM_NAMES),
            "placeables": _vocab_hash(PLACEABLE_NAMES),
            "recipes": _vocab_hash(RECIPE_NAMES),
            "technologies": _vocab_hash(TECH_NAMES),
        },
        "network_config": {
            "state_width": network.encoder.output_size,
            "prefix_width": network.prefix_size,
            "parameter_counts": network.parameter_counts(),
        },
        "replay_config": {
            "capacity": replay_size,
            "prioritized": False,
            "n_step": args.n_step,
        },
        "epsilon_schedule": {
            "start": args.eps_start,
            "end": args.eps_end,
            "decay_steps": args.eps_decay_steps,
        },
        "mask_toggles": {
            "verb": True,
            "place_location": True,
            "place_item": True,
            "inventory_item": True,
            "contained_item": True,
            "entity": True,
            "recipe": True,
            "technology": True,
        },
        "buildability_size": [BUILD_CHANNELS, BUILD_SIDE, BUILD_SIDE],
        "buildability_budget": 63,
        "freshness_thresholds": {"place_max_age_ticks": PLACE_MAX_AGE_TICKS},
        "minimap_chunk_budget": 0 if args.fake else "environment-configured",
        "reward_definition": "automated_production_score_delta",
        "smdp_half_life_seconds": SMDP_HALF_LIFE_SECONDS,
        "discount_mode": args.discount,
        "cli": dict(vars(args)),
    }


class SafeWandb:
    """Best-effort W&B logging; every integration failure is non-fatal."""

    def __init__(
        self,
        mode: str,
        config: Mapping[str, Any],
        demo: bool,
        run_name: str | None = None,
    ) -> None:
        self.run = None
        self._wandb = None
        if mode == "disabled":
            return
        try:
            import wandb

            timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
            kind = "demo" if demo else "scratch"
            name = run_name or f"v0-mask-{kind}-{timestamp}"
            Path("/tmp/fle-v0-wandb").mkdir(parents=True, exist_ok=True)
            self.run = wandb.init(
                entity=WANDB_ENTITY,
                project=WANDB_PROJECT,
                name=name,
                config=dict(config),
                mode=mode,
                dir="/tmp/fle-v0-wandb",
                settings=wandb.Settings(silent=True),
            )
            self._wandb = wandb
        except Exception as error:  # noqa: BLE001 - logging cannot kill training
            print(f"wandb_disabled error={error!r}", file=sys.stderr, flush=True)
            self.run = None

    def histogram(self, values: Sequence[int]) -> Any:
        if self._wandb is None:
            return list(values)
        try:
            return self._wandb.Histogram(list(values))
        except Exception as error:  # noqa: BLE001 - logging cannot kill training
            print(f"wandb_histogram_failed error={error!r}", file=sys.stderr)
            return list(values)

    def log(self, metrics: Mapping[str, Any], step: int) -> None:
        if self.run is None:
            return
        try:
            self.run.log(dict(metrics), step=step)
        except Exception as error:  # noqa: BLE001 - logging cannot kill training
            print(f"wandb_log_failed error={error!r}", file=sys.stderr, flush=True)
            self.run = None

    def finish(self) -> None:
        if self.run is None:
            return
        try:
            self.run.finish()
        except Exception as error:  # noqa: BLE001 - logging cannot kill training
            print(f"wandb_finish_failed error={error!r}", file=sys.stderr, flush=True)


def synthetic_observation(
    legacy_obs: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Build a schema-exact fake state without importing concurrent V0 modules."""
    obs = {
        key: np.zeros(shape, dtype=dtype) for key, (shape, dtype) in OBS_SPEC.items()
    }
    obs["buildability"].fill(-1)
    obs["build_age"].fill(np.iinfo(np.uint16).max)
    obs["entity_mask"][0] = 1.0
    obs["local_exact"][12, 32, 32] = 1.0
    if legacy_obs is not None:
        count = min(3, len(legacy_obs), len(obs["globals"]))
        obs["globals"][:count] = legacy_obs[:count]
    return obs


def fake_masks() -> dict[str, np.ndarray]:
    """A maximally constrained valid grammar keeps offline smoke runs quick."""
    masks = full_masks()
    masks["verb"].fill(0)
    masks["verb"][VERB_INDEX["FAST_FORWARD"]] = 1
    for head in masks:
        if head != "verb":
            masks[head].fill(0)
            masks[head][0] = 1
    return masks


class ObservationBatcher:
    """Assemble acting batches with reusable float conversion buffers."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.buildability = np.empty(
            (1, *OBS_SPEC["buildability"][0]), dtype=np.float32
        )
        self.build_age = np.empty((1, *OBS_SPEC["build_age"][0]), dtype=np.float32)

    def batch(self, obs: Mapping[str, np.ndarray]) -> dict[str, torch.Tensor]:
        result = {}
        for key in OBS_SPEC:
            value = obs[key]
            if key == "buildability":
                np.copyto(self.buildability[0], value, casting="unsafe")
                array = self.buildability
            elif key == "build_age":
                np.copyto(self.build_age[0], value, casting="unsafe")
                array = self.build_age
            else:
                array = value[None]
            result[key] = torch.as_tensor(array, device=self.device)
        return result


def _tensor_masks(
    masks: Mapping[str, np.ndarray], device: torch.device
) -> dict[str, torch.Tensor]:
    return {
        key: torch.as_tensor(value, dtype=torch.bool, device=device).unsqueeze(0)
        for key, value in masks.items()
    }


def epsilon_at(step: int, args: argparse.Namespace) -> float:
    fraction = min(1.0, step / max(1, args.eps_decay_steps))
    return args.eps_start + fraction * (args.eps_end - args.eps_start)


def _observation_payload(obs: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Drop environment-only metadata while preserving schema array identity."""
    return {key: obs[key] for key in OBS_SPEC}


def _live_mask_source(env: Any, obs: Mapping[str, Any]) -> Any:
    """Build a batched resolver over the environment's immutable decision frame."""
    from fle.rl.v0_actions import v0_masks_for_prefix

    world, context = env.v0_learner_context()
    base_masks = obs["masks"]

    def resolve(head: str, prefix: Mapping[str, torch.Tensor]) -> np.ndarray:
        base = np.asarray(base_masks[head], dtype=np.uint8)
        if not prefix:
            return base
        host_prefix = {
            key: value.detach().cpu().numpy().reshape(-1)
            for key, value in prefix.items()
        }
        rows = len(next(iter(host_prefix.values())))
        resolved = np.empty((rows, base.size), dtype=np.uint8)
        for row in range(rows):
            verb_index = int(host_prefix["verb"][row])
            verb = VERBS[verb_index]
            scalar_prefix = {"verb": verb_index}
            for earlier_head in VERB_HEADS[verb]:
                if earlier_head == head:
                    break
                scalar_prefix[earlier_head] = int(host_prefix[earlier_head][row])
            conditional = v0_masks_for_prefix(scalar_prefix, obs, world, context)[head]
            resolved[row] = np.asarray(conditional, dtype=np.uint8) & base
        return resolved

    return resolve


def _selected_place_state(env: Any, action: np.ndarray) -> str:
    """Classify the selected PLACE cell from the same cache used by masking."""
    if int(action[HEADS.index("verb")]) != VERB_INDEX["PLACE"]:
        return "none"
    _, context = env.v0_learner_context()
    cache = context["buildability"]
    item = PLACEABLE_NAMES[int(action[HEADS.index("place_item")])]
    prototype = context["vocab"].place_results.get(item, item)
    direction = DIRECTIONS[int(action[HEADS.index("direction")])]
    try:
        channel = cache.channel_for(prototype, direction)
    except KeyError:
        return "unknown"
    player_x, player_y = context["position"]
    world_x, world_y = cell_to_world(
        int(action[HEADS.index("location")]), player_x, player_y
    )
    cache_x = world_x - cache.origin[0]
    cache_y = world_y - cache.origin[1]
    if not (
        0 <= cache_y < cache.values.shape[1] and 0 <= cache_x < cache.values.shape[2]
    ):
        return "unknown"
    value = int(cache.values[channel, cache_y, cache_x])
    if value < 0 or not cache.sampled_ticks.size:
        return "unknown"
    sampled_tick = int(cache.sampled_ticks[channel, cache_y // 8, cache_x // 8])
    if sampled_tick < 0:
        return "unknown"
    age = int(context["tick"]) - sampled_tick
    if age < 0 or age > int(context["place_max_age_ticks"]):
        return "stale"
    return "known_legal" if value == 1 else "known_blocked"


@dataclass(frozen=True)
class LiveDecisionResult:
    """Outcome of one failure-contained live decision."""

    obs: Mapping[str, Any]
    action: np.ndarray | None
    reward: float
    info: Mapping[str, Any]
    done: bool
    duration: float
    transitions_added: int
    action_seconds: float
    environment_seconds: float
    place_masked_fraction: float
    place_state: str
    q_value: float
    error: Exception | None = None


def live_decision_step(
    env: Any,
    obs: Mapping[str, Any],
    online: V0DQN,
    replay: V0ReplayBuffer,
    accumulator: V0NStepAccumulator,
    batcher: ObservationBatcher,
    device: torch.device,
    epsilon: float,
) -> LiveDecisionResult:
    """Decode, execute, and retain one live transition without leaking errors."""
    action_started = time.perf_counter()
    try:
        mask_source = _live_mask_source(env, obs)
        tensor_obs = batcher.batch(obs)
        with torch.no_grad():
            decoded = behaviour_actions(online, tensor_obs, mask_source, epsilon)
        _synchronize(device)
        action_seconds = time.perf_counter() - action_started
        action = decoded.actions[0].detach().cpu().numpy().astype(np.int64, copy=False)

        place_masked_fraction = 0.0
        if int(action[HEADS.index("verb")]) == VERB_INDEX["PLACE"]:
            prefix = {
                head: torch.as_tensor([action[index]], device=device)
                for index, head in enumerate(HEADS)
            }
            location_mask = np.asarray(mask_source("location", prefix)).reshape(-1)
            place_masked_fraction = float(1.0 - location_mask.mean())
        place_state = _selected_place_state(env, action)

        environment_started = time.perf_counter()
        next_obs, reward, terminated, truncated, info = env.step(action)
        environment_seconds = time.perf_counter() - environment_started
        done = bool(terminated or truncated)
        duration = float(info.get("ticks", 0.0)) / 60.0
        emitted = accumulator.append(
            _observation_payload(obs),
            action,
            float(reward),
            _observation_payload(next_obs),
            duration,
            done,
            obs["masks"],
            next_obs["masks"],
        )
        for transition in emitted:
            replay.add(transition)
        return LiveDecisionResult(
            obs=next_obs,
            action=action,
            reward=float(reward),
            info=info,
            done=done,
            duration=duration,
            transitions_added=len(emitted),
            action_seconds=action_seconds,
            environment_seconds=environment_seconds,
            place_masked_fraction=place_masked_fraction,
            place_state=place_state,
            q_value=float(decoded.q_values[0].detach().cpu()),
        )
    except Exception as error:  # noqa: BLE001 - an unattended run must continue
        return LiveDecisionResult(
            obs=obs,
            action=None,
            reward=0.0,
            info={},
            done=False,
            duration=0.0,
            transitions_added=0,
            action_seconds=time.perf_counter() - action_started,
            environment_seconds=0.0,
            place_masked_fraction=0.0,
            place_state="none",
            q_value=0.0,
            error=error,
        )


def train_fake(args: argparse.Namespace) -> dict[str, float | str | int]:
    # Import lazily: the V0 learner tests do not depend on any environment work.
    from fle.rl import schema as legacy_schema
    from fle.rl.fake_env import FakeMacroEnv

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.torch_threads)
    rng = np.random.default_rng(args.seed)
    device = select_device(args.device)
    print(f"device={device}", flush=True)

    online = V0DQN().to(device)
    target = V0DQN().to(device)
    target.load_state_dict(online.state_dict())
    target.eval()
    total_parameters = sum(parameter.numel() for parameter in online.parameters())
    if total_parameters >= 10_000_000:
        raise RuntimeError(f"V0 network has {total_parameters:,} parameters")
    counts = online.parameter_counts()
    print(
        "parameters "
        + " ".join(f"{name}={count}" for name, count in counts.items())
        + f" total={total_parameters}",
        flush=True,
    )

    # Fake runs deliberately cap resident replay while retaining the contract's
    # 8192 default for real runs.
    replay_size = min(args.replay_size, args.fake_replay_size)
    replay = V0ReplayBuffer(replay_size)
    optimizer_options: dict[str, Any] = {}
    if device.type == "cuda":
        optimizer_options["fused"] = True
    else:
        # Measured on MPS: fused and foreach are within noise of each other;
        # foreach is the safer default on this backend.
        optimizer_options["foreach"] = True
    optimizer = torch.optim.AdamW(
        online.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        **optimizer_options,
    )
    accumulator = V0NStepAccumulator(
        args.n_step, use_smdp=args.discount == "smdp", gamma=args.gamma
    )
    logger = SafeWandb(
        args.wandb_mode, run_config(args, online, replay_size), demo=True
    )
    print(
        f"replay capacity={replay.capacity} "
        f"bytes_per_transition={replay.bytes_per_transition} nbytes={replay.nbytes}",
        flush=True,
    )

    env = FakeMacroEnv(horizon=args.fake_horizon, seed=args.seed)
    legacy_obs, _ = env.reset(seed=args.seed)
    obs = synthetic_observation(legacy_obs)
    masks = fake_masks()
    acting_batcher = ObservationBatcher(device)
    episode_return = 0.0
    updates = 0
    profiled_updates = 0
    last_loss: float | None = None
    last_loss_tensor: torch.Tensor | None = None
    grad_norm_tensor: torch.Tensor | None = None
    action_forward_seconds = 0.0
    update_forward_seconds = 0.0
    backward_seconds = 0.0
    pending_timing_events: tuple[Any, Any, Any, Any] | None = None
    start = time.perf_counter()

    try:
        for step in range(1, args.total_steps + 1):
            epsilon = epsilon_at(step, args)
            tensor_obs = acting_batcher.batch(obs)
            tensor_masks = _tensor_masks(masks, device)
            before = time.perf_counter()
            with torch.no_grad():
                decoded = behaviour_actions(online, tensor_obs, tensor_masks, epsilon)
            _synchronize(device)
            action_forward_seconds += time.perf_counter() - before
            action = decoded.actions[0].cpu().numpy()

            # The legacy fake environment supplies deterministic server-free
            # dynamics. Its grammar differs, so its own valid action drives the
            # dynamics while the V0 action drives learner/replay coverage.
            legacy_action = legacy_schema.random_valid_action(legacy_obs, rng)
            legacy_next, reward, terminated, truncated, info = env.step(legacy_action)
            done = terminated or truncated
            next_obs = synthetic_observation(legacy_next)
            duration = 1.0
            episode_return += reward
            emitted = accumulator.append(
                obs,
                action,
                reward,
                next_obs,
                duration,
                done,
                masks,
                masks,
            )
            for transition in emitted:
                replay.add(transition)

            if replay.size >= args.learning_starts and step % args.update_every == 0:
                batch = replay.sample(args.batch_size, rng, device)
                optimizer.zero_grad(set_to_none=True)
                profile_update = step % args.log_interval == 0
                if profile_update:
                    if device.type == "mps":
                        events = tuple(
                            torch.mps.Event(enable_timing=True) for _ in range(4)
                        )
                    elif device.type == "cuda":
                        events = tuple(
                            torch.cuda.Event(enable_timing=True) for _ in range(4)
                        )
                    else:
                        events = None
                    if events is not None:
                        events[0].record()
                update_forward_start = (
                    time.perf_counter()
                    if profile_update and device.type == "cpu"
                    else 0.0
                )
                loss, td_error = double_dqn_loss(online, target, batch)
                if profile_update:
                    if events is not None:
                        events[1].record()
                        events[2].record()
                    else:
                        update_forward_seconds += (
                            time.perf_counter() - update_forward_start
                        )
                backward_start = (
                    time.perf_counter()
                    if profile_update and device.type == "cpu"
                    else 0.0
                )
                loss.backward()
                grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
                    online.parameters(),
                    args.max_grad_norm,
                    foreach=device.type != "mps",
                )
                optimizer.step()
                if profile_update:
                    if events is not None:
                        events[3].record()
                        pending_timing_events = events
                    else:
                        backward_seconds += time.perf_counter() - backward_start
                    profiled_updates += 1
                last_loss_tensor = loss.detach()
                updates += 1
                if updates % args.target_period == 0:
                    target.load_state_dict(online.state_dict())

            if step == 1 or step % args.log_interval == 0:
                _synchronize(device)
                if pending_timing_events is not None:
                    forward_start, forward_end, backward_start, backward_end = (
                        pending_timing_events
                    )
                    update_forward_seconds += (
                        forward_start.elapsed_time(forward_end) / 1000.0
                    )
                    backward_seconds += (
                        backward_start.elapsed_time(backward_end) / 1000.0
                    )
                    pending_timing_events = None
                if last_loss_tensor is not None:
                    last_loss = float(last_loss_tensor.cpu())
                    if not np.isfinite(last_loss):
                        raise FloatingPointError(f"non-finite loss at step {step}")
                elapsed = max(time.perf_counter() - start, 1e-9)
                timing_count = max(profiled_updates, 1)
                forward_ms = 1000.0 * update_forward_seconds / timing_count
                action_forward_ms = 1000.0 * action_forward_seconds / step
                backward_ms = 1000.0 * backward_seconds / timing_count
                metrics = {
                    "env/step": step,
                    "env/episode_return": episode_return,
                    "env/simulated_seconds": step,
                    "env/automated_score": float(info.get("score_automated", 0.0)),
                    "env/general_score": float(info.get("score_player", 0.0)),
                    "env/invalid_rate": 0.0,
                    "action/verb": int(action[HEADS.index("verb")]),
                    "action/invalid_rate": 0.0,
                    "build/known_fraction": 0.0,
                    "build/fresh_fraction": 0.0,
                    "build/place_masked_fraction": 0.0,
                    "build/place_selected_known_legal": 0,
                    "build/place_selected_known_blocked": 0,
                    "build/place_selected_unknown": 0,
                    "build/place_selected_stale": 0,
                    "build/mask_lookup_miss": 0,
                    "train/replay_size": replay.size,
                    "train/replay_bytes_per_transition": replay.bytes_per_transition,
                    "train/replay_bytes": replay.nbytes,
                    "explore/epsilon": epsilon,
                    "perf/network_forward_ms": forward_ms,
                    "perf/network_backward_ms": backward_ms,
                    "perf/action_forward_ms": action_forward_ms,
                    "perf/updates_per_second": updates / elapsed,
                    "perf/env_steps_per_second": step / elapsed,
                }
                if updates and last_loss is not None:
                    assert grad_norm_tensor is not None
                    metrics.update(
                        {
                            "train/loss": last_loss,
                            "train/td_error": float(td_error.detach().cpu()),
                            "train/grad_norm": float(grad_norm_tensor.cpu()),
                        }
                    )
                logger.log(metrics, step)
                loss_text = f" loss={last_loss:.6g}" if last_loss is not None else ""
                print(
                    f"step={step}{loss_text} replay={replay.size} "
                    f"epsilon={epsilon:.4f} forward_ms={forward_ms:.3f} "
                    f"backward_ms={backward_ms:.3f} updates_per_second="
                    f"{updates / elapsed:.3f}",
                    flush=True,
                )

            if done:
                legacy_obs, _ = env.reset(seed=args.seed + step)
                obs = synthetic_observation(legacy_obs)
                episode_return = 0.0
            else:
                legacy_obs = legacy_next
                obs = next_obs
    finally:
        env.close()
        logger.finish()

    if updates == 0:
        raise RuntimeError("training ended before the first update")
    assert last_loss_tensor is not None
    _synchronize(device)
    last_loss = float(last_loss_tensor.cpu())
    elapsed = max(time.perf_counter() - start, 1e-9)
    result: dict[str, float | str | int] = {
        "device": str(device),
        "final_loss": last_loss,
        "forward_ms": 1000.0 * update_forward_seconds / max(profiled_updates, 1),
        "backward_ms": 1000.0 * backward_seconds / max(profiled_updates, 1),
        "updates_per_second": updates / elapsed,
        "bytes_per_transition": replay.bytes_per_transition,
        "updates": updates,
    }
    print(
        "final " + " ".join(f"{key}={value}" for key, value in result.items()),
        flush=True,
    )
    return result


def _live_environment(args: argparse.Namespace) -> Any:
    from fle.rl.env import FleMacroEnv

    return FleMacroEnv(
        port=args.port,
        speed=args.speed,
        max_steps=args.episode_max_steps,
        max_ticks=10**9,
        seed=args.seed,
        action_grammar="v0",
        v0_batch_snapshot=True,
    )


def _connection_failure(error: Exception) -> bool:
    if isinstance(error, (ConnectionError, TimeoutError, OSError)):
        return True
    message = str(error).lower()
    return any(
        marker in message
        for marker in (
            "broken pipe",
            "connection",
            "not connected",
            "rcon",
            "socket",
            "timed out",
        )
    )


def _close_environment(env: Any) -> None:
    try:
        env.close()
    except Exception as error:  # noqa: BLE001 - shutdown is best effort
        print(f"environment_close_failed error={error!r}", file=sys.stderr, flush=True)


def _start_live_environment(args: argparse.Namespace, seed: int) -> tuple[Any, Any]:
    """Construct/reset a live environment, retrying only connection failures."""
    while True:
        env = None
        try:
            env = _live_environment(args)
            return env, env.reset(seed=seed)
        except Exception as error:
            if env is not None:
                _close_environment(env)
            if not _connection_failure(error):
                raise
            print(
                f"environment_connection_retry error={error!r}",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(5.0)


def _flush_accumulator(accumulator: V0NStepAccumulator, replay: V0ReplayBuffer) -> None:
    for transition in accumulator.flush():
        replay.add(transition)


def _build_fractions(env: Any) -> tuple[float, float, float]:
    world, context = env.v0_learner_context()
    cache = world.buildability
    known = cache.values >= 0
    if not known.size:
        return 0.0, 1.0, 0.0
    if cache.sampled_ticks.size:
        age = int(context["tick"]) - cache.sampled_ticks
        fresh_blocks = (
            (cache.sampled_ticks >= 0)
            & (age >= 0)
            & (age <= int(context["place_max_age_ticks"]))
        )
        fresh = fresh_blocks.repeat(8, axis=1).repeat(8, axis=2) & known
    else:
        fresh = np.zeros_like(known)
    known_fraction = float(known.mean())
    return known_fraction, 1.0 - known_fraction, float(fresh.mean())


def _save_checkpoint(
    run_name: str,
    step: int,
    online: V0DQN,
    target: V0DQN,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
) -> Path:
    directory = Path("checkpoints") / run_name
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"step-{step:09d}.pt"
    temporary = destination.with_suffix(".tmp")
    torch.save(
        {
            "step": step,
            "online": online.state_dict(),
            "target": target.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": dict(vars(args)),
            "git_sha": _git_value("rev-parse", "HEAD"),
            "git_branch": _git_value("branch", "--show-current"),
        },
        temporary,
    )
    temporary.replace(destination)
    return destination


def train_live(args: argparse.Namespace) -> dict[str, float | str | int]:
    """Run the unattended learner against one reusable live FLE environment."""
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.torch_threads)
    rng = np.random.default_rng(args.seed)
    device = select_device(args.device)
    print(f"device={device}", flush=True)

    if args.run_name is None:
        timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        args.run_name = f"v0-mask-scratch-{timestamp}"
    online = V0DQN().to(device)
    target = V0DQN().to(device)
    target.load_state_dict(online.state_dict())
    target.eval()
    total_parameters = sum(parameter.numel() for parameter in online.parameters())
    if total_parameters >= 10_000_000:
        raise RuntimeError(f"V0 network has {total_parameters:,} parameters")
    print(
        "parameters "
        + " ".join(
            f"{name}={count}" for name, count in online.parameter_counts().items()
        )
        + f" total={total_parameters}",
        flush=True,
    )

    replay = V0ReplayBuffer(args.replay_size)
    optimizer_options: dict[str, Any] = {}
    if device.type == "cuda":
        optimizer_options["fused"] = True
    else:
        optimizer_options["foreach"] = True
    optimizer = torch.optim.AdamW(
        online.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        **optimizer_options,
    )
    accumulator = V0NStepAccumulator(
        args.n_step, use_smdp=args.discount == "smdp", gamma=args.gamma
    )
    batcher = ObservationBatcher(device)
    logger = SafeWandb(
        args.wandb_mode,
        run_config(args, online, args.replay_size),
        demo=False,
        run_name=args.run_name,
    )
    print(
        f"replay capacity={replay.capacity} "
        f"bytes_per_transition={replay.bytes_per_transition} nbytes={replay.nbytes}",
        flush=True,
    )

    env, (obs, _) = _start_live_environment(args, args.seed)
    start = time.perf_counter()
    decisions = 0
    episodes = 1
    updates = 0
    update_budget = 0.0
    simulated_seconds = 0.0
    accumulated_reward = 0.0
    successes = 0
    place_attempts = 0
    place_successes = 0
    place_masked_sum = 0.0
    random_action_expectation = 0.0
    verb_samples: list[int] = []
    place_states: Counter[str] = Counter()
    last_loss_tensor: torch.Tensor | None = None
    last_td_tensor: torch.Tensor | None = None
    last_grad_tensor: torch.Tensor | None = None
    last_q = 0.0
    action_seconds = 0.0
    environment_seconds = 0.0
    update_seconds = 0.0
    update_forward_seconds = 0.0
    update_backward_seconds = 0.0
    replay_sample_seconds = 0.0
    profiled_updates = 0

    try:
        while decisions < args.total_steps:
            epsilon = epsilon_at(decisions + 1, args)
            result = live_decision_step(
                env,
                obs,
                online,
                replay,
                accumulator,
                batcher,
                device,
                epsilon,
            )
            if result.error is not None:
                print(
                    f"live_step_failed step={decisions + 1} error={result.error!r}",
                    file=sys.stderr,
                    flush=True,
                )
                logger.log({"env/step_exception": 1}, max(decisions, 1))
                if _connection_failure(result.error):
                    _flush_accumulator(accumulator, replay)
                    _close_environment(env)
                    episodes += 1
                    env, (obs, _) = _start_live_environment(args, args.seed + episodes)
                continue

            decisions += 1
            obs = result.obs
            simulated_seconds += result.duration
            accumulated_reward += result.reward
            action_seconds += result.action_seconds
            environment_seconds += result.environment_seconds
            last_q = result.q_value
            random_action_expectation += epsilon
            assert result.action is not None
            verb_index = int(result.action[HEADS.index("verb")])
            verb_samples.append(verb_index)
            succeeded = bool(result.info.get("success", True))
            successes += int(succeeded)
            if verb_index == VERB_INDEX["PLACE"]:
                place_attempts += 1
                place_successes += int(succeeded)
                place_masked_sum += result.place_masked_fraction
                place_states[result.place_state] += 1

            if replay.size >= args.learning_starts:
                update_budget += args.updates_per_step
                scheduled_updates = int(update_budget)
                update_budget -= scheduled_updates
            else:
                scheduled_updates = 0
            for _ in range(scheduled_updates):
                update_started = time.perf_counter()
                sample_started = time.perf_counter()
                batch = replay.sample(args.batch_size, rng, device)
                replay_sample_seconds += time.perf_counter() - sample_started
                optimizer.zero_grad(set_to_none=True)
                profile = decisions % args.log_interval == 0
                if profile:
                    _synchronize(device)
                    forward_started = time.perf_counter()
                loss, td_error = double_dqn_loss(online, target, batch)
                if profile:
                    _synchronize(device)
                    update_forward_seconds += time.perf_counter() - forward_started
                    backward_started = time.perf_counter()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    online.parameters(),
                    args.max_grad_norm,
                    foreach=device.type != "mps",
                )
                optimizer.step()
                if profile:
                    _synchronize(device)
                    update_backward_seconds += time.perf_counter() - backward_started
                    profiled_updates += 1
                update_seconds += time.perf_counter() - update_started
                last_loss_tensor = loss.detach()
                last_td_tensor = td_error.detach()
                last_grad_tensor = grad_norm.detach()
                updates += 1
                if updates % args.target_period == 0:
                    target.load_state_dict(online.state_dict())

            should_log = decisions == 1 or decisions % args.log_interval == 0
            if should_log:
                _synchronize(device)
                elapsed = max(time.perf_counter() - start, 1e-9)
                known, unknown, fresh = _build_fractions(env)
                metrics: dict[str, Any] = {
                    "env/automated_score": float(
                        result.info.get("score_automated", 0.0)
                    ),
                    "env/production_score": float(result.info.get("score_player", 0.0)),
                    "env/reward": result.reward,
                    "env/decisions_per_wall_second": decisions / elapsed,
                    "env/sim_seconds_per_wall_second": simulated_seconds / elapsed,
                    "action/verb_histogram": logger.histogram(verb_samples),
                    "action/success_rate": successes / decisions,
                    "action/place_success_rate": (
                        place_successes / place_attempts if place_attempts else 0.0
                    ),
                    "build/known_fraction": known,
                    "build/unknown_fraction": unknown,
                    "build/fresh_fraction": fresh,
                    "build/place_masked_fraction": (
                        place_masked_sum / place_attempts if place_attempts else 0.0
                    ),
                    "build/place_selected_known_legal": place_states["known_legal"],
                    "build/place_selected_known_blocked": place_states["known_blocked"],
                    "build/place_selected_unknown": place_states["unknown"],
                    "build/place_selected_stale": place_states["stale"],
                    "train/q_mean": last_q,
                    "train/updates_per_second": (
                        updates / update_seconds if update_seconds else 0.0
                    ),
                    "train/replay_size": replay.size,
                    "train/replay_bytes": replay.nbytes,
                    "train/replay_bytes_per_transition": replay.bytes_per_transition,
                    "explore/epsilon": epsilon,
                    "explore/random_action_fraction": (
                        random_action_expectation / decisions
                    ),
                    "perf/network_forward_ms": (
                        1000.0 * update_forward_seconds / max(profiled_updates, 1)
                    ),
                    "perf/network_backward_ms": (
                        1000.0 * update_backward_seconds / max(profiled_updates, 1)
                    ),
                    "perf/action_forward_ms": 1000.0 * action_seconds / decisions,
                    "perf/environment_step_ms": (
                        1000.0 * environment_seconds / decisions
                    ),
                    "perf/replay_sample_ms": (
                        1000.0 * replay_sample_seconds / max(updates, 1)
                    ),
                    "perf/update_ms": 1000.0 * update_seconds / max(updates, 1),
                }
                loss_text = ""
                if last_loss_tensor is not None:
                    assert last_td_tensor is not None and last_grad_tensor is not None
                    loss_value = float(last_loss_tensor.cpu())
                    if not np.isfinite(loss_value):
                        raise FloatingPointError(
                            f"non-finite loss at live step {decisions}"
                        )
                    metrics.update(
                        {
                            "train/loss": loss_value,
                            "train/td_error": float(last_td_tensor.cpu()),
                            "train/grad_norm": float(last_grad_tensor.cpu()),
                        }
                    )
                    loss_text = f" loss={loss_value:.6g}"
                logger.log(metrics, decisions)
                verb_samples.clear()
                print(
                    f"step={decisions}{loss_text} replay={replay.size} "
                    f"epsilon={epsilon:.4f} updates={updates} "
                    f"decisions_per_second={decisions / elapsed:.3f}",
                    flush=True,
                )

            if args.checkpoint_every and decisions % args.checkpoint_every == 0:
                try:
                    checkpoint = _save_checkpoint(
                        args.run_name, decisions, online, target, optimizer, args
                    )
                    print(f"checkpoint={checkpoint}", flush=True)
                except Exception as error:  # noqa: BLE001 - keep training alive
                    print(
                        f"checkpoint_failed step={decisions} error={error!r}",
                        file=sys.stderr,
                        flush=True,
                    )

            if result.done:
                episodes += 1
                try:
                    obs, _ = env.reset(seed=args.seed + episodes)
                except Exception as error:  # noqa: BLE001 - rebuild and continue
                    print(
                        f"environment_reset_failed error={error!r}",
                        file=sys.stderr,
                        flush=True,
                    )
                    _close_environment(env)
                    env, (obs, _) = _start_live_environment(args, args.seed + episodes)
    finally:
        _flush_accumulator(accumulator, replay)
        _close_environment(env)
        logger.finish()

    _synchronize(device)
    final_loss: float | str = "no-update"
    if last_loss_tensor is not None:
        final_loss = float(last_loss_tensor.cpu())
    result_summary: dict[str, float | str | int] = {
        "device": str(device),
        "final_loss": final_loss,
        "updates": updates,
        "decisions": decisions,
        "replay_size": replay.size,
        "bytes_per_transition": replay.bytes_per_transition,
    }
    print(
        "final " + " ".join(f"{key}={value}" for key, value in result_summary.items()),
        flush=True,
    )
    return result_summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fake", action="store_true")
    parser.add_argument("--port", type=int, default=27010)
    parser.add_argument("--speed", type=float, default=10.0)
    parser.add_argument("--episode-max-steps", type=int, default=256)
    parser.add_argument("--updates-per-step", type=float, default=1.0)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--run-name")
    parser.add_argument("--total-steps", type=int, default=20_000)
    parser.add_argument(
        "--wandb-mode", choices=("online", "offline", "disabled"), default="online"
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--replay-size", type=int, default=4096)
    parser.add_argument("--fake-replay-size", type=int, default=64)
    parser.add_argument("--learning-starts", type=int, default=16)
    parser.add_argument("--update-every", type=int, default=16)
    parser.add_argument("--target-period", type=int, default=10)
    parser.add_argument("--n-step", type=int, default=3)
    parser.add_argument("--discount", choices=("smdp", "legacy"), default="smdp")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=10.0)
    parser.add_argument("--eps-start", type=float, default=1.0)
    parser.add_argument("--eps-end", type=float, default=0.05)
    parser.add_argument("--eps-decay-steps", type=int, default=10_000)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--fake-horizon", type=int, default=64)
    args = parser.parse_args(argv)
    positive = (
        "total_steps",
        "torch_threads",
        "batch_size",
        "replay_size",
        "fake_replay_size",
        "learning_starts",
        "update_every",
        "target_period",
        "n_step",
        "eps_decay_steps",
        "log_interval",
        "fake_horizon",
        "port",
        "episode_max_steps",
    )
    if any(getattr(args, name) <= 0 for name in positive):
        parser.error("step, size, period, and thread arguments must be positive")
    if not (0 <= args.eps_start <= 1 and 0 <= args.eps_end <= 1):
        parser.error("epsilon values must be in [0, 1]")
    if args.speed <= 0 or args.updates_per_step < 0 or args.checkpoint_every < 0:
        parser.error(
            "speed must be positive; update/checkpoint rates cannot be negative"
        )
    return args


def main(argv: Sequence[str] | None = None) -> dict[str, float | str | int]:
    args = parse_args(argv)
    return train_fake(args) if args.fake else train_live(args)


if __name__ == "__main__":
    main()
