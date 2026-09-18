"""Threaded SM-ARQ training runner.

Environment workers perform I/O-bound semantic steps.  Replay insertion is
serialized, and all learner work remains on the main thread (and therefore on
one MPS context when the neural backend is available).
"""

from __future__ import annotations

import argparse
import inspect
import os
import pickle
import queue
import random
import signal
import sys
import threading
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from fle.smarq import contract as C
from fle.smarq.demos import (
    BurnerAutomationDemo,
    HandMiningDemo,
    collect_demonstration,
    masks_as_dict,
    semantic_action,
)
from fle.smarq.logging_ import JsonlLogger, episode_record


@dataclass
class TrainingConfig:
    run_name: str = "smarq"
    env: str = "fake"
    ports: tuple[int, ...] = (27000,)
    workers: int = 1
    replay_ratio: float = 4.0
    raster_tiles: int = C.RASTER_TILES_DEFAULT
    reward_mode: str = "automated"
    episode_game_minutes: float = C.EPISODE_GAME_MINUTES_DEFAULT
    decision_cap: int = 200
    discount_horizon: float = C.DISCOUNT_HORIZON_SECONDS_DEFAULT
    demos: int = 0
    total_decisions: int = 400
    checkpoint_every: int = 1_000
    resume: str | None = None
    seed: int = 1
    dry_run: bool = False
    run_root: Path = Path("runs")
    replay_capacity: int = 20_000
    force_fallback: bool = False

    @property
    def run_dir(self) -> Path:
        return Path(self.run_root) / self.run_name


@dataclass
class TrainingResult:
    decisions: int
    updates: int
    demo_transitions: int
    replay_size: int
    run_dir: Path
    checkpoint: Path
    backend: str
    wall_seconds: float
    simulated_seconds: float
    replay_memory_bytes: int


@dataclass(frozen=True)
class EpsilonSchedule:
    start: float = 1.0
    end: float = 0.05
    decay_decisions: int = 100_000

    def value(self, decision: int) -> float:
        fraction = min(1.0, max(0.0, decision / max(1, self.decay_decisions)))
        return float(self.start + fraction * (self.end - self.start))


EPSILON_DECAY_DECISIONS = 100_000


def epsilon_per_head(decision: int) -> dict[str, float]:
    """Separate schedules, with geometry retaining exploration the longest.

    The decay length must match the number of decisions a run will actually
    collect.  A schedule tuned for 100k decisions leaves epsilon near 0.9 for a
    run that collects 10k, so the policy never acts on anything it has learned
    and the run measures exploration rather than learning.
    """
    span = max(1, EPSILON_DECAY_DECISIONS)
    values = {"verb": EpsilonSchedule(decay_decisions=span).value(decision)}
    for head in C.HEADS:
        schedule = (
            EpsilonSchedule(end=0.25, decay_decisions=span * 2)
            if head == C.POSITION
            else EpsilonSchedule(decay_decisions=span)
        )
        values[head] = schedule.value(decision)
    return values


def set_epsilon_decay(decisions: int) -> None:
    global EPSILON_DECAY_DECISIONS
    EPSILON_DECAY_DECISIONS = max(1, int(decisions))


class FallbackReplay:
    """Small in-memory PER used only when the neural replay is unavailable."""

    def __init__(self, capacity: int = 128, seed: int = 0) -> None:
        self.capacity = max(1, int(capacity))
        self.transitions: list[C.Transition] = []
        self.priorities: list[float] = []
        self._next = 0
        self.rng = np.random.default_rng(seed)

    def add(self, transition: C.Transition, priority: float = 1.0) -> None:
        priority = max(float(priority), 1e-6)
        if len(self.transitions) < self.capacity:
            self.transitions.append(transition)
            self.priorities.append(priority)
        else:
            self.transitions[self._next] = transition
            self.priorities[self._next] = priority
            self._next = (self._next + 1) % self.capacity

    def sample(self, batch_size: int) -> tuple[list[C.Transition], np.ndarray]:
        if not self.transitions:
            raise ValueError("cannot sample an empty replay")
        weights = np.asarray(self.priorities, dtype=np.float64) ** 0.6
        weights /= weights.sum()
        count = min(int(batch_size), len(self.transitions))
        indices = self.rng.choice(len(self.transitions), size=count, replace=True, p=weights)
        return [self.transitions[int(i)] for i in indices], np.asarray(indices)

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        for index, priority in zip(indices, priorities, strict=True):
            self.priorities[int(index)] = max(float(priority), 1e-6)

    @property
    def max_priority(self) -> float:
        return max(self.priorities, default=1.0)

    def __len__(self) -> int:
        return len(self.transitions)

    def state_dict(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "transitions": self.transitions,
            "priorities": self.priorities,
            "next": self._next,
            "rng": self.rng.bit_generator.state,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.capacity = int(state["capacity"])
        self.transitions = list(state["transitions"])
        self.priorities = list(state["priorities"])
        self._next = int(state["next"])
        self.rng.bit_generator.state = state["rng"]


@dataclass
class PolicyDecision:
    action: C.Action
    q_value: float
    exploratory: bool
    epsilons: dict[str, float]


class FallbackRandomPolicy:
    """Deterministic-seed structural random policy for offline runner tests."""

    def __init__(self, vocab: C.VocabProtocol, seed: int = 0) -> None:
        self.vocab = vocab
        self.rng = np.random.default_rng(seed)

    def state_dict(self) -> dict[str, Any]:
        return {"rng": self.rng.bit_generator.state}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.rng.bit_generator.state = state["rng"]

    def _choice(self, mask: np.ndarray) -> int:
        choices = np.flatnonzero(np.asarray(mask, dtype=bool))
        if choices.size == 0:
            raise ValueError("no structurally valid choice")
        return int(self.rng.choice(choices))

    def select(
        self,
        observation: C.Observation,
        masks: C.Masks,
        decision: int,
    ) -> PolicyDecision:
        epsilons = epsilon_per_head(decision)
        legal_verbs = np.asarray(masks.verb, dtype=bool).copy()
        for name in ("PICKUP", "ROTATE", "INSERT", "EXTRACT", "SET_RECIPE"):
            if not masks.entity[C.VERB_INDEX[name]].any():
                legal_verbs[C.VERB_INDEX[name]] = False
        verb = C.VERBS[self._choice(legal_verbs)]
        kwargs: dict[str, Any] = {}
        sequence = C.HEAD_SEQUENCE[verb]
        if C.POSITION in sequence:
            position = int(self.rng.integers(observation.raster_tiles**2))
            kwargs["tile"] = C.position_to_tile(
                position, observation.raster_origin, observation.raster_tiles
            )
        if C.PROTOTYPE in sequence:
            kwargs["prototype"] = self.vocab.prototypes[self._choice(masks.prototype)]
        if C.DIRECTION in sequence:
            kwargs["direction"] = C.DIRECTIONS[int(self.rng.integers(C.N_DIRECTIONS))]
        if C.ENTITY in sequence:
            kwargs["entity_slot"] = self._choice(masks.entity[C.VERB_INDEX[verb]])
        if C.ITEM in sequence:
            kwargs["item"] = self.vocab.items[self._choice(masks.item)]
        if C.QUANTITY in sequence:
            kwargs["quantity"] = C.QUANTITIES[int(self.rng.integers(C.N_QUANTITIES))]
        if C.RECIPE in sequence:
            recipe_mask = masks.craft_recipe
            if verb == "SET_RECIPE" and callable(masks.recipe_for_entity):
                recipe_mask = masks.recipe_for_entity(kwargs["entity_slot"])
            kwargs["recipe"] = self.vocab.recipes[self._choice(recipe_mask)]
        if C.TECHNOLOGY in sequence:
            kwargs["technology"] = self.vocab.technologies[self._choice(masks.technology)]
        if C.DURATION in sequence:
            kwargs["duration_seconds"] = C.DURATIONS_SECONDS[
                int(self.rng.integers(C.N_DURATIONS))
            ]
        action = semantic_action(verb, observation, self.vocab, **kwargs)
        return PolicyDecision(action, 0.0, True, epsilons)


class FallbackLearner:
    def __init__(self, replay: FallbackReplay, seed: int = 0) -> None:
        self.replay = replay
        self.rng = np.random.default_rng(seed)
        self.updates = 0

    def update(self) -> dict[str, float]:
        transitions, indices = self.replay.sample(32)
        td_errors = np.asarray([transition.reward for transition in transitions], dtype=float)
        priorities = np.abs(td_errors) + 1e-3
        self.replay.update_priorities(indices, priorities)
        self.updates += 1
        abs_td = np.abs(td_errors)
        return {
            "loss": float(np.mean(np.where(abs_td < 1.0, 0.5 * abs_td**2, abs_td - 0.5))),
            "grad_norm": 0.0,
            "mean_max_q": 0.0,
            "td_error_mean": float(abs_td.mean()),
            "td_error_p95": float(np.percentile(abs_td, 95)),
        }

    def state_dict(self) -> dict[str, Any]:
        return {"updates": self.updates, "rng": self.rng.bit_generator.state}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.updates = int(state["updates"])
        self.rng.bit_generator.state = state["rng"]


class _ExternalPolicyAdapter:
    def __init__(self, policy: Any) -> None:
        self.policy = policy

    def select(
        self,
        observation: C.Observation,
        masks: C.Masks,
        decision: int,
    ) -> PolicyDecision:
        import torch

        epsilons = epsilon_per_head(decision)
        with torch.no_grad():
            encoded = self.policy.network.encode(observation)
            verb, heads, selected, chosen_q = self.policy.select_indices(
                encoded, masks, epsilon=epsilons
            )
        kwargs: dict[str, Any] = {}
        if C.POSITION in selected:
            kwargs["tile"] = C.position_to_tile(
                selected[C.POSITION], observation.raster_origin, observation.raster_tiles
            )
        if C.PROTOTYPE in selected:
            kwargs["prototype"] = self.policy.vocab.prototypes[selected[C.PROTOTYPE]]
        if C.DIRECTION in selected:
            kwargs["direction"] = C.DIRECTIONS[selected[C.DIRECTION]]
        if C.ENTITY in selected:
            kwargs["entity_slot"] = selected[C.ENTITY]
        if C.ITEM in selected:
            kwargs["item"] = self.policy.vocab.items[selected[C.ITEM]]
        if C.QUANTITY in selected:
            kwargs["quantity"] = C.QUANTITIES[selected[C.QUANTITY]]
        if C.RECIPE in selected:
            kwargs["recipe"] = self.policy.vocab.recipes[selected[C.RECIPE]]
        if C.TECHNOLOGY in selected:
            kwargs["technology"] = self.policy.vocab.technologies[
                selected[C.TECHNOLOGY]
            ]
        if C.DURATION in selected:
            kwargs["duration_seconds"] = C.DURATIONS_SECONDS[selected[C.DURATION]]
        action = semantic_action(C.VERBS[verb], observation, self.policy.vocab, **kwargs)
        if not np.array_equal(action.heads, heads):  # pragma: no cover - contract invariant
            raise RuntimeError("policy head decoding changed the selected indices")
        return PolicyDecision(
            action,
            float(chosen_q[-1]),
            any(value > 0 for value in epsilons.values()),
            epsilons,
        )

    def state_dict(self) -> dict[str, Any]:
        rng = getattr(self.policy, "rng", None)
        return {"rng": rng.bit_generator.state} if rng is not None else {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        rng = getattr(self.policy, "rng", None)
        if rng is not None and "rng" in state:
            rng.bit_generator.state = state["rng"]


@dataclass
class _Backend:
    name: str
    replay: Any
    learner: Any
    policies: list[Any]
    state_dict: Callable[[], dict[str, Any]]
    load_state_dict: Callable[[dict[str, Any]], None]


class _ExternalLearnerAdapter:
    def __init__(self, learner: Any, replay: Any) -> None:
        self.learner = learner
        self.replay = replay

    def update(self) -> dict[str, float]:
        stats = self.learner.update_from_replay(self.replay)
        td_errors = np.asarray(stats.td_errors, dtype=float)
        return {
            "loss": float(stats.loss),
            "grad_norm": float(stats.grad_norm),
            "mean_max_q": float(stats.mean_q),
            "td_error_mean": float(stats.mean_abs_td_error),
            "td_error_p95": float(np.percentile(td_errors, 95)),
        }


def _supported_kwargs(callable_: Any, values: dict[str, Any]) -> dict[str, Any]:
    try:
        signature = inspect.signature(callable_)
    except (TypeError, ValueError):
        return values
    if any(p.kind == p.VAR_KEYWORD for p in signature.parameters.values()):
        return values
    return {key: value for key, value in values.items() if key in signature.parameters}


def _build_backend(
    env: C.SemanticEnvProtocol, config: TrainingConfig
) -> _Backend:
    """Lazily use the neural modules when their complete public API exists."""
    try:
        if config.force_fallback:
            raise ImportError("fallback explicitly requested by offline harness")
        import torch

        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS is unavailable")
        from fle.smarq.learner import DoubleDQNLearner
        from fle.smarq.net import SMARQNetwork
        from fle.smarq.policy import AutoregressivePolicy
        from fle.smarq.replay import PrioritizedReplay

        torch.manual_seed(config.seed)
        network = SMARQNetwork(
            **_supported_kwargs(
                SMARQNetwork,
                {"vocab": env.vocab, "raster_tiles": config.raster_tiles},
            )
        )
        replay = PrioritizedReplay(
            **_supported_kwargs(
                PrioritizedReplay,
                {"capacity": config.replay_capacity, "seed": config.seed},
            )
        )
        learner = DoubleDQNLearner(
            **_supported_kwargs(
                DoubleDQNLearner,
                {
                    "network": network,
                    "horizon_seconds": config.discount_horizon,
                    "device": "mps",
                    "seed": config.seed,
                },
            )
        )
        learner_adapter = _ExternalLearnerAdapter(learner, replay)
        policies = [
            _ExternalPolicyAdapter(
                AutoregressivePolicy(network, env.vocab, seed=config.seed + 10_000 + worker)
            )
            for worker in range(config.workers)
        ]

        def state_dict() -> dict[str, Any]:
            return {
                "learner": {
                    "online": learner.online.state_dict(),
                    "target": learner.target.state_dict(),
                    "optimizer": learner.optimizer.state_dict(),
                    "updates": learner.updates,
                },
                "replay": replay,
                "policies": [policy.state_dict() for policy in policies],
            }

        def load_state_dict(state: dict[str, Any]) -> None:
            learner_state = state["learner"]
            learner.online.load_state_dict(learner_state["online"])
            learner.target.load_state_dict(learner_state["target"])
            learner.optimizer.load_state_dict(learner_state["optimizer"])
            learner.updates = int(learner_state["updates"])
            replay.__dict__.update(state["replay"].__dict__)
            for policy, policy_state in zip(policies, state["policies"], strict=False):
                policy.load_state_dict(policy_state)

        print("SM-ARQ backend: neural learner/policy on MPS", file=sys.stderr, flush=True)
        return _Backend("neural", replay, learner_adapter, policies, state_dict, load_state_dict)
    except (ImportError, AttributeError, TypeError, RuntimeError) as exc:
        print(
            f"SM-ARQ backend: FALLBACK random policy ({type(exc).__name__}: {exc})",
            file=sys.stderr,
            flush=True,
        )

    replay = FallbackReplay(min(config.replay_capacity, 128), config.seed)
    learner = FallbackLearner(replay, config.seed + 1)
    policies = [
        FallbackRandomPolicy(env.vocab, config.seed + 10_000 + worker)
        for worker in range(config.workers)
    ]

    def state_dict() -> dict[str, Any]:
        return {
            "learner": learner.state_dict(),
            "replay": replay.state_dict(),
            "policies": [policy.state_dict() for policy in policies],
        }

    def load_state_dict(state: dict[str, Any]) -> None:
        learner.load_state_dict(state["learner"])
        replay.load_state_dict(state["replay"])
        for policy, policy_state in zip(policies, state["policies"], strict=False):
            policy.load_state_dict(policy_state)

    return _Backend("fallback", replay, learner, policies, state_dict, load_state_dict)


def _default_env_factory(
    config: TrainingConfig,
) -> Callable[[int, int], C.SemanticEnvProtocol]:
    def factory(worker: int, seed: int) -> C.SemanticEnvProtocol:
        tick_budget = int(config.episode_game_minutes * 60 * C.TICKS_PER_SECOND)
        if config.env == "fake":
            from fle.smarq.fake import FakeSemanticEnv

            return FakeSemanticEnv(
                raster_tiles=config.raster_tiles,
                tick_budget=tick_budget,
                decision_cap=config.decision_cap,
                reward_mode=config.reward_mode,
                seed=seed,
            )
        try:
            from fle.smarq.env import SemanticEnv
        except ImportError as exc:  # pragma: no cover - requires live session
            raise RuntimeError("live environment module is not available") from exc
        port = config.ports[worker % len(config.ports)]
        kwargs = _supported_kwargs(
            SemanticEnv,
            {
                "port": port,
                "raster_tiles": config.raster_tiles,
                "tick_budget": tick_budget,
                "decision_cap": config.decision_cap,
                "reward_mode": config.reward_mode,
                "seed": seed,
            },
        )
        return SemanticEnv(**kwargs)

    return factory


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> Path:
    """Atomically persist a checkpoint in its destination directory."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, destination)
    return destination


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as stream:
        value = pickle.load(stream)
    if not isinstance(value, dict):
        raise ValueError("checkpoint payload is not a mapping")
    return value


@dataclass
class _StepEvent:
    worker: int
    transition: C.Transition
    duration_game_seconds: float


@dataclass
class _SharedState:
    claimed: int = 0
    completed: int = 0
    simulated_seconds: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock)


def _initial_priority(replay: Any) -> float:
    value = getattr(replay, "max_priority", 1.0)
    return float(value() if callable(value) else value)


def _learner_update(learner: Any) -> dict[str, float]:
    value = learner.update()
    if value is None:
        return {}
    if isinstance(value, dict):
        return {key: float(item) for key, item in value.items() if np.isscalar(item)}
    if hasattr(value, "__dict__"):
        return {
            key: float(item)
            for key, item in vars(value).items()
            if np.isscalar(item)
        }
    return {"loss": float(value)}


def _memory_bytes(value: Any, seen: set[int] | None = None) -> int:
    """Estimate owned replay memory while preserving shared-array identity."""
    seen = set() if seen is None else seen
    identity = id(value)
    if identity in seen:
        return 0
    seen.add(identity)
    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    if isinstance(value, dict):
        return sys.getsizeof(value) + sum(
            _memory_bytes(key, seen) + _memory_bytes(item, seen)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return sys.getsizeof(value) + sum(_memory_bytes(item, seen) for item in value)
    if hasattr(value, "__dict__"):
        return sys.getsizeof(value) + _memory_bytes(vars(value), seen)
    return sys.getsizeof(value)


def run_training(
    config: TrainingConfig,
    *,
    env_factory: Callable[[int, int], C.SemanticEnvProtocol] | None = None,
) -> TrainingResult:
    """Run collection and learning to ``config.total_decisions``."""
    if config.workers < 1:
        raise ValueError("workers must be positive")
    if config.replay_ratio <= 0:
        raise ValueError("replay_ratio must be positive")
    if config.reward_mode not in C.REWARD_MODES:
        raise ValueError(f"unknown reward mode {config.reward_mode!r}")
    if config.env == "live" and len(config.ports) < config.workers:
        raise ValueError("live training requires one distinct --ports entry per worker")
    random.seed(config.seed)
    np.random.seed(config.seed)
    factory = env_factory or _default_env_factory(config)
    envs = [factory(worker, config.seed + worker) for worker in range(config.workers)]
    backend = _build_backend(envs[0], config)
    shared = _SharedState()
    updates = 0
    demo_transitions = 0
    resume_payload: dict[str, Any] | None = None
    if config.resume:
        resume_payload = load_checkpoint(config.resume)
        backend.load_state_dict(resume_payload["backend_state"])
        shared.claimed = int(resume_payload.get("decisions", 0))
        shared.completed = shared.claimed
        updates = int(resume_payload.get("updates", 0))
        demo_transitions = int(resume_payload.get("demo_transitions", 0))
    initial_updates = updates

    run_dir = config.run_dir
    logger = JsonlLogger(run_dir)
    replay_lock = threading.Lock()
    model_lock = threading.Lock()
    if config.demos and resume_payload is None:
        demo_env = envs[0]
        for index in range(config.demos):
            if config.env == "live":
                # Toy coordinates do not exist on a generated map, so plan the
                # same burner chain against the world this worker actually has.
                from fle.smarq.live_demos import LiveBurnerDemo, nearest_resource_tile

                target = nearest_resource_tile(demo_env.instance)
                policy = LiveBurnerDemo(
                    target=target,
                    drills=2 if index % 2 == 0 else 1,
                    wait_blocks=4,
                    max_actions=18,
                )
            else:
                policy = (
                    BurnerAutomationDemo(craft_furnace=index % 2 == 0)
                    if index % 2 == 0
                    else HandMiningDemo()
                )
            with replay_lock:
                result = collect_demonstration(
                    demo_env,
                    policy,
                    backend.replay,
                    seed=config.seed + 50_000 + index,
                    initial_priority=10.0,
                )
            demo_transitions += len(result.transitions)

    stop_event = threading.Event()
    events: queue.Queue[_StepEvent] = queue.Queue(maxsize=max(8, config.workers * 4))
    errors: queue.Queue[BaseException] = queue.Queue()
    original_handlers: dict[int, Any] = {}

    def request_stop(signum: int, _frame: Any) -> None:
        print(f"received signal {signum}; finishing current semantic steps", file=sys.stderr)
        stop_event.set()

    if threading.current_thread() is threading.main_thread():
        for signum in (signal.SIGINT, signal.SIGTERM):
            original_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, request_stop)

    start_wall = time.perf_counter()
    starting_decisions = shared.completed

    def worker_loop(worker: int, env: C.SemanticEnvProtocol) -> None:
        episode_index = 0
        decision_index = starting_decisions + (
            (worker - starting_decisions % config.workers) % config.workers
        )
        try:
            observation, masks = env.reset(config.seed + worker)
            episode_start = time.perf_counter()
            episode_return = 0.0
            max_automated = observation.automated_production_score
            failures: Counter[str] = Counter()
            verbs: Counter[str] = Counter()
            episode_steps = 0
            last_result: C.StepResult | None = None

            def finish_episode(result: C.StepResult, end_reason: str) -> None:
                nonlocal episode_index, episode_steps
                logger.log_episode(
                    episode_record(
                        episode_return=episode_return,
                        final_automated_score=result.automated_production_score,
                        max_automated_score=max_automated,
                        final_production_score=result.production_score,
                        decisions=result.episode_decisions,
                        simulated_ticks=result.episode_ticks,
                        wall_seconds=time.perf_counter() - episode_start,
                        failure_reason_histogram=failures,
                        verb_histogram=verbs,
                        end_reason=end_reason,
                        worker=worker,
                        episode=episode_index,
                    )
                )
                episode_index += 1
                episode_steps = 0

            while not stop_event.is_set():
                if decision_index >= config.total_decisions:
                    break
                with shared.lock:
                    shared.claimed += 1
                with model_lock:
                    selected = backend.policies[worker].select(
                        observation, masks, decision_index
                    )
                result, next_masks = env.step(selected.action)
                last_result = result
                episode_steps += 1
                transition = C.Transition(
                    observation=observation.as_dict(),
                    action_heads=selected.action.heads.copy(),
                    verb=C.VERB_INDEX[selected.action.verb],
                    reward=result.reward,
                    tau_seconds=result.duration_game_seconds,
                    next_observation=result.observation.as_dict(),
                    done=result.done,
                    success=result.success,
                    failure_reason=result.failure_reason,
                    masks=masks_as_dict(masks),
                    next_masks=masks_as_dict(next_masks),
                )
                priority = max(_initial_priority(backend.replay), abs(result.reward) + 1e-3)
                with replay_lock:
                    backend.replay.add(transition, priority=priority)
                requested = selected.action.quantity
                executed = result.info.get("executed_quantity")
                if executed is None:
                    executed = requested if result.success and requested != "ALL" else None
                logger.log_step(
                    worker,
                    {
                        "tick": result.observation.tick,
                        "simulated_seconds_elapsed": result.episode_ticks
                        / C.TICKS_PER_SECOND,
                        "wall_seconds": result.wall_seconds,
                        "game_seconds_per_wall_second": result.duration_game_seconds
                        / max(result.wall_seconds, 1e-12),
                        "production_score": result.production_score,
                        "automated_production_score": result.automated_production_score,
                        "delta_production_score": result.delta_production_score,
                        "delta_automated_production_score": result.delta_automated_production_score,
                        "verb": selected.action.verb,
                        "heads": selected.action.heads,
                        "failure_reason": result.failure_reason,
                        "success": result.success,
                        "requested_quantity": requested,
                        "executed_quantity": executed,
                        "q_value": selected.q_value,
                        "td_error": abs(result.reward),
                        "per_priority": priority,
                        "exploratory": selected.exploratory,
                        "epsilon_per_head": selected.epsilons,
                        "worker": worker,
                        "decision": decision_index,
                    },
                )
                episode_return += result.reward
                max_automated = max(max_automated, result.automated_production_score)
                failures[result.failure_reason] += 1
                verbs[selected.action.verb] += 1
                with shared.lock:
                    shared.completed += 1
                    shared.simulated_seconds += result.duration_game_seconds
                    reached_total = shared.completed >= config.total_decisions
                while True:
                    try:
                        events.put(
                            _StepEvent(worker, transition, result.duration_game_seconds),
                            timeout=0.1,
                        )
                        break
                    except queue.Full:
                        if not errors.empty():
                            break
                        continue
                forced_end = reached_total or stop_event.is_set()
                if result.done or forced_end:
                    end_reason = result.end_reason or (
                        "signal" if stop_event.is_set() else "total_decisions"
                    )
                    finish_episode(result, end_reason)
                    if forced_end:
                        break
                    observation, masks = env.reset(
                        config.seed + worker + episode_index * config.workers
                    )
                    episode_start = time.perf_counter()
                    episode_return = 0.0
                    max_automated = observation.automated_production_score
                    failures.clear()
                    verbs.clear()
                else:
                    observation, masks = result.observation, next_masks
                decision_index += config.workers
            if episode_steps and last_result is not None:
                finish_episode(
                    last_result,
                    "signal" if stop_event.is_set() else "total_decisions",
                )
        except BaseException as exc:
            errors.put(exc)
            stop_event.set()
        finally:
            env.close()

    threads = [
        threading.Thread(
            target=worker_loop,
            args=(worker, env),
            name=f"smarq-env-{worker}",
            daemon=False,
        )
        for worker, env in enumerate(envs)
    ]
    for thread in threads:
        thread.start()

    processed = 0
    metric_values: list[dict[str, float]] = []
    metric_start = start_wall
    checkpoint_path = run_dir / "checkpoint_final.pkl"

    def checkpoint(path: Path) -> None:
        with model_lock, replay_lock:
            payload = {
                "schema": C.SCHEMA_VERSION,
                "config": {**asdict(config), "run_root": str(config.run_root)},
                "decisions": shared.completed,
                "updates": updates,
                "demo_transitions": demo_transitions,
                "backend_name": backend.name,
                "backend_state": backend.state_dict(),
                "python_random_state": random.getstate(),
                "numpy_random_state": np.random.get_state(),
            }
        save_checkpoint(path, payload)

    def log_metric() -> None:
        nonlocal metric_start
        now = time.perf_counter()
        elapsed = max(now - start_wall, 1e-12)
        window_elapsed = max(now - metric_start, 1e-12)
        keys = ("loss", "grad_norm", "mean_max_q", "td_error_mean", "td_error_p95")
        means = {
            key: float(np.mean([value.get(key, 0.0) for value in metric_values]))
            for key in keys
        }
        eps = epsilon_per_head(shared.completed)
        logger.log_metrics(
            {
                **means,
                "epsilon": float(np.mean(list(eps.values()))),
                "replay_size": len(backend.replay),
                "replay_ratio": (updates - initial_updates) / max(1, processed),
                "updates_per_second": len(metric_values) / window_elapsed,
                "transitions_per_wall_second": processed / elapsed,
                "simulated_seconds_per_wall_second": shared.simulated_seconds / elapsed,
                "updates": updates,
                "transitions": processed,
                "wall_seconds": elapsed,
            }
        )
        metric_values.clear()
        metric_start = now

    try:
        while any(thread.is_alive() for thread in threads) or not events.empty():
            if not errors.empty():
                raise errors.get()
            try:
                events.get(timeout=0.1)
            except queue.Empty:
                continue
            processed += 1
            target_updates = initial_updates + processed * config.replay_ratio
            while updates < target_updates:
                with model_lock, replay_lock:
                    metrics = _learner_update(backend.learner)
                metric_values.append(metrics)
                updates += 1
                if updates % 100 == 0:
                    log_metric()
                if config.checkpoint_every > 0 and updates % config.checkpoint_every == 0:
                    checkpoint(run_dir / f"checkpoint_{updates:09d}.pkl")
            events.task_done()
        for thread in threads:
            thread.join()
        if not errors.empty():
            raise errors.get()
        if metric_values or updates == 0:
            log_metric()
        checkpoint(checkpoint_path)
    finally:
        stop_event.set()
        for thread in threads:
            thread.join(timeout=10)
        try:
            checkpoint(checkpoint_path)
        finally:
            logger.close()
            for signum, handler in original_handlers.items():
                signal.signal(signum, handler)

    elapsed = time.perf_counter() - start_wall
    return TrainingResult(
        decisions=shared.completed,
        updates=updates,
        demo_transitions=demo_transitions,
        replay_size=len(backend.replay),
        run_dir=run_dir,
        checkpoint=checkpoint_path,
        backend=backend.name,
        wall_seconds=elapsed,
        simulated_seconds=shared.simulated_seconds,
        replay_memory_bytes=_memory_bytes(backend.replay),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="smarq")
    parser.add_argument("--env", choices=("fake", "live"), default="fake")
    parser.add_argument("--ports", default="27000")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--replay-ratio",
        type=float,
        default=4.0,
        help="learner updates per environment step; measured sustainable rate on this "
        "machine is about 0.5 at batch 128 on MPS, so values below 1 are legitimate",
    )
    parser.add_argument("--raster-tiles", type=int, default=C.RASTER_TILES_DEFAULT)
    parser.add_argument("--reward-mode", choices=C.REWARD_MODES, default="automated")
    parser.add_argument(
        "--episode-game-minutes", type=float, default=C.EPISODE_GAME_MINUTES_DEFAULT
    )
    parser.add_argument("--decision-cap", type=int, default=200)
    parser.add_argument("--discount-horizon", type=float, default=3600.0)
    parser.add_argument("--demos", type=int, default=0)
    parser.add_argument(
        "--epsilon-decay-decisions",
        type=int,
        default=100_000,
        help="decisions over which per-head epsilon anneals; set it to the number "
        "of decisions the run will really collect",
    )
    parser.add_argument("--total-decisions", type=int)
    parser.add_argument("--checkpoint-every", type=int, default=1_000)
    parser.add_argument("--resume")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    total_decisions = args.total_decisions
    if total_decisions is None:
        total_decisions = 400 if args.dry_run else 100_000
    set_epsilon_decay(args.epsilon_decay_decisions)
    config = TrainingConfig(
        run_name=args.run_name,
        env="fake" if args.dry_run else args.env,
        ports=tuple(int(value) for value in args.ports.split(",") if value),
        workers=args.workers,
        replay_ratio=args.replay_ratio,
        raster_tiles=args.raster_tiles,
        reward_mode=args.reward_mode,
        episode_game_minutes=args.episode_game_minutes,
        decision_cap=args.decision_cap,
        discount_horizon=args.discount_horizon,
        demos=args.demos,
        total_decisions=total_decisions,
        checkpoint_every=args.checkpoint_every,
        resume=args.resume,
        seed=args.seed,
        dry_run=args.dry_run,
    )
    result = run_training(config)
    print(
        f"decisions={result.decisions} updates={result.updates} "
        f"replay={result.replay_size} backend={result.backend} "
        f"wall={result.wall_seconds:.3f}s run_dir={result.run_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
