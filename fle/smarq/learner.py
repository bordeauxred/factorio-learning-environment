"""Double-DQN learner for the SM-ARQ internal and semantic transitions."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from fle.smarq import contract as C
from fle.smarq.net import EncodedState, SMARQNetwork
from fle.smarq.policy import AutoregressivePolicy, mask_for_head, masked_argmax
from fle.smarq.replay import PrioritizedReplay, ReplayBatch


def smdp_target(
    reward: float,
    tau_seconds: float,
    next_value: float,
    done: bool = False,
    horizon_seconds: float = C.DISCOUNT_HORIZON_SECONDS_DEFAULT,
) -> float:
    """Standalone scalar Bellman target, useful for arithmetic checks."""
    return float(reward) + (
        0.0
        if done
        else C.smdp_discount(float(tau_seconds), horizon_seconds) * float(next_value)
    )


def default_device() -> torch.device:
    """Prefer Apple MPS and otherwise provide a deterministic CPU fallback."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class LearnerStats:
    loss: float
    grad_norm: float
    grad_clipped: bool
    mean_q: float
    mean_abs_td_error: float
    updates: int
    target_updated: bool
    td_errors: np.ndarray


class DoubleDQNLearner:
    """Train all internal decisions and the final SMDP decision together."""

    def __init__(
        self,
        network: SMARQNetwork,
        *,
        device: str | torch.device | None = None,
        learning_rate: float = 2e-4,
        batch_size: int = 128,
        grad_clip: float = 10.0,
        target_update_interval: int = 2_000,
        horizon_seconds: float = C.DISCOUNT_HORIZON_SECONDS_DEFAULT,
        n_step: int = 1,
        weight_decay: float = 1e-4,
        mixed_precision: bool | None = None,
    ) -> None:
        if target_update_interval <= 0:
            raise ValueError("target_update_interval must be positive")
        if horizon_seconds <= 0:
            raise ValueError("horizon_seconds must be positive")
        if n_step <= 0:
            raise ValueError("n_step must be positive")
        self.device = torch.device(device) if device is not None else default_device()
        self.online = network.to(self.device)
        self.target = copy.deepcopy(network).to(self.device)
        self.target.eval()
        for parameter in self.target.parameters():
            parameter.requires_grad_(False)
        self.optimizer = torch.optim.AdamW(
            self.online.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.batch_size = int(batch_size)
        self.grad_clip = float(grad_clip)
        self.target_update_interval = int(target_update_interval)
        self.horizon_seconds = float(horizon_seconds)
        self.n_step = int(n_step)
        self.mixed_precision = (
            self.device.type == "mps" if mixed_precision is None else mixed_precision
        )
        self.updates = 0
        self.policy = AutoregressivePolicy(self.online, network.vocab)

    def discount(self, tau_seconds: float | Tensor) -> float | Tensor:
        if isinstance(tau_seconds, Tensor):
            return torch.pow(
                torch.tensor(2.0, device=tau_seconds.device),
                -tau_seconds / self.horizon_seconds,
            )
        return C.smdp_discount(float(tau_seconds), self.horizon_seconds)

    def semantic_target(
        self,
        reward: float,
        tau_seconds: float,
        done: bool,
        next_value: float,
    ) -> float:
        return smdp_target(reward, tau_seconds, next_value, done, self.horizon_seconds)

    @staticmethod
    def internal_target(next_q_values: Tensor, legal: np.ndarray | None = None) -> Tensor:
        """The internal transition has r=0, tau=0 and therefore Gamma=1."""
        if legal is None:
            return next_q_values.max()
        legal_tensor = torch.as_tensor(legal, device=next_q_values.device, dtype=torch.bool)
        if not legal_tensor.any():
            raise ValueError("internal transition has no legal next decision")
        return next_q_values.masked_fill(~legal_tensor, -torch.inf).max()

    @staticmethod
    def _slice(encoded: EncodedState, index: int) -> EncodedState:
        return encoded.select(index)

    @staticmethod
    def _prefix(selected: Tensor, indices: Tensor) -> dict[str, Tensor]:
        rows = selected.index_select(0, indices)
        return {head: rows[:, C.HEAD_INDEX[head]] for head in C.HEADS}

    def _legal_batch(
        self,
        masks: Sequence[C.Masks | Mapping[str, Any] | None],
        head: str,
        verbs: np.ndarray,
        selected: Tensor,
        size: int,
    ) -> Tensor:
        """Collate Python mask metadata once, then mask Q values on device."""
        if head in (C.POSITION, C.DIRECTION, C.QUANTITY, C.DURATION):
            return torch.ones((len(masks), size), dtype=torch.bool, device=self.device)
        entity_choices = None
        if head in (C.ITEM, C.RECIPE):
            entity_choices = (
                selected[:, C.HEAD_INDEX[C.ENTITY]].detach().cpu().numpy()
            )
        rows = []
        for index, (mask, verb) in enumerate(zip(masks, verbs, strict=True)):
            prefix = {}
            if entity_choices is not None and entity_choices[index] >= 0:
                prefix[C.ENTITY] = int(entity_choices[index])
            row = mask_for_head(mask, head, int(verb), prefix, size)
            if not row.any():
                raise ValueError(f"no legal {head} choice for {C.VERBS[int(verb)]}")
            rows.append(row)
        return torch.from_numpy(np.stack(rows)).to(self.device)

    @staticmethod
    def _sequence_tables() -> tuple[np.ndarray, np.ndarray]:
        max_depth = max(len(sequence) for sequence in C.HEAD_SEQUENCE.values())
        heads = np.full((C.N_VERBS, max_depth), -1, dtype=np.int64)
        lengths = np.zeros(C.N_VERBS, dtype=np.int64)
        for verb, name in enumerate(C.VERBS):
            sequence = C.HEAD_SEQUENCE[name]
            lengths[verb] = len(sequence)
            for depth, head in enumerate(sequence):
                heads[verb, depth] = C.HEAD_INDEX[head]
        return heads, lengths

    def _head_q(
        self,
        network: SMARQNetwork,
        encoded: EncodedState,
        indices: Tensor,
        head: str,
        verbs: Tensor,
        selected: Tensor,
    ) -> Tensor:
        return network.q_head(
            encoded.index_select(indices),
            head,
            verbs.index_select(0, indices),
            self._prefix(selected, indices),
        )

    @staticmethod
    def _head_groups(
        codes: np.ndarray,
        depth: int,
        head_index: int,
        head: str,
        verbs: np.ndarray,
        coarse_move: bool,
    ) -> tuple[np.ndarray, ...]:
        group = np.flatnonzero(codes[:, depth] == head_index)
        if not coarse_move or head != C.POSITION or group.size == 0:
            return (group,)
        move = verbs[group] == C.VERB_INDEX["MOVE_TO"]
        return tuple(part for part in (group[move], group[~move]) if part.size)

    def _batched_next_values(
        self,
        online_next: EncodedState,
        target_next: EncodedState,
        transitions: Sequence[C.Transition],
    ) -> Tensor:
        """Online autoregressive argmax and target evaluation, fully batched."""
        batch_size = len(transitions)
        values = torch.zeros(batch_size, device=self.device, dtype=torch.float32)
        active_np = np.flatnonzero([not transition.done for transition in transitions])
        if active_np.size == 0:
            return values
        root_indices = torch.as_tensor(active_np, device=self.device)
        online_encoded = online_next.index_select(root_indices)
        target_encoded = target_next.index_select(root_indices)
        active_transitions = [transitions[index] for index in active_np]
        masks = [transition.next_masks for transition in active_transitions]

        verb_q = self.online.q_verbs(online_encoded)
        verb_legal = np.stack(
            [
                self.policy._constructible_verbs(mask, online_encoded.entities.shape[1])
                for mask in masks
            ]
        )
        verb_legal_t = torch.from_numpy(verb_legal).to(self.device)
        verbs = verb_q.masked_fill(~verb_legal_t, -torch.inf).argmax(dim=1)
        verbs_np = verbs.detach().cpu().numpy()
        sequence_heads, sequence_lengths = self._sequence_tables()
        codes = sequence_heads[verbs_np]
        lengths = sequence_lengths[verbs_np]
        selected = torch.full(
            (len(active_np), len(C.HEADS)), -1, dtype=torch.long, device=self.device
        )
        local_values = torch.zeros(
            len(active_np), device=self.device, dtype=torch.float32
        )

        for depth in range(codes.shape[1]):
            for head_index, head in enumerate(C.HEADS):
                for group_np in self._head_groups(
                    codes,
                    depth,
                    head_index,
                    head,
                    verbs_np,
                    self.online.coarse_move,
                ):
                    if group_np.size == 0:
                        continue
                    group = torch.as_tensor(group_np, device=self.device)
                    group_masks = [masks[index] for index in group_np]
                    group_verbs_np = verbs_np[group_np]
                    online_q = self._head_q(
                        self.online, online_encoded, group, head, verbs, selected
                    )
                    legal = self._legal_batch(
                        group_masks,
                        head,
                        group_verbs_np,
                        selected.index_select(0, group),
                        online_q.shape[1],
                    )
                    choices = online_q.masked_fill(~legal, -torch.inf).argmax(dim=1)

                    final_np = lengths[group_np] == depth + 1
                    if final_np.any():
                        final_local_np = np.flatnonzero(final_np)
                        final_local = torch.as_tensor(final_local_np, device=self.device)
                        final_group = group.index_select(0, final_local)
                        target_q = self._head_q(
                            self.target,
                            target_encoded,
                            final_group,
                            head,
                            verbs,
                            selected,
                        )
                        chosen = choices.index_select(0, final_local)
                        evaluated = target_q.gather(1, chosen[:, None]).squeeze(1).float()
                        local_values.index_copy_(0, final_group, evaluated)
                    selected[group, head_index] = choices
        values.index_copy_(0, root_indices, local_values)
        return values

    def _vectorized_terms(
        self,
        transitions: Sequence[C.Transition],
        online_state: EncodedState,
        target_state: EncodedState,
        online_next: EncodedState,
        target_next: EncodedState,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return per-transition loss, max TD error, and mean chosen Q."""
        batch_size = len(transitions)
        verbs_np = np.fromiter(
            (int(transition.verb) for transition in transitions),
            dtype=np.int64,
            count=batch_size,
        )
        verbs = torch.from_numpy(verbs_np).to(self.device)
        action_heads_np = np.stack(
            [np.asarray(transition.action_heads, dtype=np.int64) for transition in transitions]
        )
        masks = [transition.masks for transition in transitions]
        sequence_heads, sequence_lengths_table = self._sequence_tables()
        codes = sequence_heads[verbs_np]
        lengths_np = sequence_lengths_table[verbs_np]
        lengths = torch.from_numpy(lengths_np).to(self.device)
        max_depth = codes.shape[1]

        verb_prediction = self.online.q_verbs(online_state).gather(
            1, verbs[:, None]
        ).squeeze(1).float()
        predictions = [verb_prediction] + [verb_prediction.new_zeros(batch_size) for _ in range(max_depth)]
        bootstraps = [verb_prediction.new_zeros(batch_size) for _ in range(max_depth)]
        selected = torch.full(
            (batch_size, len(C.HEADS)), -1, dtype=torch.long, device=self.device
        )

        for depth in range(max_depth):
            for head_index, head in enumerate(C.HEADS):
                for group_np in self._head_groups(
                    codes,
                    depth,
                    head_index,
                    head,
                    verbs_np,
                    self.online.coarse_move,
                ):
                    if group_np.size == 0:
                        continue
                    group = torch.as_tensor(group_np, device=self.device)
                    choices_np = action_heads_np[group_np, head_index]
                    if (choices_np < 0).any():
                        verb = C.VERBS[verbs_np[group_np[0]]]
                        raise ValueError(f"transition is missing {head} for {verb}")
                    choices = torch.as_tensor(choices_np, device=self.device)
                    online_q = self._head_q(
                        self.online, online_state, group, head, verbs, selected
                    )
                    chosen_q = online_q.gather(1, choices[:, None]).squeeze(1).float()
                    predictions[depth + 1] = predictions[depth + 1].index_add(
                        0, group, chosen_q
                    )
                    with torch.no_grad():
                        legal = self._legal_batch(
                            [masks[index] for index in group_np],
                            head,
                            verbs_np[group_np],
                            selected.index_select(0, group),
                            online_q.shape[1],
                        )
                        greedy = online_q.detach().masked_fill(~legal, -torch.inf).argmax(dim=1)
                        target_q = self._head_q(
                            self.target, target_state, group, head, verbs, selected
                        )
                        bootstrap = target_q.gather(1, greedy[:, None]).squeeze(1).float()
                        bootstraps[depth] = bootstraps[depth].index_add(
                            0, group, bootstrap
                        )
                    selected[group, head_index] = choices

        with torch.no_grad():
            next_values = self._batched_next_values(
                online_next, target_next, transitions
            )
            rewards = torch.as_tensor(
                [transition.reward for transition in transitions], device=self.device
            )
            tau = torch.as_tensor(
                [transition.tau_seconds for transition in transitions], device=self.device
            )
            done = torch.as_tensor(
                [transition.done for transition in transitions],
                device=self.device,
                dtype=torch.bool,
            )
            external = rewards + (~done) * self.discount(tau) * next_values

        targets = [bootstraps[0]]
        valid = [torch.ones(batch_size, device=self.device, dtype=torch.bool)]
        for depth in range(max_depth):
            active = lengths > depth
            final = lengths == depth + 1
            if depth + 1 < max_depth:
                target = torch.where(final, external, bootstraps[depth + 1])
            else:
                target = external
            targets.append(target)
            valid.append(active)
        prediction_matrix = torch.stack(predictions, dim=1)
        target_matrix = torch.stack(targets, dim=1).to(prediction_matrix.dtype)
        valid_matrix = torch.stack(valid, dim=1)
        element_loss = F.smooth_l1_loss(
            prediction_matrix, target_matrix, reduction="none"
        )
        counts = valid_matrix.sum(dim=1)
        loss = (element_loss * valid_matrix).sum(dim=1) / counts
        absolute_td = (prediction_matrix.detach() - target_matrix).abs()
        errors = absolute_td.masked_fill(~valid_matrix, -torch.inf).max(dim=1).values
        mean_q = (prediction_matrix.detach() * valid_matrix).sum(dim=1) / counts
        return loss, errors, mean_q

    # The methods below intentionally retain the original scalar algorithm as
    # a slow numerical oracle for batch<=8 equivalence tests.
    def _double_head_value(
        self,
        online_encoded: EncodedState,
        target_encoded: EncodedState,
        head: str,
        verb: int,
        selected: Mapping[str, int],
        masks: C.Masks | Mapping[str, Any] | None,
    ) -> Tensor:
        online_q = self.online.q_head(online_encoded, head, verb, selected)[0]
        legal = mask_for_head(masks, head, verb, selected, online_q.numel())
        choice = masked_argmax(online_q, legal)
        target_q = self.target.q_head(target_encoded, head, verb, selected)[0]
        return target_q[choice]

    def _next_state_value(
        self,
        online_encoded: EncodedState,
        target_encoded: EncodedState,
        masks: C.Masks | Mapping[str, Any] | None,
    ) -> Tensor:
        # Walking the complete online sequence is intentional: it uses exactly
        # the same maximisation routine as acting, including contextual masks.
        verb, heads, selected, _ = self.policy.select_indices(
            online_encoded, masks, epsilon=0.0
        )
        sequence = C.HEAD_SEQUENCE[C.VERBS[verb]]
        final_head = sequence[-1]
        prefix = dict(selected)
        final_choice = prefix.pop(final_head)
        return self.target.q_head(target_encoded, final_head, verb, prefix)[0, final_choice]

    def _transition_loss_reference(
        self,
        transition: C.Transition,
        online_state: EncodedState,
        target_state: EncodedState,
        online_next: EncodedState,
        target_next: EncodedState,
    ) -> tuple[Tensor, Tensor, Tensor]:
        verb = int(transition.verb)
        sequence = C.HEAD_SEQUENCE[C.VERBS[verb]]
        selected: dict[str, int] = {}
        predictions: list[Tensor] = []
        targets: list[Tensor] = []

        # Internal transition 1: choose verb, then bootstrap from the maximum
        # over the first argument. It advances zero simulated time.
        predictions.append(self.online.q_verbs(online_state)[0, verb])
        with torch.no_grad():
            targets.append(
                self._double_head_value(
                    online_state.detached(), target_state, sequence[0], verb, selected, transition.masks
                )
            )

        for depth, head in enumerate(sequence):
            choice = int(transition.action_heads[C.HEAD_INDEX[head]])
            if choice < 0:
                raise ValueError(f"transition is missing {head} for {C.VERBS[verb]}")
            q_values = self.online.q_head(online_state, head, verb, selected)[0]
            predictions.append(q_values[choice])
            selected[head] = choice
            with torch.no_grad():
                if depth + 1 < len(sequence):
                    next_head = sequence[depth + 1]
                    targets.append(
                        self._double_head_value(
                            online_state.detached(),
                            target_state,
                            next_head,
                            verb,
                            selected,
                            transition.masks,
                        )
                    )
                else:
                    if transition.done:
                        next_value = torch.zeros((), device=self.device)
                    else:
                        next_value = self._next_state_value(
                            online_next, target_next, transition.next_masks
                        )
                    targets.append(
                        torch.as_tensor(transition.reward, device=self.device)
                        + (0.0 if transition.done else self.discount(transition.tau_seconds) * next_value)
                    )

        prediction = torch.stack(predictions)
        target = torch.stack(targets).to(prediction.dtype)
        losses = F.smooth_l1_loss(prediction, target, reduction="none")
        td = prediction.detach() - target
        return losses.mean(), td.abs().max(), prediction.detach().mean()

    def _encode_batch(
        self, transitions: Sequence[C.Transition]
    ) -> tuple[EncodedState, EncodedState, EncodedState, EncodedState]:
        observations = [transition.observation for transition in transitions]
        next_observations = [transition.next_observation for transition in transitions]
        current_batch = self.online.prepare_batch(observations)
        next_batch = self.online.prepare_batch(next_observations)
        online_state = self.online.encode_prepared(current_batch)
        with torch.no_grad():
            online_next = self.online.encode_prepared(next_batch)
            target_state = self.target.encode_prepared(current_batch)
            target_next = self.target.encode_prepared(next_batch)
        return online_state, target_state, online_next, target_next

    @torch.no_grad()
    def compare_vectorized_reference(
        self, transitions: Sequence[C.Transition]
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Numerically compare the batched path with the old scalar oracle."""
        if not transitions or len(transitions) > 8:
            raise ValueError("reference comparison requires a batch of 1..8")
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=self.mixed_precision,
        ):
            encoded = self._encode_batch(transitions)
            vectorized = self._vectorized_terms(transitions, *encoded)
            reference_rows = [
                self._transition_loss_reference(
                    transition,
                    encoded[0].select(index),
                    encoded[1].select(index),
                    encoded[2].select(index),
                    encoded[3].select(index),
                )
                for index, transition in enumerate(transitions)
            ]
        reference = tuple(torch.stack(values) for values in zip(*reference_rows, strict=True))
        names = ("loss", "td_error", "mean_q")
        return {
            name: (
                vectorized[index].cpu().numpy(),
                reference[index].cpu().numpy(),
            )
            for index, name in enumerate(names)
        }

    def train_batch(
        self,
        batch: ReplayBatch | Sequence[C.Transition],
        weights: np.ndarray | None = None,
    ) -> LearnerStats:
        if isinstance(batch, ReplayBatch):
            transitions = batch.transitions
            weights_array = batch.weights
        else:
            transitions = list(batch)
            weights_array = (
                np.ones(len(transitions), dtype=np.float32)
                if weights is None
                else np.asarray(weights, dtype=np.float32)
            )
        if not transitions:
            raise ValueError("empty learner batch")
        if len(weights_array) != len(transitions):
            raise ValueError("one importance weight is required per transition")

        self.online.train()
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=self.mixed_precision,
        ):
            encoded = self._encode_batch(transitions)
            loss_vector, errors, q_values = self._vectorized_terms(transitions, *encoded)
            weight_tensor = torch.as_tensor(
                weights_array, device=self.device, dtype=loss_vector.dtype
            )
            loss = (loss_vector * weight_tensor).sum() / weight_tensor.sum().clamp_min(1e-8)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm_tensor = nn.utils.clip_grad_norm_(self.online.parameters(), self.grad_clip)
        grad_norm = float(grad_norm_tensor.detach().cpu().item())
        self.optimizer.step()
        self.updates += 1
        target_updated = self.updates % self.target_update_interval == 0
        if target_updated:
            self.target.load_state_dict(self.online.state_dict())

        td_errors = errors.detach().cpu().numpy().astype(np.float64)
        return LearnerStats(
            loss=float(loss.detach().cpu().item()),
            grad_norm=grad_norm,
            grad_clipped=grad_norm > self.grad_clip,
            mean_q=float(q_values.mean().cpu().item()),
            mean_abs_td_error=float(td_errors.mean()),
            updates=self.updates,
            target_updated=target_updated,
            td_errors=td_errors,
        )

    def update_from_replay(self, replay: PrioritizedReplay) -> LearnerStats:
        batch = replay.sample(self.batch_size)
        stats = self.train_batch(batch)
        replay.update_priorities(batch.indices, stats.td_errors)
        return stats

    def make_n_step_transition(self, transitions: Sequence[C.Transition]) -> C.Transition:
        """Combine up to ``n_step`` consecutive semantic transitions.

        Rewards use the product of the preceding per-step discounts. Since the
        discount is exponential in elapsed time, the eventual bootstrap factor
        is equivalently ``2**(-sum(tau)/H)``.
        """
        if not transitions:
            raise ValueError("n-step construction needs at least one transition")
        used = list(transitions[: self.n_step])
        reward = 0.0
        prefix_discount = 1.0
        total_tau = 0.0
        last = used[0]
        for transition in used:
            reward += prefix_discount * float(transition.reward)
            total_tau += float(transition.tau_seconds)
            prefix_discount *= float(self.discount(transition.tau_seconds))
            last = transition
            if transition.done:
                break
        first = used[0]
        return C.Transition(
            observation=first.observation,
            action_heads=np.asarray(first.action_heads, dtype=np.int64).copy(),
            verb=first.verb,
            reward=reward,
            tau_seconds=total_tau,
            next_observation=last.next_observation,
            done=last.done,
            success=first.success,
            failure_reason=first.failure_reason,
            masks=first.masks,
            next_masks=last.next_masks,
            is_demo=first.is_demo,
        )


Learner = DoubleDQNLearner
