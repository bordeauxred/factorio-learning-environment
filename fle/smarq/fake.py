"""A deterministic stand-in for the live environment.

It exists so that the network, the learner, the replay and the training loop can
be built and tested without a Factorio server, and so that the SMDP arithmetic
can be checked against hand-computable numbers.  It is faithful to
``fle.smarq.contract`` in shapes, head order, mask discipline and timing
semantics, and faithful to nothing else: the physics is a toy.

Toy world: a furnace placed on a tile that holds ore starts producing once it
has been fed coal.  Producing furnaces add automated score while simulated time
passes; hand mining adds production score only.  Actions have different
simulated durations, which is the property the SMDP tests need.
"""

from __future__ import annotations

import numpy as np

from fle.smarq import contract as C
from fle.smarq.contract import (
    Action,
    Masks,
    Observation,
    StepResult,
)

PROTOTYPES = ("<none>", "stone-furnace", "burner-mining-drill", "transport-belt", "wooden-chest")
ITEMS = ("<none>", "coal", "iron-ore", "iron-plate", "stone", "stone-furnace", "burner-mining-drill")
RECIPES = ("<none>", "stone-furnace", "burner-mining-drill", "iron-gear-wheel", "transport-belt")
TECHNOLOGIES = ("<none>", "automation", "logistics")
ENTITY_TYPES = ("<none>", "furnace", "mining-drill", "transport-belt", "container")

# Simulated cost of each verb, in ticks, before navigation is added.
BASE_TICKS = {
    "MOVE_TO": 0,
    "MINE": 120,
    "CRAFT": 60,
    "PLACE": 6,
    "PICKUP": 6,
    "ROTATE": 3,
    "INSERT": 6,
    "EXTRACT": 6,
    "SET_RECIPE": 3,
    "RESEARCH": 3,
    "FAST_FORWARD": 0,
}
WALK_TICKS_PER_TILE = 8  # what navigation costs in this toy world


class FakeVocab:
    prototypes = PROTOTYPES
    items = ITEMS
    recipes = RECIPES
    technologies = TECHNOLOGIES
    entity_types = ENTITY_TYPES

    def head_sizes(self, raster_tiles: int) -> dict[str, int]:
        return C.VocabProtocol.head_sizes(self, raster_tiles)  # type: ignore[arg-type]


class FakeSemanticEnv:
    """Toy SMDP with the contract's shapes and timing semantics."""

    def __init__(
        self,
        raster_tiles: int = 16,
        entity_slots: int = 64,
        tick_budget: int = 18_000,
        decision_cap: int = 200,
        reward_mode: str = "automated",
        seed: int = 0,
        coarse_move: bool = False,
    ) -> None:
        if reward_mode not in C.REWARD_MODES:
            raise ValueError(reward_mode)
        self.raster_tiles = raster_tiles
        self.entity_slots = entity_slots
        self.tick_budget = tick_budget
        self.decision_cap = decision_cap
        self.reward_mode = reward_mode
        self.coarse_move = bool(coarse_move)
        self._rng = np.random.default_rng(seed)
        self._vocab = FakeVocab()
        self.reset(seed)

    # -- protocol ---------------------------------------------------------
    @property
    def vocab(self) -> FakeVocab:
        return self._vocab

    def reset(self, seed: int | None = None) -> tuple[Observation, Masks]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.tick = 0
        self.decisions = 0
        self.player = (0, 0)
        self.production_score = 0.0
        self.automated_score = 0.0
        self.inventory = {"coal": 20, "stone-furnace": 2, "iron-ore": 10}
        # Entities: list of dicts with a stable unit number.
        self.entities: list[dict] = []
        self._next_unit = 1
        # A patch of ore tiles the toy furnace needs to sit on.
        half = self.raster_tiles // 2
        self.ore_tiles = {(x, y) for x in range(2, 6) for y in range(-2, 2)}
        self.water_tiles = {(x, y) for x in range(-half, -half + 3) for y in range(-2, 3)}
        return self.observe(), self.masks()

    def close(self) -> None:  # pragma: no cover - nothing to release
        pass

    # -- observation ------------------------------------------------------
    def observe(self) -> Observation:
        grid = np.zeros((C.GRID_CHANNELS, C.GRID_SIZE, C.GRID_SIZE), dtype=np.float32)
        ev = np.zeros((self.entity_slots, C.ENTITY_FEATURES), dtype=np.float32)
        mask = np.zeros(self.entity_slots, dtype=bool)
        ids = np.zeros(self.entity_slots, dtype=np.int64)
        for slot, ent in enumerate(self.entities[: self.entity_slots]):
            ev[slot, C.HEAD_INDEX.get("position", 0)] = 0.0
            ev[slot, 0] = ent["x"] - self.player[0]
            ev[slot, 1] = ent["y"] - self.player[1]
            ev[slot, 4] = ENTITY_TYPES.index(ent["type"])
            ev[slot, 5] = float(ent["working"])
            ev[slot, 20] = np.log1p(ent["fuel"])
            mask[slot] = True
            ids[slot] = ent["unit"]
            gx = int(np.clip(C.GRID_SIZE // 2 + (ent["x"] - self.player[0]) // C.GRID_CELL_TILES, 0, C.GRID_SIZE - 1))
            gy = int(np.clip(C.GRID_SIZE // 2 + (ent["y"] - self.player[1]) // C.GRID_CELL_TILES, 0, C.GRID_SIZE - 1))
            grid[0, gy, gx] += 1.0
        globals_ = np.zeros(C.N_GLOBALS, dtype=np.float32)
        globals_[0] = self.player[0] / 100.0
        globals_[1] = self.player[1] / 100.0
        globals_[2] = np.log1p(self.inventory.get("coal", 0))
        globals_[3] = np.log1p(self.inventory.get("stone-furnace", 0))
        globals_[4] = np.log1p(self.inventory.get("iron-ore", 0))
        globals_[5] = self.tick / max(1, self.tick_budget)
        globals_[6] = np.log1p(self.automated_score)
        globals_[7] = np.log1p(self.production_score)
        origin = (self.player[0] - self.raster_tiles // 2, self.player[1] - self.raster_tiles // 2)
        raster = np.zeros((C.RASTER_CHANNELS, self.raster_tiles, self.raster_tiles), dtype=np.float32)
        for ent in self.entities:
            p = C.tile_to_position((ent["x"], ent["y"]), origin, self.raster_tiles)
            if p is not None:
                raster[0, p // self.raster_tiles, p % self.raster_tiles] = 1.0
                raster[4, p // self.raster_tiles, p % self.raster_tiles] = 1.0
        for tile in self.water_tiles:
            p = C.tile_to_position(tile, origin, self.raster_tiles)
            if p is not None:
                raster[1, p // self.raster_tiles, p % self.raster_tiles] = 1.0
        for tile in self.ore_tiles:
            p = C.tile_to_position(tile, origin, self.raster_tiles)
            if p is not None:
                raster[2, p // self.raster_tiles, p % self.raster_tiles] = 1.0
        p = C.tile_to_position(self.player, origin, self.raster_tiles)
        if p is not None:
            raster[6, p // self.raster_tiles, p % self.raster_tiles] = 1.0
        raster[7] = 1.0  # toy world: everything in the window is buildable-in-reach
        return Observation(
            grid=grid,
            entity_view=ev,
            entity_mask=mask,
            entity_ids=ids,
            globals=globals_,
            raster=raster,
            raster_origin=origin,
            raster_tiles=self.raster_tiles,
            player_tile=self.player,
            tick=self.tick,
            production_score=self.production_score,
            automated_production_score=self.automated_score,
        )

    def masks(self) -> Masks:
        verb = np.ones(C.N_VERBS, dtype=bool)
        prototype = np.zeros(len(PROTOTYPES), dtype=bool)
        prototype[1:] = True  # <none> is never selectable
        item = np.zeros(len(ITEMS), dtype=bool)
        item[1:] = True
        craft = np.zeros(len(RECIPES), dtype=bool)
        craft[1:] = True
        tech = np.zeros(len(TECHNOLOGIES), dtype=bool)
        tech[1:] = True
        live = np.zeros(self.entity_slots, dtype=bool)
        live[: len(self.entities)] = True
        ent = np.zeros((C.N_VERBS, self.entity_slots), dtype=bool)
        for name in ("PICKUP", "ROTATE", "INSERT", "EXTRACT", "SET_RECIPE"):
            ent[C.VERB_INDEX[name]] = live
        return Masks(
            verb=verb,
            prototype=prototype,
            item=item,
            craft_recipe=craft,
            technology=tech,
            entity=ent,
            recipe_for_entity=lambda slot: craft,
        )

    # -- dynamics ---------------------------------------------------------
    def _walk_ticks(self, tile: tuple[int, int]) -> int:
        d = abs(tile[0] - self.player[0]) + abs(tile[1] - self.player[1])
        return int(d * WALK_TICKS_PER_TILE)

    def _produce(self, ticks: int) -> float:
        """Automated score accrued by working furnaces over ``ticks``."""
        gain = 0.0
        for ent in self.entities:
            if ent["type"] != "furnace" or not ent["working"]:
                continue
            burn = min(ent["fuel"], ticks / 60.0)
            ent["fuel"] -= burn
            gain += burn * 2.0  # two points of automated score per simulated second
            if ent["fuel"] <= 0:
                ent["working"] = False
        return gain

    def step(self, action: Action) -> tuple[StepResult, Masks]:
        import time

        wall0 = time.perf_counter()
        before_p, before_a = self.production_score, self.automated_score
        verb = action.verb
        ticks = BASE_TICKS[verb]
        success, reason = True, "ok"

        if verb == "FAST_FORWARD":
            ticks = action.duration_seconds * C.TICKS_PER_SECOND
        elif verb in ("MOVE_TO", "MINE", "PLACE"):
            tile = action.tile
            ticks += self._walk_ticks(tile)
            if verb == "MOVE_TO":
                if tile in self.water_tiles:
                    success, reason = False, "unreachable"
                else:
                    self.player = tile
            elif verb == "MINE":
                if tile in self.ore_tiles:
                    n = 4 if action.quantity == "ALL" else int(action.quantity)
                    self.inventory["iron-ore"] = self.inventory.get("iron-ore", 0) + n
                    self.production_score += 0.5 * n  # hand mining: production only
                else:
                    success, reason = False, "invalid_argument"
            else:  # PLACE
                proto = action.prototype
                if self.inventory.get(proto, 0) <= 0:
                    success, reason = False, "insufficient_inventory"
                elif tile in self.water_tiles or any((e["x"], e["y"]) == tile for e in self.entities):
                    success, reason = False, "blocked"
                else:
                    self.inventory[proto] -= 1
                    self.entities.append(
                        {
                            "unit": self._next_unit,
                            "x": tile[0],
                            "y": tile[1],
                            "type": "furnace" if proto == "stone-furnace" else "container",
                            "on_ore": tile in self.ore_tiles,
                            "fuel": 0.0,
                            "working": False,
                            "proto": proto,
                        }
                    )
                    self._next_unit += 1
        elif verb in ("INSERT", "EXTRACT", "PICKUP", "ROTATE", "SET_RECIPE"):
            slot = action.entity_slot
            if slot is None or slot >= len(self.entities):
                success, reason = False, "no_such_entity"
            else:
                ent = self.entities[slot]
                ticks += self._walk_ticks((ent["x"], ent["y"]))
                if verb == "INSERT":
                    n = 8 if action.quantity == "ALL" else int(action.quantity)
                    if self.inventory.get(action.item, 0) < n:
                        success, reason = False, "insufficient_inventory"
                    elif action.item != "coal":
                        success, reason = False, "invalid_argument"
                    else:
                        self.inventory["coal"] -= n
                        ent["fuel"] += n * 4.0
                        ent["working"] = ent["type"] == "furnace" and ent["on_ore"]
                elif verb == "PICKUP":
                    self.inventory[ent["proto"]] = self.inventory.get(ent["proto"], 0) + 1
                    self.entities.pop(slot)
        elif verb == "CRAFT":
            n = 1 if action.quantity == "ALL" else int(action.quantity)
            self.inventory[action.recipe] = self.inventory.get(action.recipe, 0) + n
            self.production_score += 1.0 * n  # hand crafting: production only

        self.automated_score += self._produce(ticks)
        self.tick += ticks
        self.decisions += 1
        after_p, after_a = self.production_score, self.automated_score
        reward = (after_a - before_a) if self.reward_mode == "automated" else (after_p - before_p)
        out_of_time = self.tick >= self.tick_budget
        done = out_of_time or self.decisions >= self.decision_cap
        end_reason = None
        if done:
            end_reason = "tick_budget" if out_of_time else "decision_cap"
        result = StepResult(
            observation=self.observe(),
            reward=float(reward),
            done=done,
            duration_ticks=int(ticks),
            duration_game_seconds=ticks / C.TICKS_PER_SECOND,
            wall_seconds=time.perf_counter() - wall0,
            success=success,
            failure_reason=reason,
            production_score=after_p,
            automated_production_score=after_a,
            delta_production_score=after_p - before_p,
            delta_automated_production_score=after_a - before_a,
            episode_ticks=self.tick,
            episode_decisions=self.decisions,
            end_reason=end_reason,
        )
        return result, self.masks()
