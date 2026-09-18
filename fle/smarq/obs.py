"""Incremental tensor observation client for SM-ARQ.

Promoted from ``tests/benchmarks/benchmark_tensor_obs.py`` (PR #414).  The
17-channel grid layout and all 38 entity feature indices are intentionally
identical to that benchmark.  This module includes the tiered reconciler so it
never imports benchmark code at runtime.  SM-ARQ adds ``entity_ids`` containing
the stable Factorio unit number behind each nearest-entity row.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from fle.smarq import contract as C
from fle.smarq.vocab import StableVocab

GRID = C.GRID_SIZE
CELL = C.GRID_CELL_TILES
HALF = GRID * CELL // 2
RECENTER_DEADZONE = 24.0
N_CHANNELS = C.GRID_CHANNELS
ENTITY_CHANNEL = {
    "stone-furnace": 0,
    "transport-belt": 1,
    "burner-inserter": 2,
    "assembling-machine-1": 3,
    "iron-chest": 4,
}
OTHER_CHANNEL = 5
C_STATUS, C_SIN, C_COS, C_ENERGY, C_PROGRESS, C_INV = 6, 7, 8, 9, 10, 11
C_WATER, C_TREE, C_ROCK, C_ORE, C_NEST = 12, 13, 14, 15, 16
N_GLOBAL = C.N_GLOBALS

F_X, F_Y = 0, 1
F_SIN, F_COS = 2, 3
F_TYPE = 4
F_STATUS = 5
F_RECIPE = 6
F_ENERGY = 7
F_PROGRESS = 8
F_HEALTH = 9
F_TILE_W, F_TILE_H = 10, 11
F_DROP_DX, F_DROP_DY = 12, 13
F_PICK_DX, F_PICK_DY = 14, 15
F_ELEC_ID = 16
F_TEMP = 17
F_FLUID_ID, F_FLUID_AMT = 18, 19
F_INV_START = 20
N_INV_SLOTS = 8
F_INV_TOTAL = F_INV_START + 2 * N_INV_SLOTS
F_INV_DISTINCT = F_INV_TOTAL + 1
TABLE_ROWS = C.ENTITY_SLOTS
TABLE_FEATS = C.ENTITY_FEATURES
VIEW_K = C.ENTITY_SLOTS


class TieredClient:
    """Reconcile observation-diff records into complete client-side state."""

    def __init__(self) -> None:
        self.entities: dict[str, str] = {}
        self.water: dict[tuple[int, int], int] = {}
        self.ores: dict[str, tuple[str, int]] = {}
        self.trees: set[str] = set()
        self.obstacles: set[str] = set()
        self.nests: dict[str, tuple[str, float, float]] = {}
        self.tick = 0
        self.research: str | None = None
        self.research_pct = 0
        self.player_x = 0.0
        self.player_y = 0.0
        self.techs_finished: list[str] = []
        self.overflow = False
        self.build_distance = 10.0


class TensorClient(TieredClient):
    """Tiered client maintaining the PR #414 grid and object tensors."""

    def __init__(
        self,
        table_rows: int = TABLE_ROWS,
        view_k: int = VIEW_K,
        vocab: StableVocab | None = None,
    ) -> None:
        super().__init__()
        self.view_k = view_k
        self.vocab = vocab
        self.grid = np.zeros((N_CHANNELS, GRID, GRID), dtype=np.float32)
        self.global_vec = np.zeros(N_GLOBAL, dtype=np.float32)
        self.table = np.zeros((table_rows, TABLE_FEATS), dtype=np.float32)
        self.table_mask = np.zeros(table_rows, dtype=np.float32)
        self.view_units: list[str] = []
        self.entity_ids = np.zeros(view_k, dtype=np.int64)
        self._contrib: dict[str, tuple[int, int, dict[int, float]]] = {}
        self._ore_contrib: dict[str, tuple[int, int, float]] = {}
        self._water_contrib: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
        self._slot_of: dict[str, int] = {}
        self._unit_at: dict[int, str] = {}
        self._free_slots = list(range(table_rows - 1, -1, -1))
        self._vocab: dict[str, int] = {}
        self._stable_maps = vocab.index_maps() if vocab is not None else {}
        self.center_x = 0
        self.center_y = 0

    def _cell_of(self, x: float, y: float) -> tuple[int, int] | None:
        gx = int((x - self.center_x + HALF) // CELL)
        gy = int((y - self.center_y + HALF) // CELL)
        return (gx, gy) if 0 <= gx < GRID and 0 <= gy < GRID else None

    def _maybe_recenter(self) -> None:
        if (
            abs(self.player_x - self.center_x) <= RECENTER_DEADZONE
            and abs(self.player_y - self.center_y) <= RECENTER_DEADZONE
        ):
            return
        self.center_x = CELL * round(self.player_x / CELL)
        self.center_y = CELL * round(self.player_y / CELL)
        self._rebuild_grid()

    @staticmethod
    def _parse_row(row: str) -> dict[str, Any]:
        fields = row.split(",")
        parsed: dict[str, Any] = {
            "name": fields[0],
            "x": float(fields[1]),
            "y": float(fields[2]),
            "direction": int(fields[3]),
            "status": int(fields[4]),
            "energy": 0.0,
            "progress": 0.0,
            "recipe": None,
            "health": 0.0,
            "tile_w": 0.0,
            "tile_h": 0.0,
            "drop": None,
            "pickup": None,
            "elec_id": 0.0,
            "temp": 0.0,
            "items": [],
            "fluids": [],
        }
        for field in fields[5:]:
            tag = field[:1]
            if tag == "E":
                parsed["energy"] = float(field[1:])
            elif tag == "P":
                parsed["progress"] = float(field[1:])
            elif tag == "R":
                parsed["recipe"] = field[1:]
            elif tag == "H":
                parsed["health"] = float(field[1:])
            elif tag == "W":
                width, height = field[1:].split(":")
                parsed["tile_w"], parsed["tile_h"] = float(width), float(height)
            elif tag == "D":
                x, y = field[1:].split(":")
                parsed["drop"] = (float(x), float(y))
            elif tag == "K":
                x, y = field[1:].split(":")
                parsed["pickup"] = (float(x), float(y))
            elif tag == "N":
                parsed["elec_id"] = float(field[1:])
            elif tag == "T":
                parsed["temp"] = float(field[1:])
            elif tag == "I":
                slot, rest = field[1:].split(".", 1)
                item, count = rest.rsplit(":", 1)
                parsed["items"].append((int(slot), item, float(count)))
            elif tag == "F":
                fluid, amount = field[1:].rsplit(":", 1)
                parsed["fluids"].append((fluid, float(amount)))
        return parsed

    def _vid(self, kind: str, name: str) -> int:
        field = {
            "type": "entity_types",
            "recipe": "recipes",
            "item": "items",
            "fluid": "fluids",
        }.get(kind)
        if field is not None and field in self._stable_maps:
            return self._stable_maps[field].get(name, 0)
        key = kind + ":" + name
        value = self._vocab.get(key)
        if value is None:
            value = len(self._vocab) + 1
            self._vocab[key] = value
        return value

    def _entity_contrib(self, row: str) -> tuple[int, int, dict[int, float]] | None:
        parsed = self._parse_row(row)
        cell = self._cell_of(parsed["x"], parsed["y"])
        if cell is None:
            return None
        gx, gy = cell
        angle = parsed["direction"] / 16.0 * 2.0 * math.pi
        total_items = sum(count for _, _, count in parsed["items"])
        return gy, gx, {
            ENTITY_CHANNEL.get(parsed["name"], OTHER_CHANNEL): 1.0,
            C_STATUS: parsed["status"] / 10.0,
            C_SIN: math.sin(angle),
            C_COS: math.cos(angle),
            C_ENERGY: parsed["energy"] / 1e6,
            C_PROGRESS: parsed["progress"] / 100.0,
            C_INV: math.log2(1.0 + total_items),
        }

    def _table_upsert(self, key: str, row: str) -> None:
        parsed = self._parse_row(row)
        slot = self._slot_of.get(key)
        if slot is None:
            if not self._free_slots:
                self.overflow = True
                return
            slot = self._free_slots.pop()
            self._slot_of[key] = slot
            self._unit_at[slot] = key
        angle = parsed["direction"] / 16.0 * 2.0 * math.pi
        tensor_row = self.table[slot]
        tensor_row[:] = 0.0
        tensor_row[F_X], tensor_row[F_Y] = parsed["x"], parsed["y"]
        tensor_row[F_SIN], tensor_row[F_COS] = math.sin(angle), math.cos(angle)
        tensor_row[F_TYPE] = self._vid("type", parsed["name"])
        tensor_row[F_STATUS] = parsed["status"]
        if parsed["recipe"]:
            tensor_row[F_RECIPE] = self._vid("recipe", parsed["recipe"])
        tensor_row[F_ENERGY] = parsed["energy"]
        tensor_row[F_PROGRESS] = parsed["progress"]
        tensor_row[F_HEALTH] = parsed["health"]
        tensor_row[F_TILE_W], tensor_row[F_TILE_H] = parsed["tile_w"], parsed["tile_h"]
        if parsed["drop"]:
            tensor_row[F_DROP_DX] = parsed["drop"][0] - parsed["x"]
            tensor_row[F_DROP_DY] = parsed["drop"][1] - parsed["y"]
        if parsed["pickup"]:
            tensor_row[F_PICK_DX] = parsed["pickup"][0] - parsed["x"]
            tensor_row[F_PICK_DY] = parsed["pickup"][1] - parsed["y"]
        tensor_row[F_ELEC_ID] = parsed["elec_id"]
        tensor_row[F_TEMP] = parsed["temp"]
        if parsed["fluids"]:
            tensor_row[F_FLUID_ID] = self._vid("fluid", parsed["fluids"][0][0])
            tensor_row[F_FLUID_AMT] = parsed["fluids"][0][1]
        merged: dict[str, float] = {}
        for _, item, count in parsed["items"]:
            merged[item] = merged.get(item, 0.0) + count
        ranked = sorted(merged.items(), key=lambda pair: -pair[1])
        for index, (item, count) in enumerate(ranked[:N_INV_SLOTS]):
            tensor_row[F_INV_START + 2 * index] = self._vid("item", item)
            tensor_row[F_INV_START + 2 * index + 1] = count
        tensor_row[F_INV_TOTAL] = sum(merged.values())
        tensor_row[F_INV_DISTINCT] = len(merged)
        self.table_mask[slot] = 1.0

    def _table_remove(self, key: str) -> None:
        slot = self._slot_of.pop(key, None)
        if slot is not None:
            self._unit_at.pop(slot, None)
            self.table[slot] = 0.0
            self.table_mask[slot] = 0.0
            self._free_slots.append(slot)

    def _apply_contrib(self, contrib: tuple[int, int, dict[int, float]], sign: float) -> None:
        gy, gx, values = contrib
        for channel, value in values.items():
            self.grid[channel, gy, gx] += sign * value

    def apply_entity(self, response: str) -> int:
        if not response:
            return 0
        records = response.split(";")
        for record in records:
            tag = record[:1]
            if tag == "u":
                key, row = record[1:].split(",", 1)
                old = self._contrib.pop(key, None)
                if old is not None:
                    self._apply_contrib(old, -1.0)
                contrib = self._entity_contrib(row)
                if contrib is not None:
                    self._contrib[key] = contrib
                    self._apply_contrib(contrib, 1.0)
                self._table_upsert(key, row)
                self.entities[key] = row
            elif tag == "r":
                key = record[1:]
                old = self._contrib.pop(key, None)
                if old is not None:
                    self._apply_contrib(old, -1.0)
                self._table_remove(key)
                self.entities.pop(key, None)
            elif tag == "h":
                parts = record[1:].split(":")
                self.tick = int(parts[0])
                self.research = None if parts[1] == "-" else parts[1]
                self.research_pct = int(parts[2])
                if len(parts) >= 5:
                    self.player_x = float(parts[3])
                    self.player_y = float(parts[4])
                    self._maybe_recenter()
            elif tag == "q":
                self.techs_finished.append(record[1:])
            elif tag == "!":
                self.overflow = True
        self._update_globals()
        return len(records)

    def _apply_water_chunk(self, cx: int, cy: int, hexmask: str) -> None:
        old = self._water_contrib.pop((cx, cy), None)
        if old is not None:
            for gy, gx, count in old:
                self.grid[C_WATER, gy, gx] -= count
        if not hexmask:
            return
        rows = np.array(
            [int(hexmask[index * 8 : (index + 1) * 8], 16) for index in range(32)],
            dtype=np.uint32,
        )
        tile_mask = (rows[:, None] >> np.arange(32, dtype=np.uint32)[None, :]) & 1
        dy, dx = np.nonzero(tile_mask)
        wx = cx * 32 + dx - self.center_x + HALF
        wy = cy * 32 + dy - self.center_y + HALF
        keep = (wx >= 0) & (wx < GRID * CELL) & (wy >= 0) & (wy < GRID * CELL)
        if not keep.any():
            return
        gx = wx[keep] // CELL
        gy = wy[keep] // CELL
        cells: dict[tuple[int, int], int] = {}
        for row, column in zip(gy, gx, strict=True):
            key = (int(row), int(column))
            cells[key] = cells.get(key, 0) + 1
        contrib = [(row, column, count) for (row, column), count in cells.items()]
        self._water_contrib[(cx, cy)] = contrib
        for row, column, count in contrib:
            self.grid[C_WATER, row, column] += count

    def _point_delta(self, key: str, channel: int, new_value: float | None) -> None:
        old = self._ore_contrib.pop(key, None)
        if old is not None:
            gy, gx, value = old
            self.grid[channel, gy, gx] -= value
        if new_value is None:
            return
        x, y = key.split(":")
        cell = self._cell_of(float(x), float(y))
        if cell is None:
            return
        gx, gy = cell
        self._ore_contrib[key] = (gy, gx, new_value)
        self.grid[channel, gy, gx] += new_value

    def apply_terrain(self, response: str) -> int:
        if not response:
            return 0
        records = response.split(";")
        for record in records:
            tag = record[:1]
            if tag == "c":
                cx_text, cy_text, hexmask = record[1:].split(":", 2)
                cx, cy = int(cx_text), int(cy_text)
                self.water[(cx, cy)] = int(hexmask, 16) if hexmask else 0
                self._apply_water_chunk(cx, cy, hexmask)
            elif tag == "o":
                name, x, y, bucket = record[1:].split(":")
                key = f"{x}:{y}"
                self.ores[key] = (name, int(bucket))
                self._point_delta(key, C_ORE, float(bucket))
            elif tag == "d":
                key = record[1:]
                self.ores.pop(key, None)
                self._point_delta(key, C_ORE, None)
            elif tag == "t":
                key = record[1:]
                if key not in self.trees:
                    self.trees.add(key)
                    cell = self._cell_of(*map(float, key.split(":")))
                    if cell:
                        self.grid[C_TREE, cell[1], cell[0]] += 1
            elif tag == "x":
                key = record[1:]
                cell = self._cell_of(*map(float, key.split(":")))
                if key in self.trees:
                    self.trees.discard(key)
                    if cell:
                        self.grid[C_TREE, cell[1], cell[0]] -= 1
                elif key in self.obstacles:
                    self.obstacles.discard(key)
                    if cell:
                        self.grid[C_ROCK, cell[1], cell[0]] -= 1
            elif tag == "k":
                key = record[1:]
                if key not in self.obstacles:
                    self.obstacles.add(key)
                    cell = self._cell_of(*map(float, key.split(":")))
                    if cell:
                        self.grid[C_ROCK, cell[1], cell[0]] += 1
            elif tag == "n":
                key, name, x, y = record[1:].split(",")
                self.nests[key] = (name, float(x), float(y))
                cell = self._cell_of(float(x), float(y))
                if cell:
                    self.grid[C_NEST, cell[1], cell[0]] += 1
            elif tag == "m":
                key = record[1:]
                nest = self.nests.pop(key, None)
                if nest:
                    cell = self._cell_of(nest[1], nest[2])
                    if cell:
                        self.grid[C_NEST, cell[1], cell[0]] -= 1
            elif tag == "!":
                self.overflow = True
        self._update_globals()
        return len(records)

    def _update_globals(self) -> None:
        vector = self.global_vec
        vector[0] = self.tick / 1e6
        vector[1] = self.research_pct / 100.0
        vector[2] = len(self.entities) / 1e4
        vector[3] = len(self.ores) / 1e4
        vector[4] = len(self.trees) / 1e5
        vector[5] = len(self.nests) / 100.0
        vector[6] = len(self.techs_finished) / 100.0
        vector[7] = 1.0 if self.overflow else 0.0
        vector[8] = self.player_x
        vector[9] = self.player_y

    def observation(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return grid, globals, nearest rows, live mask, and stable unit IDs."""
        view = np.zeros((self.view_k, TABLE_FEATS), dtype=np.float32)
        view_mask = np.zeros(self.view_k, dtype=bool)
        entity_ids = np.zeros(self.view_k, dtype=np.int64)
        live_idx = np.nonzero(self.table_mask)[0]
        if live_idx.size:
            dx = self.table[live_idx, F_X] - self.player_x
            dy = self.table[live_idx, F_Y] - self.player_y
            distance2 = dx * dx + dy * dy
            if live_idx.size > self.view_k:
                selected = np.argpartition(distance2, self.view_k - 1)[: self.view_k]
                selected = selected[np.argsort(distance2[selected])]
            else:
                selected = np.argsort(distance2)
            chosen = live_idx[selected]
            count = chosen.size
            view[:count] = self.table[chosen]
            view[:count, F_X] -= self.player_x
            view[:count, F_Y] -= self.player_y
            view_mask[:count] = True
            self.view_units = [self._unit_at[int(slot)] for slot in chosen]
            entity_ids[:count] = np.asarray(self.view_units, dtype=np.int64)
        else:
            self.view_units = []
        self.entity_ids = entity_ids
        return self.grid, self.global_vec, view, view_mask, entity_ids

    def _rebuild_grid(self) -> None:
        self.grid[:] = 0.0
        self._contrib.clear()
        self._ore_contrib.clear()
        self._water_contrib.clear()
        for key, row in self.entities.items():
            contrib = self._entity_contrib(row)
            if contrib is not None:
                self._contrib[key] = contrib
                self._apply_contrib(contrib, 1.0)
        for key, (_, bucket) in self.ores.items():
            self._point_delta(key, C_ORE, float(bucket))
        for key in self.trees:
            cell = self._cell_of(*map(float, key.split(":")))
            if cell:
                self.grid[C_TREE, cell[1], cell[0]] += 1
        for key in self.obstacles:
            cell = self._cell_of(*map(float, key.split(":")))
            if cell:
                self.grid[C_ROCK, cell[1], cell[0]] += 1
        for _, x, y in self.nests.values():
            cell = self._cell_of(x, y)
            if cell:
                self.grid[C_NEST, cell[1], cell[0]] += 1
        for (cx, cy), mask in self.water.items():
            self._apply_water_chunk(cx, cy, format(mask, "0256x") if mask else "")

    def rebuild(self) -> None:
        self._rebuild_grid()
        self.table[:] = 0.0
        self.table_mask[:] = 0.0
        self._slot_of.clear()
        self._unit_at.clear()
        self._free_slots = list(range(self.table.shape[0] - 1, -1, -1))
        for key, row in sorted(self.entities.items()):
            self._table_upsert(key, row)

    def full_sync(self, rcon: Any) -> None:
        """Replace all client state using the scenario's authoritative snapshots."""
        self.__init__(self.table.shape[0], self.view_k, self.vocab)
        entity_response = rcon.send_command("/sc obs_diff_full_sync()") or ""
        terrain_response = rcon.send_command("/sc obs_terrain_full_sync()") or ""
        self.apply_entity(entity_response)
        self.apply_terrain(terrain_response)

    def drain(self, rcon: Any) -> None:
        response = rcon.send_command("/sc obs_all_drain()") or ""
        entity_part, _, terrain_part = response.partition("~")
        self.apply_entity(entity_part)
        self.apply_terrain(terrain_part)
        if self.overflow:
            self.full_sync(rcon)
