"""Thin client for the open-world tiered observation wire protocol."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any

from fle.commons.observation.buildability import BuildabilityCache
from fle.commons.observation.minimap import MinimapCache


@dataclass(frozen=True)
class EntityRow:
    unit: int
    name: str
    x: float
    y: float
    direction: int
    status: int
    energy: int | None = None
    progress: int | None = None
    recipe: str | None = None
    health: int | None = None
    tile_width: int | None = None
    tile_height: int | None = None
    drop: tuple[float, float] | None = None
    pickup: tuple[float, float] | None = None
    network_id: int | None = None
    temperature: int | None = None
    inventories: dict[int, dict[str, int]] = field(default_factory=dict)
    fluids: dict[str, float] = field(default_factory=dict)

    @property
    def items(self) -> dict[str, int]:
        result: dict[str, int] = {}
        for inventory in self.inventories.values():
            for name, count in inventory.items():
                result[name] = result.get(name, 0) + count
        return result


def parse_entity_row(record: str) -> EntityRow:
    """Parse one ``u<unit>,<rich_row>`` wire record."""
    if not record.startswith("u"):
        raise ValueError(f"Entity row must start with 'u': {record[:40]}")
    fields = record[1:].split(",")
    if len(fields) < 6:
        raise ValueError(f"Entity row has too few fields: {record[:80]}")

    unit = int(fields[0])
    name = fields[1]
    x, y = float(fields[2]), float(fields[3])
    direction, status = int(fields[4]), int(fields[5])
    values: dict[str, Any] = {}
    inventories: dict[int, dict[str, int]] = {}
    fluids: dict[str, float] = {}

    for extra in fields[6:]:
        if not extra:
            continue
        tag, payload = extra[0], extra[1:]
        if tag == "E":
            values["energy"] = int(payload)
        elif tag == "P":
            values["progress"] = int(payload)
        elif tag == "R":
            values["recipe"] = payload
        elif tag == "H":
            values["health"] = int(payload)
        elif tag == "W":
            width, height = payload.split(":", 1)
            values["tile_width"] = int(width)
            values["tile_height"] = int(height)
        elif tag in {"D", "K"}:
            px, py = payload.split(":", 1)
            values["drop" if tag == "D" else "pickup"] = (float(px), float(py))
        elif tag == "N":
            values["network_id"] = int(payload)
        elif tag == "T":
            values["temperature"] = int(payload)
        elif tag == "I":
            inventory_part, count_part = payload.rsplit(":", 1)
            index_part, item = inventory_part.split(".", 1)
            inventories.setdefault(int(index_part), {})[item] = int(count_part)
        elif tag == "F":
            fluid, amount = payload.rsplit(":", 1)
            fluids[fluid] = float(amount)
        else:
            raise ValueError(f"Unknown entity-row field {extra!r}")

    return EntityRow(
        unit=unit,
        name=name,
        x=x,
        y=y,
        direction=direction,
        status=status,
        inventories=inventories,
        fluids=fluids,
        **values,
    )


@dataclass(frozen=True)
class OrePatch:
    id: str
    name: str
    n_tiles: int
    bbox: tuple[int, int, int, int]
    _tiles: tuple[tuple[int, int], ...] = field(repr=False)

    def nearest_tile(self, px: float, py: float) -> tuple[int, int]:
        return min(
            self._tiles,
            key=lambda tile: ((tile[0] - px) ** 2 + (tile[1] - py) ** 2, tile),
        )


class WorldClient:
    """Reconcile entity and terrain records into a reusable world cache."""

    def __init__(self, rcon_client: Any, namespace: Any):
        self.rcon = rcon_client
        self.namespace = namespace
        self._init_state()

    def configure_v0_caches(
        self,
        rcon: Any,
        *,
        build_size: int = 128,
        check_budget: int = 256,
        minimap_chunk_budget: int = 4,
    ) -> tuple[Any, Any]:
        """Enable the V0 caches consumed from this client's combined drain."""
        build_response = self.buildability.configure(
            rcon, size=build_size, check_budget=check_budget
        )
        minimap_response = self.minimap.configure(
            rcon, chunk_budget=minimap_chunk_budget, visibility="generated"
        )
        return build_response, minimap_response

    def _init_state(self) -> None:
        if not hasattr(self, "buildability"):
            self.buildability = BuildabilityCache()
        if not hasattr(self, "minimap"):
            self.minimap = MinimapCache()
        self.entities: dict[int, EntityRow] = {}
        self.water: dict[tuple[int, int], int] = {}
        self.ores: dict[tuple[int, int], tuple[str, int]] = {}
        self.trees: set[tuple[int, int]] = set()
        self.obstacles: set[tuple[int, int]] = set()
        self.nests: dict[int, tuple[str, float, float]] = {}
        self.tick = 0
        self.research: str | None = None
        self.research_pct = 0
        self.player_x = 0.0
        self.player_y = 0.0
        self.techs_finished: list[str] = []
        self.overflow = False
        self._terrain_synced = False

    def apply_entity(self, response: str) -> int:
        if not response:
            return 0
        records = response.split(";")
        for record in records:
            tag = record[:1]
            if tag == "u":
                row = parse_entity_row(record)
                if row.name == "character":
                    self.entities.pop(row.unit, None)
                else:
                    self.entities[row.unit] = row
            elif tag == "r":
                self.entities.pop(int(record[1:]), None)
            elif tag == "h":
                parts = record[1:].split(":")
                self.tick = int(parts[0])
                self.research = None if parts[1] == "-" else parts[1]
                self.research_pct = int(parts[2])
                if len(parts) >= 5:
                    self.player_x = float(parts[3])
                    self.player_y = float(parts[4])
            elif tag == "q":
                self.techs_finished.append(record[1:])
            elif tag == "!":
                self.overflow = True
        return len(records)

    def apply_terrain(self, response: str) -> int:
        if not response:
            return 0
        records = response.split(";")
        for record in records:
            if self.minimap.apply_record(record):
                continue
            if self.buildability.apply_record(record):
                continue
            tag = record[:1]
            if tag == "c":
                cx, cy, hexmask = record[1:].split(":", 2)
                self.water[(int(cx), int(cy))] = int(hexmask, 16) if hexmask else 0
            elif tag == "o":
                name, x, y, amount = record[1:].split(":")
                self.ores[(int(x), int(y))] = (name, int(amount))
            elif tag == "d":
                x, y = record[1:].split(":")
                self.ores.pop((int(x), int(y)), None)
            elif tag == "t":
                x, y = record[1:].split(":")
                self.trees.add((int(x), int(y)))
            elif tag == "x":
                x, y = record[1:].split(":")
                pos = (int(x), int(y))
                self.trees.discard(pos)
                self.obstacles.discard(pos)
            elif tag == "k":
                x, y = record[1:].split(":")
                self.obstacles.add((int(x), int(y)))
            elif tag == "n":
                unit, name, x, y = record[1:].split(",")
                self.nests[int(unit)] = (name, float(x), float(y))
            elif tag == "m":
                self.nests.pop(int(record[1:]), None)
            elif tag == "!":
                self.overflow = True
        return len(records)

    def obs_terrain_full_sync(self, force: bool = False) -> int:
        """Full terrain snapshot (about 23 s on the open-world map).

        Call once per process; pass ``force=True`` only to recover from a
        ``!overflow`` marker, when the incremental terrain buffer was truncated.
        """
        if self._terrain_synced and not force:
            raise RuntimeError(
                "obs_terrain_full_sync may only be called once per process"
            )
        response = self.rcon.send_command("/sc obs_terrain_full_sync()") or ""
        self.water.clear()
        self.ores.clear()
        self.trees.clear()
        self.obstacles.clear()
        self.nests.clear()
        self.overflow = False
        self._terrain_synced = True
        return self.apply_terrain(response)

    def obs_diff_full_sync(self) -> int:
        response = self.rcon.send_command("/sc obs_diff_full_sync()") or ""
        self.entities.clear()
        self.techs_finished.clear()
        self.overflow = False
        return self.apply_entity(response)

    def obs_all_drain(self) -> int:
        response = self.rcon.send_command("/sc obs_all_drain()") or ""
        entity_part, separator, terrain_part = response.partition("~")
        count = self.apply_entity(entity_part)
        if separator:
            count += self.apply_terrain(terrain_part)
        if self.overflow:
            self.overflow_events = getattr(self, "overflow_events", 0) + 1
            count += self.obs_diff_full_sync()
            count += self.obs_terrain_full_sync(force=True)
        return count

    terrain_full_sync = obs_terrain_full_sync
    entity_full_sync = obs_diff_full_sync
    all_drain = obs_all_drain

    def patches(self) -> list[OrePatch]:
        remaining = set(self.ores)
        result: list[OrePatch] = []
        while remaining:
            start = min(remaining, key=lambda tile: (self.ores[tile][0], tile))
            name = self.ores[start][0]
            component: set[tuple[int, int]] = set()
            stack = [start]
            remaining.remove(start)
            while stack:
                tile = stack.pop()
                component.add(tile)
                x, y = tile
                for neighbor in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                    if neighbor in remaining and self.ores[neighbor][0] == name:
                        remaining.remove(neighbor)
                        stack.append(neighbor)
            tiles = tuple(sorted(component))
            min_tile = min(tiles)
            xs = [tile[0] for tile in tiles]
            ys = [tile[1] for tile in tiles]
            result.append(
                OrePatch(
                    id=f"{name}:{min_tile[0]}:{min_tile[1]}",
                    name=name,
                    n_tiles=len(tiles),
                    bbox=(min(xs), min(ys), max(xs), max(ys)),
                    _tiles=tiles,
                )
            )
        return sorted(result, key=lambda patch: patch.id)

    def is_water(self, x: int, y: int) -> bool:
        chunk = (math.floor(x / 32), math.floor(y / 32))
        mask = self.water.get(chunk)
        if mask is None or mask == 0:
            return False
        local_x, local_y = x % 32, y % 32
        bit = (31 - local_y) * 32 + local_x
        return bool((mask >> bit) & 1)

    def is_known(self, x: int, y: int) -> bool:
        """Return whether terrain for the tile's chunk is in the full cache."""
        return (math.floor(x / 32), math.floor(y / 32)) in self.water

    def known_cells(
        self, px: float, py: float, radius: float = 64
    ) -> list[tuple[float, float]]:
        cells: list[tuple[float, float]] = []
        min_cell_x = math.floor((px - radius) / 4)
        max_cell_x = math.floor((px + radius) / 4)
        min_cell_y = math.floor((py - radius) / 4)
        max_cell_y = math.floor((py + radius) / 4)
        for cell_x in range(min_cell_x, max_cell_x + 1):
            for cell_y in range(min_cell_y, max_cell_y + 1):
                x, y = cell_x * 4 + 2.0, cell_y * 4 + 2.0
                if math.hypot(x - px, y - py) > radius:
                    continue
                chunk = (math.floor(x / 32), math.floor(y / 32))
                if chunk in self.water and not self.is_water(
                    math.floor(x), math.floor(y)
                ):
                    cells.append((x, y))
        return cells

    def _read_json(self, body: str) -> Any:
        response = self.rcon.send_command(f"/sc {body}")
        if response is None or response == "":
            raise RuntimeError("RCON JSON read returned no response")
        try:
            return json.loads(response)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"RCON JSON read returned invalid JSON: {response[:200]}"
            ) from exc

    def read_reach(self) -> dict[str, float]:
        return self._read_json(
            "local c=storage.agent_characters[1] "
            "rcon.print(helpers.table_to_json({resource_reach=c.resource_reach_distance,build=c.build_distance}))"
        )

    def read_player_pos(self) -> tuple[float, float]:
        result = self._read_json(
            "local c=storage.agent_characters[1] "
            "rcon.print(helpers.table_to_json({x=c.position.x,y=c.position.y}))"
        )
        return float(result["x"]), float(result["y"])

    def read_tick(self) -> int:
        result = self._read_json("rcon.print(helpers.table_to_json({tick=game.tick}))")
        return int(result["tick"])

    def read_current_research(self) -> str | None:
        result = self._read_json(
            "local r=game.forces.player.current_research "
            "rcon.print(helpers.table_to_json({name=r and r.name or false}))"
        )
        return result["name"] or None

    def read_research_queue(self) -> tuple[str, ...]:
        """Read technologies queued behind the force's current research."""
        result = self._read_json(
            "local out={} for _,t in pairs(game.forces.player.research_queue or {}) do "
            "out[#out+1]=t.name end rcon.print(helpers.table_to_json(out))"
        )
        return tuple(str(name) for name in result)

    def read_crafting_categories(self) -> list[str]:
        return self._read_json(
            "local c=storage.agent_characters[1] local out={} "
            "for k,_ in pairs(c.prototype.crafting_categories or {}) do out[#out+1]=k end "
            "table.sort(out) rcon.print(helpers.table_to_json(out))"
        )

    def read_enabled_recipes(self) -> list[str]:
        return self._read_json(
            "local out={} for name,r in pairs(game.forces.player.recipes) do "
            "if r.enabled then out[#out+1]=name end end table.sort(out) "
            "rcon.print(helpers.table_to_json(out))"
        )

    def read_research_state(self) -> dict[str, dict[str, Any]]:
        return self._read_json(
            "local out={} for name,t in pairs(game.forces.player.technologies) do "
            "local p={} for _,v in pairs(t.prerequisites or {}) do p[#p+1]=v.name end "
            "table.sort(p) out[name]={researched=t.researched,enabled=t.enabled,prerequisites=p} "
            "end rcon.print(helpers.table_to_json(out))"
        )

    def read_research_triggers(self) -> list[str]:
        """Read technologies which Factorio refuses through ``add_research``."""
        return self._read_json(
            "local out={} for name,t in pairs(game.forces.player.technologies) do "
            "if t.prototype.research_trigger ~= nil then out[#out+1]=name end end "
            "table.sort(out) rcon.print(helpers.table_to_json(out))"
        )

    def inventory(self) -> dict[str, int]:
        inventory = self.namespace.inspect_inventory()
        return {str(name): int(count) for name, count in dict(inventory).items()}
