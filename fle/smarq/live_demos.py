"""Demonstrations that work on a real Factorio map.

``fle.smarq.demos`` scripts a fixed sequence of coordinates, which is right for
the toy environment and useless on a generated map where the nearest ore is a
couple of hundred tiles away.  This module plans the same burner-automation
chain against whatever the live world actually contains, while staying inside
the rules that matter:

* every coordinate is exact and chosen by the script itself;
* nothing consults a build planner, ``nearest_buildable`` or any candidate
  generator - ore tiles come from the observation's own resource channel, and
  the initial bearing comes from one read of the world, which a scripted
  teacher is allowed to do;
* the resulting transitions are ordinary off-policy data, never a training
  target for behaviour cloning.

The chain is the one measured in TOY-B: walk to ore, place a burner mining
drill on an exact ore tile, fuel it, and let simulated time pass.
"""

from __future__ import annotations

import numpy as np

from fle.smarq import contract as C
from fle.smarq.demos import ScriptedDemo, semantic_action
from fle.smarq.raster import RESOURCE


def nearest_resource_tile(instance, name: str = "iron-ore") -> tuple[int, int] | None:
    """One read of the world to learn where the script should head."""
    # A radius ladder, never a `limit`: find_entities_filtered with a limit
    # returns an arbitrary subset, so taking the nearest of a truncated set
    # reports ore hundreds of tiles further away than it really is.
    lua = (
        "/sc local ch = storage.agent_characters[1] "
        "if not ch then rcon.print('') return end "
        "local s = game.surfaces[1] "
        "for _, radius in ipairs({16, 32, 64, 128, 256, 400}) do "
        f"  local r = s.find_entities_filtered{{name='{name}', position=ch.position, radius=radius}} "
        "  if #r > 0 then "
        "    local best, bx, by = nil, 0, 0 "
        "    for _, e in pairs(r) do "
        "      local d = (e.position.x - ch.position.x)^2 + (e.position.y - ch.position.y)^2 "
        "      if not best or d < best then best, bx, by = d, e.position.x, e.position.y end "
        "    end "
        "    rcon.print(math.floor(bx) .. ',' .. math.floor(by)) "
        "    return "
        "  end "
        "end "
        "rcon.print('')"
    )
    raw = instance.rcon_client.send_command(lua) or ""
    if "," not in raw:
        return None
    x, y = raw.split(",")[:2]
    return int(float(x)), int(float(y))


class LiveBurnerDemo(ScriptedDemo):
    """Walk to ore, put a fuelled burner drill on it, then let it run.

    The script replans at every decision from the current observation, so it
    survives a walk that lands somewhere unexpected and it addresses entities by
    the row they occupy now rather than by a remembered slot index.
    """

    def __init__(
        self,
        target: tuple[int, int] | None = None,
        *,
        drills: int = 2,
        coal_per_drill: int = 8,
        wait_blocks: int = 8,
        max_actions: int = 60,
    ) -> None:
        super().__init__([])
        self.target = target
        self.drills = drills
        self.coal_per_drill = coal_per_drill
        self.wait_blocks = wait_blocks
        self.max_actions = max_actions
        self.reset()

    def reset(self) -> None:  # type: ignore[override]
        self._emitted = 0
        self._placed: list[tuple[int, int]] = []
        self._fuelled: set[tuple[int, int]] = set()
        self._served: set[tuple[int, int]] = set()
        self._furnaces: dict[tuple[int, int], tuple[int, int]] = {}
        self._waits = 0

    @property
    def finished(self) -> bool:  # type: ignore[override]
        return self._emitted >= self.max_actions or (
            len(self._placed) >= self.drills
            and len(self._fuelled) >= self.drills
            and len(self._served) >= self.drills
            and self._waits >= self.wait_blocks
        )

    # -- helpers ---------------------------------------------------------
    @staticmethod
    def _ore_tiles(observation: C.Observation) -> list[tuple[int, int]]:
        """Exact ore tiles visible in the observation's own resource channel."""
        ys, xs = np.nonzero(observation.raster[RESOURCE] > 0)
        ox, oy = observation.raster_origin
        return [(int(ox + x), int(oy + y)) for y, x in zip(ys, xs)]

    @staticmethod
    def _entity_slot_on(
        observation: C.Observation, tile: tuple[int, int], vocab: C.VocabProtocol | None = None
    ) -> int | None:
        """The row holding a machine on this tile.

        The character occupies the same tile as a drill it just placed, and
        inserting coal into the character is a `tool_error`, so rows whose type
        is the character are skipped.
        """
        px, py = observation.player_tile
        character_ids = set()
        if vocab is not None:
            character_ids = {
                index
                for index, name in enumerate(vocab.entity_types)
                if name in ("character", "player", "<none>")
            }
        for slot in range(observation.entity_view.shape[0]):
            if not observation.entity_mask[slot]:
                continue
            if int(observation.entity_view[slot, 4]) in character_ids:
                continue
            ex = float(observation.entity_view[slot, 0]) + px
            ey = float(observation.entity_view[slot, 1]) + py
            if abs(ex - tile[0]) < 1.5 and abs(ey - tile[1]) < 1.5:
                return slot
        return None

    # -- policy ----------------------------------------------------------
    def next_action(
        self,
        observation: C.Observation,
        masks: C.Masks,
        vocab: C.VocabProtocol,
    ) -> C.Action:
        del masks
        if self.finished:
            raise StopIteration
        self._emitted += 1
        px, py = observation.player_tile
        half = observation.raster_tiles // 2 - 2

        ore = self._ore_tiles(observation)
        free_ore = [t for t in ore if t not in self._placed]

        # 1. Fuel a drill that is standing but empty.
        for tile in self._placed:
            if tile in self._fuelled:
                continue
            slot = self._entity_slot_on(observation, tile, vocab)
            if slot is None:
                continue
            self._fuelled.add(tile)
            return semantic_action(
                "INSERT",
                observation,
                vocab,
                entity_slot=slot,
                item="coal",
                quantity=self.coal_per_drill,
            )

        # 1b. A burner drill with nowhere to put its ore stalls after a few
        # items, so give each one a furnace on its own drop tile and fuel that
        # too.  The drop offset is read from the observation's entity row, not
        # from any planner.
        for tile in self._placed:
            if tile in self._served:
                continue
            slot = self._entity_slot_on(observation, tile, vocab)
            if slot is None:
                continue
            drop = (
                int(round(tile[0] + float(observation.entity_view[slot, 12]))),
                int(round(tile[1] + float(observation.entity_view[slot, 13]))),
            )
            if drop == tile:
                self._served.add(tile)
                continue
            if drop not in self._furnaces:
                self._furnaces[drop] = tile
                return semantic_action(
                    "PLACE",
                    observation,
                    vocab,
                    prototype="stone-furnace",
                    tile=drop,
                    direction="NORTH",
                )
            fslot = self._entity_slot_on(observation, drop, vocab)
            if fslot is None:
                self._served.add(tile)
                continue
            self._served.add(tile)
            return semantic_action(
                "INSERT",
                observation,
                vocab,
                entity_slot=fslot,
                item="coal",
                quantity=self.coal_per_drill,
            )

        # 2. Place a drill on an exact ore tile we can already see.
        if free_ore and len(self._placed) < self.drills:
            tile = min(free_ore, key=lambda t: (t[0] - px) ** 2 + (t[1] - py) ** 2)
            distance = max(abs(tile[0] - px), abs(tile[1] - py))
            if distance > 6:
                # Stand next to it first; the executor may navigate, but the
                # script keeps its own target exact either way.
                return semantic_action(
                    "MOVE_TO", observation, vocab, tile=(tile[0] + 2, tile[1] + 2)
                )
            self._placed.append(tile)
            return semantic_action(
                "PLACE",
                observation,
                vocab,
                prototype="burner-mining-drill",
                tile=tile,
                direction="SOUTH",
            )

        # 3. No ore in the window yet: hop toward the bearing we were given.
        if not ore and self.target is not None:
            dx = int(np.clip(self.target[0] - px, -half, half))
            dy = int(np.clip(self.target[1] - py, -half, half))
            if abs(dx) + abs(dy) > 0:
                return semantic_action(
                    "MOVE_TO", observation, vocab, tile=(px + dx, py + dy)
                )

        # 4. Everything is running: spend simulated time and let it produce.
        self._waits += 1
        return semantic_action("FAST_FORWARD", observation, vocab, duration_seconds=60)
