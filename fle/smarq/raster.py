"""Exact-tile egocentric raster derived solely from ``TensorClient`` state."""

from __future__ import annotations

import math

import numpy as np

from fle.smarq import contract as C
from fle.smarq.obs import TensorClient

OCCUPIED, IMPASSABLE, RESOURCE, BELT, MACHINE, TREE_ROCK, PLAYER, BUILD_REACH = range(
    C.RASTER_CHANNELS
)


def _tile_span(center: float, width: float) -> range:
    """Tiles covered by a Factorio footprint centred at ``center``."""
    width = max(1.0, width)
    start = math.floor(center - width / 2.0 + 1e-6)
    return range(start, start + max(1, int(round(width))))


def _is_belt(name: str) -> bool:
    return "transport-belt" in name or name in {"splitter", "loader", "loader-1x1"}


def _is_machine(name: str) -> bool:
    return any(
        token in name
        for token in (
            "assembling-machine",
            "furnace",
            "mining-drill",
            "chemical-plant",
            "oil-refinery",
            "centrifuge",
            "rocket-silo",
        )
    )


class ExactTileRaster:
    """Build contract raster channels without querying placement feasibility."""

    def __init__(self, side: int = C.RASTER_TILES_DEFAULT) -> None:
        if side <= 0:
            raise ValueError("raster side must be positive")
        self.side = int(side)

    def build(self, client: TensorClient) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
        player_tile = (math.floor(client.player_x), math.floor(client.player_y))
        origin = (player_tile[0] - self.side // 2, player_tile[1] - self.side // 2)
        raster = np.zeros((C.RASTER_CHANNELS, self.side, self.side), dtype=np.float32)

        def mark(channel: int, x: int, y: int) -> None:
            rx, ry = x - origin[0], y - origin[1]
            if 0 <= rx < self.side and 0 <= ry < self.side:
                raster[channel, ry, rx] = 1.0

        for row in client.entities.values():
            parsed = client._parse_row(row)
            width = parsed["tile_w"] or 1.0
            height = parsed["tile_h"] or 1.0
            for x in _tile_span(parsed["x"], width):
                for y in _tile_span(parsed["y"], height):
                    mark(OCCUPIED, x, y)
                    if _is_belt(parsed["name"]):
                        mark(BELT, x, y)
                    if _is_machine(parsed["name"]):
                        mark(MACHINE, x, y)

        for (cx, cy), mask in client.water.items():
            if not mask:
                continue
            hexmask = format(mask, "0256x")
            for dy in range(32):
                row_bits = int(hexmask[dy * 8 : (dy + 1) * 8], 16)
                while row_bits:
                    least = row_bits & -row_bits
                    dx = least.bit_length() - 1
                    mark(IMPASSABLE, cx * 32 + dx, cy * 32 + dy)
                    row_bits ^= least

        for key in client.ores:
            x, y = map(int, key.split(":"))
            mark(OCCUPIED, x, y)
            mark(RESOURCE, x, y)

        for key in client.trees | client.obstacles:
            x, y = map(int, key.split(":"))
            mark(OCCUPIED, x, y)
            mark(IMPASSABLE, x, y)
            mark(TREE_ROCK, x, y)

        for _, x, y in client.nests.values():
            mark(OCCUPIED, math.floor(x), math.floor(y))

        mark(PLAYER, *player_tile)
        reach2 = float(client.build_distance) ** 2
        xs = np.arange(origin[0], origin[0] + self.side, dtype=np.float32) + 0.5
        ys = np.arange(origin[1], origin[1] + self.side, dtype=np.float32) + 0.5
        dx2 = (xs - client.player_x) ** 2
        dy2 = (ys - client.player_y) ** 2
        raster[BUILD_REACH] = (dy2[:, None] + dx2[None, :] <= reach2).astype(np.float32)
        return raster, origin, player_tile
