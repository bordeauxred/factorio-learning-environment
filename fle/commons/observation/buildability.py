"""Tile-resolution client for progressive buildability records.

Values are -1 (unknown), 0 (blocked), or 1 (legal at sampled tick). Tick stamps
are per 8x8 block, not a promise that a cached answer is current. Instances
consume one ordered server drain stream, like the existing observation client.
"""

import json
from pathlib import Path

import numpy as np

MANIFEST = json.loads(
    Path(__file__).with_name("buildability_channels.json").read_text()
)
CHANNELS = MANIFEST["channels"]
CHANNEL_COUNT = len(CHANNELS)
BLOCK_SIZE = 8
CHANNEL_FOR = {
    (name, direction): channel["id"]
    for channel in CHANNELS
    for name, direction in channel["members"]
}


class BuildabilityCache:
    def __init__(self):
        self.generation = -1
        self.surface = self.force = None
        self.origin = (0, 0)
        self.size = 0
        self.values = np.full((CHANNEL_COUNT, 0, 0), -1, dtype=np.int8)
        self.sampled_ticks = np.full((CHANNEL_COUNT, 0, 0), -1, dtype=np.int64)

    @staticmethod
    def channel_for(name, direction):
        """Direction is Factorio's engine cardinal value (0, 4, 8, 12)."""
        return CHANNEL_FOR[("straight-rail" if name == "rail" else name, direction)]

    def configure(self, rcon, *, size=128, check_budget=256, refresh_ticks=600):
        """Enable a single-player cache; data arrives in obs_all_drain()."""
        if size < 16 or size > 288 or size % 16:
            raise ValueError("size must be a multiple of 16 in [16, 288]")
        if check_budget < 128 or check_budget > 8192 or check_budget % 128:
            raise ValueError("check_budget must be a multiple of 128 in [128, 8192]")
        if not isinstance(refresh_ticks, int) or refresh_ticks < 1:
            raise ValueError("refresh_ticks must be a positive integer")
        return rcon.send_command(
            "/sc obs_buildability_configure{size=%d,check_budget=%d,refresh_ticks=%d}"
            % (size, check_budget, refresh_ticks)
        )

    def _metadata(self, parts):
        if parts[0] == "Boff":
            generation = int(parts[1])
            if generation >= self.generation:
                self.__init__()
                self.generation = generation
            return
        if parts[0][1:] != MANIFEST["id"]:
            raise ValueError("Buildability channel manifest differs from the server")
        generation, surface, force, x, y, size = map(int, parts[1:])
        if generation < self.generation:
            return
        if size < 16 or size > 288 or size % 16 or x % 8 or y % 8:
            raise ValueError("Invalid buildability window")
        keep = (
            generation == self.generation
            and surface == self.surface
            and force == self.force
            and size == self.size
        )
        values = np.full((CHANNEL_COUNT, size, size), -1, dtype=np.int8)
        ticks = np.full((CHANNEL_COUNT, size // 8, size // 8), -1, dtype=np.int64)
        if keep:
            ox, oy = self.origin
            left, top = max(x, ox), max(y, oy)
            right, bottom = min(x + size, ox + size), min(y + size, oy + size)
            if left < right and top < bottom:
                values[:, top - y : bottom - y, left - x : right - x] = self.values[
                    :, top - oy : bottom - oy, left - ox : right - ox
                ]
                ticks[
                    :,
                    (top - y) // 8 : (bottom - y) // 8,
                    (left - x) // 8 : (right - x) // 8,
                ] = self.sampled_ticks[
                    :,
                    (top - oy) // 8 : (bottom - oy) // 8,
                    (left - ox) // 8 : (right - ox) // 8,
                ]
        self.generation, self.surface, self.force = generation, surface, force
        self.origin, self.size = (x, y), size
        self.values, self.sampled_ticks = values, ticks

    def apply_record(self, record):
        """Consume a buildability record; return False for other terrain tags."""
        if not record or record[0] not in "BbRJ":
            return False
        parts = record.split(",")
        if record[0] == "B":
            self._metadata(parts)
            return True
        if int(parts[0][1:]) != self.generation or not self.size:
            return True
        if record[0] == "J":
            channel = int(parts[1])
            self.values[channel].fill(-1)
            self.sampled_ticks[channel].fill(-1)
        elif record[0] == "R":
            x0, y0, x1, y1 = map(int, parts[1:])
            ox, oy = self.origin[0] // 8, self.origin[1] // 8
            x0, y0 = max(0, x0 - ox), max(0, y0 - oy)
            x1, y1 = min(self.size // 8, x1 - ox + 1), min(self.size // 8, y1 - oy + 1)
            if x0 < x1 and y0 < y1:
                self.values[:, y0 * 8 : y1 * 8, x0 * 8 : x1 * 8] = -1
                self.sampled_ticks[:, y0:y1, x0:x1] = -1
        else:
            channel, cx, cy, tick = map(int, parts[1:5])
            bx, by = cx - self.origin[0] // 8, cy - self.origin[1] // 8
            if not (0 <= channel < CHANNEL_COUNT):
                raise ValueError("Invalid buildability channel")
            if not (0 <= bx < self.size // 8 and 0 <= by < self.size // 8):
                return True
            if tick < self.sampled_ticks[channel, by, bx]:
                return True
            payload = parts[5]
            if len(payload) != 16:
                raise ValueError("Invalid buildability bitmask")
            words = np.array(
                [int(payload[:8], 16), int(payload[8:], 16)], dtype=np.uint32
            )
            bits = ((words[:, None] >> np.arange(32, dtype=np.uint32)) & 1).reshape(
                8, 8
            )
            self.values[channel, by * 8 : by * 8 + 8, bx * 8 : bx * 8 + 8] = bits
            self.sampled_ticks[channel, by, bx] = tick
        return True

    def legal_mask(self, current_tick, max_age_ticks):
        """Conservative dense action mask: unknown/expired samples are False.

        Age limits bound accepted staleness, not silent-world-mutation detection.
        Revalidate the selected action against the engine before execution.
        """
        if max_age_ticks < 0:
            raise ValueError("max_age_ticks must be nonnegative")
        age = current_tick - self.sampled_ticks
        fresh = (self.sampled_ticks >= 0) & (age >= 0) & (age <= max_age_ticks)
        return (self.values == 1) & fresh.repeat(8, axis=1).repeat(8, axis=2)

    def observation(self, current_tick):
        """Zero-copy arrays and their coordinate/freshness metadata.

        Copy arrays if retaining this observation across subsequent drains.
        Array [channel, y, x] represents world tile origin + (x,y); add the
        selected entity's normal integer/half-integer anchor offset to query it.
        """
        return {
            "values": self.values,
            "sampled_ticks": self.sampled_ticks,
            "origin": self.origin,
            "surface": self.surface,
            "force": self.force,
            "tick": current_tick,
            "generation": self.generation,
            "block_size": BLOCK_SIZE,
            "manifest": MANIFEST["id"],
        }
