"""Independent, north-up 14-channel minimap decoder (four tiles per cell)."""

import numpy as np

CHANNELS = (
    "explored",
    "water",
    "iron_ore",
    "copper_ore",
    "coal",
    "stone",
    "uranium_ore",
    "crude_oil",
    "trees",
    "rocks_cliffs",
    "friendly_structures",
    "enemy_structures",
    "enemy_units",
    "player",
)
STRIDE, SIZE = 4, 128


class MinimapCache:
    def __init__(self):
        self.generation = -1
        self.surface = self.force = None
        self.origin = (0, 0)
        self.tick = 0
        self.enabled = False
        self.visibility = "charted"
        self.values = np.empty((14, 0, 0), dtype=np.float32)
        self.sampled_ticks = np.empty((0, 0), dtype=np.int64)
        self._chunks = {}
        self._invalid = set()

    def configure(self, rcon, *, chunk_budget=4, visibility="generated"):
        if not isinstance(chunk_budget, int) or not 1 <= chunk_budget <= 64:
            raise ValueError("chunk_budget must be an integer in [1,64]")
        if visibility not in ("charted", "generated"):
            raise ValueError("visibility must be charted or generated")
        return rcon.send_command(
            f'/sc obs_minimap_configure{{chunk_budget={chunk_budget},visibility="{visibility}"}}'
        )

    def _slices(self, cx, cy):
        x, y = cx * 8 - self.origin[0] // 4, cy * 8 - self.origin[1] // 4
        left, top, right, bottom = (
            max(0, x),
            max(0, y),
            min(SIZE, x + 8),
            min(SIZE, y + 8),
        )
        if left >= right or top >= bottom:
            return None
        return (slice(top, bottom), slice(left, right)), (
            slice(top - y, bottom - y),
            slice(left - x, right - x),
        )

    def _paint(self, key):
        slices = self._slices(*key)
        if slices is None:
            return
        dst, src = slices
        data, tick = self._chunks[key]
        if key in self._invalid:
            self.values[(slice(0, 13), *dst)] = -1
            self.sampled_ticks[dst] = -1
        else:
            self.values[(slice(0, 13), *dst)] = data[(slice(None), *src)]
            self.sampled_ticks[dst] = tick

    def apply_record(self, record):
        if not record or record[0] not in "MCVS":
            return False
        parts = record.split(",")
        tag = parts[0]
        if tag == "Moff":
            generation = int(parts[1])
            if generation >= self.generation:
                self.__init__()
                self.generation = generation
            return True
        if tag.startswith("M"):
            if tag != "M1":
                raise ValueError("Unsupported minimap protocol")
            generation, surface, force, x, y, tick = map(int, parts[1:7])
            if generation < self.generation:
                return True
            if x % STRIDE or y % STRIDE:
                raise ValueError("Unaligned minimap origin")
            reset = (
                generation != self.generation
                or surface != self.surface
                or force != self.force
                or not self.enabled
            )
            moved = (x, y) != self.origin
            self.generation, self.surface, self.force = generation, surface, force
            self.origin, self.tick, self.enabled = (x, y), tick, True
            self.visibility = parts[10] if len(parts) > 10 else "charted"
            if reset:
                self._chunks, self._invalid = {}, set()
            if reset or moved:
                self.values = np.full((14, SIZE, SIZE), -1, dtype=np.float32)
                self.sampled_ticks = np.full((SIZE, SIZE), -1, dtype=np.int64)
                self._chunks = {
                    k: v
                    for k, v in self._chunks.items()
                    if self._slices(*k) is not None
                }
                self._invalid.intersection_update(self._chunks)
                for key in self._chunks:
                    self._paint(key)
            self.values[13].fill(0)
            if int(parts[9]):
                px, py = map(float, parts[7:9])
                gx, gy = int((px - x) // STRIDE), int((py - y) // STRIDE)
                if 0 <= gx < SIZE and 0 <= gy < SIZE:
                    self.values[13, gy, gx] = 1
            return True
        generation, cx, cy = int(tag[1:]), int(parts[1]), int(parts[2])
        if (
            generation != self.generation
            or not self.enabled
            or self._slices(cx, cy) is None
        ):
            return True
        key = (cx, cy)
        if tag[0] == "V":
            self._invalid.add(key)
            if key in self._chunks:
                self._paint(key)
            return True
        tick = int(parts[3])
        old = self._chunks.get(key)
        if old and tick < old[1]:
            return True
        if tag[0] == "C":
            data = np.zeros(13 * 64, dtype=np.float32)
            if len(parts) > 4 and parts[4]:
                for pair in parts[4:]:
                    index, value = pair.split(":")
                    index = int(index)
                    if not 0 <= index < data.size:
                        raise ValueError("Invalid minimap cell")
                    data[index] = float(value)
            self._chunks[key] = (data.reshape(13, 8, 8), tick)
        elif old:
            self._chunks[key] = (old[0], tick)
        else:
            # Missing a prior chunk requires full_sync; do not mark unknown as fresh.
            return True
        self._invalid.discard(key)
        self._paint(key)
        return True

    def apply(self, response):
        """Consume obs_minimap_drain(), or the terrain half of obs_all_drain()."""
        return sum(self.apply_record(r) for r in response.split(";"))

    def observation(self):
        """Zero-copy view. First 13 channels are -1 until sampled/after invalidation."""
        return {
            "values": self.values,
            "sampled_ticks": self.sampled_ticks,
            "origin": self.origin,
            "stride": STRIDE,
            "channels": CHANNELS,
            "tick": self.tick,
            "surface": self.surface,
            "force": self.force,
            "generation": self.generation,
            "visibility": self.visibility,
        }
