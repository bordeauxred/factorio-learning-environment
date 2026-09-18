"""An open-play environment whose episodes start beside an ore patch.

This is a task definition, not a change to the action space: the agent still
chooses every verb, every prototype, every exact tile and every entity itself,
and nothing is masked that was not masked before.  The only difference is where
the episode begins.

Its purpose is to separate two explanations for a null result in open play.  If
SM-ARQ learns the automation rung when it starts next to ore but not when it
starts 53 tiles away, the bottleneck is reaching the ore, and the fix is a
curriculum over start positions rather than a different learner.  If it learns
neither, the bottleneck is the learner or the credit assignment.
"""

from __future__ import annotations

from typing import Any

from fle.smarq import contract as C
from fle.smarq.env import SemanticEnv


def _nearest_ore(instance: Any, name: str = "iron-ore") -> tuple[int, int] | None:
    """A radius ladder, never a `limit`: a limited query is not the nearest."""
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
        "    rcon.print(math.floor(bx) .. ',' .. math.floor(by)) return "
        "  end "
        "end "
        "rcon.print('')"
    )
    raw = instance.rcon_client.send_command(lua) or ""
    if "," not in raw:
        return None
    x, y = raw.split(",")[:2]
    return int(float(x)), int(float(y))


class NearOreEnv(SemanticEnv):
    """`SemanticEnv` that teleports the character beside ore at every reset."""

    def __init__(self, *args: Any, start_offset: int = 5, **kwargs: Any) -> None:
        # The training loop filters kwargs against this class's signature, and a
        # bare **kwargs accepts everything, so drop what the base class does not
        # take rather than forwarding it blindly.
        kwargs.pop("seed", None)
        super().__init__(*args, **kwargs)
        self.start_offset = int(start_offset)
        self.ore_tile: tuple[int, int] | None = None

    def reset(self, seed: int | None = None) -> tuple[C.Observation, C.Masks]:
        observation, masks = super().reset(seed)
        tile = _nearest_ore(self.instance)
        self.ore_tile = tile
        if tile is None:
            return observation, masks
        # Stand a few tiles off the patch: close enough that ore is inside the
        # exact-tile window from the first decision, far enough that the agent
        # still has to choose where to stand and what to put down.
        target = (tile[0] + self.start_offset, tile[1] + self.start_offset)
        self.instance.rcon_client.send_command(
            "/sc local ch = storage.agent_characters[1] "
            f"if ch then ch.teleport({{{target[0]}, {target[1]}}}) end rcon.print('ok')"
        )
        # Re-read the world after the teleport so the window is centred correctly.
        observation = self._observe(full=True)
        return observation, self.masks()
