"""Stable vocabularies for the SM-ARQ environment.

Names are obtained from Factorio's authoritative prototype tables and sorted
lexicographically.  The sort, reserved ``<none>`` entry, source string, and
game version form the cache payload, so independently started environment and
learner processes assign identical indices.

The payload can be serialized with :meth:`StableVocab.write_json`.  Runtime
code deliberately does not rewrite the package directory; checked-in callers
may choose an explicit cache path when preparing an experiment artifact.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from fle.env.game_types import Prototype, Technology
from fle.smarq import contract as C

NONE = "<none>"
SOURCE = "Factorio runtime prototype tables (sorted lexicographically)"


def _stable(names: Iterable[str]) -> tuple[str, ...]:
    return (NONE, *sorted({str(name) for name in names if name and name != NONE}))


def _rcon(rcon: Any, lua: str) -> str:
    response = rcon.send_command("/sc " + lua)
    return response or ""


def _prototype_names(rcon: Any, kind: str, predicate: str = "true") -> tuple[str, ...]:
    lua = (
        "local a={} for n,p in pairs(prototypes."
        + kind
        + ") do if "
        + predicate
        + " then a[#a+1]=n end end table.sort(a) "
        "rcon.print(table.concat(a,'|'))"
    )
    return _stable(_rcon(rcon, lua).split("|"))


@dataclass(frozen=True)
class StableVocab:
    """A JSON-serializable implementation of :class:`C.VocabProtocol`."""

    prototypes: tuple[str, ...]
    items: tuple[str, ...]
    recipes: tuple[str, ...]
    technologies: tuple[str, ...]
    entity_types: tuple[str, ...]
    fluids: tuple[str, ...] = (NONE,)
    source: str = SOURCE
    game_version: str = "unknown"
    schema_version: str = C.SCHEMA_VERSION

    def __post_init__(self) -> None:
        for field_name in (
            "prototypes",
            "items",
            "recipes",
            "technologies",
            "entity_types",
            "fluids",
        ):
            values = getattr(self, field_name)
            if not values or values[0] != NONE:
                raise ValueError(f"{field_name}[0] must be {NONE!r}")
            if len(values) != len(set(values)):
                raise ValueError(f"duplicate symbols in {field_name}")

    @classmethod
    def from_rcon(cls, rcon: Any) -> "StableVocab":
        """Build from Factorio's own tables, without inventory-dependent data."""
        game_version = _rcon(rcon, "rcon.print(script.active_mods.base)")
        # A placeable prototype is an item whose place_result is an entity.
        prototypes = _prototype_names(rcon, "item", "p.place_result ~= nil")
        items = _prototype_names(rcon, "item")
        recipes = _prototype_names(rcon, "recipe")
        technologies = _prototype_names(rcon, "technology")
        # Rich observation rows encode entity prototype names in F_TYPE.
        entity_types = _prototype_names(rcon, "entity")
        fluids = _prototype_names(rcon, "fluid")
        return cls(
            prototypes=prototypes,
            items=items,
            recipes=recipes,
            technologies=technologies,
            entity_types=entity_types,
            fluids=fluids,
            game_version=game_version or "unknown",
        )

    @classmethod
    def fallback(cls) -> "StableVocab":
        """Deterministic offline vocabulary sourced from FLE's enums."""
        prototype_names = [p.value[0] for p in Prototype]
        placeable = [p.value[0] for p in Prototype if p.value[1] is not None]
        technologies = [t.value for t in Technology]
        return cls(
            prototypes=_stable(placeable),
            items=_stable(prototype_names),
            recipes=_stable(prototype_names),
            technologies=_stable(technologies),
            entity_types=_stable(placeable),
            fluids=(NONE,),
            source="FLE fle.env.game_types enums (offline fallback)",
            game_version="offline",
        )

    @classmethod
    def read_json(cls, path: str | Path) -> "StableVocab":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        for key in (
            "prototypes",
            "items",
            "recipes",
            "technologies",
            "entity_types",
            "fluids",
        ):
            if key in payload:
                payload[key] = tuple(payload[key])
        return cls(**payload)

    def write_json(self, path: str | Path) -> None:
        """Write the stable cache atomically to an explicitly selected path."""
        target = Path(path)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(
            json.dumps(asdict(self), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(target)

    def head_sizes(self, raster_tiles: int) -> dict[str, int]:
        return C.VocabProtocol.head_sizes(self, raster_tiles)  # type: ignore[arg-type]

    def index_maps(self) -> dict[str, dict[str, int]]:
        return {
            name: {symbol: index for index, symbol in enumerate(getattr(self, name))}
            for name in (
                "prototypes",
                "items",
                "recipes",
                "technologies",
                "entity_types",
                "fluids",
            )
        }


def load_vocab(rcon: Any | None = None, cache_path: str | Path | None = None) -> StableVocab:
    """Load a cache or build from a live Factorio prototype table.

    Calling this without either RCON or an existing cache is refused: the FLE
    enum fallback is intentionally incomplete and must never define neural
    categorical indices.  Offline callers that explicitly need test data can
    use :meth:`StableVocab.fallback` directly.
    """
    if cache_path is not None and Path(cache_path).exists():
        return StableVocab.read_json(cache_path)
    if rcon is None:
        raise RuntimeError(
            "load_vocab requires a live RCON client or an existing cache; "
            "StableVocab.fallback() is test-only and must not build a network"
        )
    vocab = StableVocab.from_rcon(rcon)
    if cache_path is not None:
        vocab.write_json(cache_path)
    return vocab
