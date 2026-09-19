# Underground-pipe mask verification — Factorio 2.0.73

**Result: four directional binary channels are necessary.** All six direction pairs have concrete witnesses on an entirely empty target tile. Underground pipes cannot be reduced to one or two final buildability masks, even with occupied sites excluded.

## Decisive fluid-connection witness

Target: `(0.5, 0.5)`, the center of tile `(0, 0)`.

- Ordinary pipe at `(0.5, -0.5)`, containing water.
- Underground pipe at `(0.5, 2.5)`, facing south (engine direction 8), containing steam.
- Target tile has no intersecting entity bounding boxes; the script asserts this.

A new north-facing underground pipe would bridge those two incompatible fluid systems. Factorio rejects it. The other three directions are legal. Rotate the whole setup around the target to distinguish all directions:

| Scene rotation | North | East | South | West |
|---|---|---|---|---|
| 0° | blocked | legal | legal | legal |
| 90° | legal | blocked | legal | legal |
| 180° | legal | legal | blocked | legal |
| 270° | legal | legal | legal | blocked |

These four rows prove that no pair of directional masks is identical across valid scenes. Four channels are therefore minimal for the four cardinal `pipe-to-ground` placement actions in a binary, one-mask-per-direction representation. Bit packing can change storage but not the number of distinct predicates.

## Scope and validation

Used the established new-construction predicate: `can_place_entity(manual) AND can_place_entity(ghost_revive)`. The world is paused, in an isolated base-game 2.0.73 Docker container. No production observation code was changed.

The verification evaluates **262 cases**, each in four directions and both engine check modes (2,096 API checks). Cases cover empty/occupied targets, neighboring pipes/belts/chests, fractional-position cars, underground distances 2/5/10/11, counterpart orientations, and same/different fluids. All setup commands use checked JSON; entity creation, actual positions, fluid assignment and target-tile clearance are asserted. Raw entities, bounding boxes, fluids and per-direction outcomes are saved in the adjacent JSON file.

The run also found asymmetric collision witnesses using a fractional-position car. Those are supplementary; the empty-tile fluid witnesses alone establish the result under the user's occupied-tile rule.

The earlier broad tests placed one fluid-bearing neighbor at a time. They did not join **two separate fluid systems** through the candidate entity, which is why they missed this difference.

## Observation implications

Keep the four existing underground-pipe channels; the conservative overall count stays **63**. The witnessed lower bound for the full channel set rises from 56 to **59**, since all four underground directions are now distinguished. Belt/heat-pipe and rolling-stock equivalences remain separate unresolved questions.

Invalidation cannot depend only on occupancy: fluid contents/connectivity and the facing/range of nearby underground endpoints affect these masks. The current quantized observation cache is insufficient to reconstruct an exact authoritative answer without additional state/invalidation or engine checks.

## Reproduce

```sh
python tests/benchmarks/verify_underground_pipe_masks.py
```

The script starts and removes its own container and asserts all six directional counterexamples. It writes `underground-pipe-verification-2.0.73.json` alongside this report.
