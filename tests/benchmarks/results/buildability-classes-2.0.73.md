# New-construction buildability classes — Factorio 2.0.73

## Scope

The user specified that occupied locations are unbuildable: no upgrades, replacement, or rotation of an existing entity. This analysis retains legitimate engine collision rules (e.g. rail-associated placements) rather than declaring every tile touched by any entity universally blocked. Inventory and reach remain separate per-agent gates. Four cardinal directions and the frozen base-game catalog are covered; mods, diagonal rail actions, additional placement modes, and arbitrary off-grid action lattices require reevaluation.

## Candidate reduced representation

**37 entity classes, 63 orientation-specific binary channels**: 21 classes × 1, 11 × 2, and 5 × 4. A four-bit direction value could pack this into 37 integer planes, but a binary-channel policy uses 63 planes.

This is a validated prototype-rule grouping, not a mathematical proof that 63 is the absolute minimum. Do not merge classes merely because they match on a finite set of scenes.

Signatures use exact rotated collision boxes, tile-anchor alignment, collision connector options, collision compatibility with every loaded entity/tile prototype, ore categories and mining radius, and directional tile rules. Rails, signals, train stops, gates and rolling stock retain special rules. Boiler/heat-exchanger fluid connections require four directions even though the rectangular footprint has only two orientations.

| Class | Members | Binary channels |
|---:|---|---:|
| 1 | accumulator | 1 |
| 2 | active-provider-chest, buffer-chest, constant-combinator, iron-chest, passive-provider-chest, requester-chest, steel-chest, storage-chest, wooden-chest | 1 |
| 3 | arithmetic-combinator, decider-combinator | 2 |
| 4 | artillery-turret, assembling-machine-1, assembling-machine-2, assembling-machine-3, beacon, centrifuge, chemical-plant, electric-furnace, lab, radar | 1 |
| 5 | big-electric-pole | 1 |
| 6 | boiler, heat-exchanger | 4 |
| 7 | bulk-inserter, burner-inserter, fast-inserter, inserter, long-handed-inserter, medium-electric-pole, small-electric-pole, small-lamp | 1 |
| 8 | burner-mining-drill | 1 |
| 9 | car | 2 |
| 10 | cargo-wagon, fluid-wagon | 2 |
| 11 | electric-mining-drill | 1 |
| 12 | express-splitter, fast-splitter, splitter | 2 |
| 13 | express-transport-belt, express-underground-belt, fast-transport-belt, fast-underground-belt, transport-belt, underground-belt | 1 |
| 14 | flamethrower-turret | 2 |
| 15 | gate | 2 |
| 16 | gun-turret, laser-turret, power-switch, steel-furnace, stone-furnace, substation | 1 |
| 17 | heat-pipe | 1 |
| 18 | land-mine | 1 |
| 19 | locomotive | 2 |
| 20 | nuclear-reactor | 1 |
| 21 | offshore-pump | 4 |
| 22 | oil-refinery | 1 |
| 23 | pipe, stone-wall | 1 |
| 24 | pipe-to-ground | 4 |
| 25 | programmable-speaker | 1 |
| 26 | pump | 2 |
| 27 | pumpjack | 1 |
| 28 | straight-rail | 2 |
| 29 | rail-chain-signal, rail-signal | 4 |
| 30 | roboport | 1 |
| 31 | rocket-silo | 1 |
| 32 | solar-panel | 1 |
| 33 | spidertron | 1 |
| 34 | steam-engine, steam-turbine | 2 |
| 35 | storage-tank | 1 |
| 36 | tank | 2 |
| 37 | train-stop | 4 |

## Why the scope matters

Manual placement alone includes replacement and rotation behavior. An existing assembling-machine-1 rejects an identical build but accepts another tier. Consequently chest and assembler tiers cannot share the final mask if upgrades are permitted. The initial manual-only suite found at least 69 different entity mask families. This is outside the clarified new-construction scope.

The new-only test predicate intersects `can_place_entity(..., build_check_type=manual)` with `can_place_entity(..., build_check_type=ghost_revive)`: retain manual special-placement restrictions while rejecting builds whose existing collision would prevent revival. This is a tested engine-query predicate, not yet a production action contract.

A first 61-channel candidate failed 32 comparisons: east/west boiler and heat-exchanger connections produced different manual outcomes near other fluid machines. Separating all four boiler orientations gives the revised 63-channel candidate.

## Reproduction

```sh
python tests/benchmarks/analyze_buildability_classes.py
python tests/benchmarks/buildability_classes.py
python tests/benchmarks/validate_buildability_classes.py --check-type new_only --extended
```

The JSON artifacts preserve prototype metadata, the complete 79-entity mapping, and live parity comparisons. Existing observation implementation has not been changed.

## Expanded validation result

The 63-channel candidate passed **15,511,176 entity/direction/position decisions across 606 scenes with zero within-class mismatches** under the new-only predicate. Each decision invokes the manual check and, when true, the ghost-revival collision check. Scenes include every catalog entity at every cardinal orientation, filled fluid networks, ore/oil, four shore orientations, natural obstacles and fractional characters. This supports shared computation for the mapped entities; it does not prove a universal minimum.

## Minimality limit

The extended suite demonstrates 56 pairwise-distinct output patterns; the conservative rule signatures retain 63. Therefore **63 is a validated sufficient candidate, not a certified irredundant minimum**. The remaining possible reductions involve belt versus heat-pipe masks, underground-pipe orientations, and rail-vehicle behavior. Fractional-obstacle follow-up (5,068,008 decisions, zero sharing mismatches) did not resolve these equivalences. Rail probing found positive placements for all three rolling-stock types and confirmed placement snaps to the track, so their initial all-false samples must not be used as evidence of equivalence.

Do not state that the smallest full-parity set has been proven. The current concrete result is a 37-class/63-channel mapping with zero counterexamples in these runs, and a witnessed lower bound of 56 binary patterns for this candidate action representation. Production integration should preserve an authoritative final placement check.

## Resolved: underground-pipe directions

Focused verification now proves all four underground-pipe direction masks distinct, using incompatible water/steam fluid networks around a completely empty target tile. All six direction pairs have counterexamples. See [the underground-pipe report](underground-pipe-verification-2.0.73.md). Retain four channels. The full-set witnessed lower bound increases to **59**, while the conservative candidate remains **63**. Only the belt/heat-pipe and rolling-stock equivalences from the previous list remain unresolved.
