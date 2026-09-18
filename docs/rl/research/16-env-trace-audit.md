# Current-environment trace audit (macro-v2)

## Scope and verdict

I took a fixed cut at **2026-09-17 01:34:26 CEST**: 12,948 steps and 69 completed episodes from all six streams in `armB-v3`, `armD-v3`, `qr-v3`, and the newly started `ctrl-v3`. Runs were still appending, so counts below describe that cut. I treated messages that differ only in coordinates/unit IDs as one message template, while retaining distinct prototypes/items; this covers all 96 exact non-empty `(op, raw_error)` strings as 54 semantic messages.

The quantity CRAFT fix is sound. The remaining dominant failures are environment support failures: global factorised masks admit impossible entity/item/direction combinations, PLACE masks one tile instead of the actual footprint, cooldowns are not represented in the returned observation, and harvest targets include unharvestable or stale resources. These cost far more steps than legitimate game failures.

## Tool rejections

Classification: **M** = deterministic refusal support should prevent (our mask/guard bug); **X** = our execution/observation path disagrees with the tool; **G** = legitimate pathfinding/game outcome.

| op (total rejected) | every distinct message/template and count | verdict |
|---|---|---|
| CONNECT (1) | `Transport belts cannot be connected directly from a Prototype.StoneFurnace ...` (1) | **M**. QR `27003/e3:s232` supplied the same furnace unit as both source and peer. The new position-endpoint call itself is correct; distinct-endpoint and connector/source compatibility masks are absent. |
| CRAFT (0), RESEARCH (0) | no rejection messages | Fine. All 389 crafts and all 2,598 research calls reached `ok`. |
| ROTATE (581) | empty/`cooldown_masked` (161); `Could not rotate stone-furnace.` (348); `Could not rotate wooden-chest.` (72) | **M**. Both prototypes are direction-invariant. |
| INSERT (500) | empty/`insert_item_not_held` (163), empty/cooldown (120); `cannot accept stone-furnace` 78, `copper-plate` 3, `wooden-chest` 3, `transport-belt` 1; `Inventory is full` 14; `LuaItemStack ... invalid for read` 24; furnace-content conflicts: existing copper-ore→iron-ore 7, stone 1, stone-furnace 1, wooden-chest 1; existing iron-ore→copper-ore 13, iron-plate 1, stone 13, stone-furnace 17; existing stone→copper-ore 9, copper-plate 1, iron-ore 12, iron-plate 1, stone-furnace 17 | **M** throughout. The item mask is the union of player and entity contents; it is not conditioned on INSERT, selected entity, slot type, current furnace input, or free capacity. The LuaItemStack error is the tool's full-slot diagnostic failing, but the deterministic full call should never be sent. |
| EXTRACT (326) | empty/`extract_item_not_in_entity` (269), empty/cooldown (38); `Could not extract <item> from <entity> ... Could not find a valid ... containing <item>` (19: chest contents 12, furnace input 7) | The empty cases are **M**, again caused by the union item mask. The 19 non-empty cases are **X**: `_invalid_combination` saw the item in that exact row, but after approach the tool did not. Furnace input can change while walking, but chest contents cannot; revalidation after approach is required. |
| PLACE (56) | empty/cooldown 17; collision with tree-01 3, tree-02 1, tree-02-stump 1, tree-03 2, tree-04 2, tree-05 1, tree-07 3, tree-08 6, dead/dry tree variants 4, big/huge rock 4, cliff 4, water footprint 1; snapped target out of reach 7 | All **M**. The offset mask tests only its centre tile. It neither rasterizes the direction-dependent footprint nor measures reach to the snapped placement centre. |
| HARVEST (146) | empty/cooldown 73; `Nothing within reach to harvest` 49; `Could not harvest. crude-oil` 22; path-not-found 1; `LuaEntity ... invalid` 1 | Crude oil and stale/depleted trees are **M**; the path failure is **G**. The LuaEntity case is **X** and important: B `27000/e0:s23` nevertheless acquired 10 wood and 1 stone, but the exception bypassed effect verification and logged `tool_rejected`. No `approach_failed` occurred. |
| MOVE (239) | empty/cooldown 223; path-not-found 12; `no known land cell in the selected direction` 4 | No-known-land is **M** because the direction was admitted before `_nearest_land_in_direction` deterministically failed. Path-not-found is **G**. |

## Effects, masks, quantities, and cooldowns

All 180 `no_effect` steps are ROTATE: 109 stone-furnace and 71 wooden-chest calls requested direction 0 when before and after were already 0. The predicate is correct; the tool accepted an idempotent call and did nothing. This is another missing conditional support rule, not observation lag or a predicate bug. No other op produced `no_effect`.

For a seeded random sample of 200 steps, the environment's packed-mask check produced **0 `masked_op/head` violations**. Index decoding was also exact over the full cut: **0 mismatches** between action-vector indices and logged PLACE item, INSERT/EXTRACT item, or CRAFT/SET_RECIPE recipe. However, this is not independently reproducible from the files: neither env nor action logs store the preceding observation/masks, and `actions_*.jsonl` stores labels rather than chosen indices. More importantly, 13/200 sampled actions were `cooldown_masked`; cooldown is checked privately in `step()` and never written into the observation tail. Thus base masks are internally consistent, but the advertised mask is not the true acceptance set.

Quantity results were:

| op | calls | clamped | clamped status |
|---|---:|---:|---|
| CRAFT | 389 | 218 (`5→1` 95, `20→1` 88, `20→5` 35) | **218 ok, 0 failed** |
| INSERT | 934 | 416 | 169 ok, 247 rejected |
| EXTRACT | 456 | 362 | 84 ok, 278 rejected |

INSERT/EXTRACT failures after clamping are not clamp failures: 269 EXTRACT and 163 INSERT calls clamped to zero because the chosen conjunction had no item, while the rest hit slot compatibility/capacity. Two successful INSERTs accepted quantity 20 but moved only 2 and 16 ore; that is legitimate partial insertion, but `executed_quantity` currently means “argument passed,” not “items transferred.”

There were **632 cooldown steps**: MOVE 223, ROTATE 161, INSERT 120, HARVEST 73, EXTRACT 38, PLACE 17. Every one is an identical refused full action repeated with an unchanged fingerprint. B `27001/e16:s53` repeated one MOVE for 57 cooldown steps. Cooldown therefore protects the tool but not compute.

## Reward, time, episode, and trace integrity

Rewards are exact: 0 step-level APS-delta mismatches and 0 episode telescoping mismatches. Every reward at most -5 was INSERT consumption (23 at -6, 12 at -7, 6 at -8, 1 at -5). The many -1 HARVEST/CRAFT transitions are the documented score floor, not a telescope error.

| op | median ticks | median wall s | p95 wall s |
|---|---:|---:|---:|
| CONNECT | 1465 | 1.515 | 1.515 |
| CRAFT | 718 | 0.922 | 2.081 |
| EXTRACT | 764 | 0.920 | 1.864 |
| HARVEST | 1181 | 1.432 | 2.153 |
| INSERT | 1038 | 1.110 | 1.858 |
| MOVE | 1191 | 1.362 | 2.595 |
| PICKUP | 926 | 1.137 | 2.050 |
| PLACE | 550 | 0.634 | 1.743 |
| RESEARCH | 979 | 1.124 | 2.685 |
| ROTATE | 968 | 1.082 | 1.815 |
| WAIT | 1380 | 1.394 | 4.580 |

No op has p95 above 10 s. There are 86 isolated >10 s steps (maximum WAIT 24.34 s), spread across ops and consistent with server stalls rather than one tool's watchdog failure.

All 53 completed B/D episodes ended before their nominal 256/512 horizon because accumulated ticks reached 216,000; QR's 16 completed episodes reached 256 steps. In particular, every completed “512-step” episode actually lasted 67–310 steps, so the explorer horizon is not being delivered. There were no deaths. Six final episodes were merely in progress at the cut. Completed episodes all have one `episode_end`; there are no duplicate steps, gaps, or action/env op misalignments.

Inventory deltas match the acting op except the partial-success HARVEST above and the two legitimate partial INSERTs. Position jumps cannot be audited because positions are absent; HARVEST and entity ops intentionally approach targets, so non-MOVE movement is expected macro semantics.

## Ranked fixes

1. **`fle/rl/observation.py::_write_masks` plus `fle/rl/dqn.py::greedy_actions_from_outputs` / `randomize_actions` / `UCBExplorer.select` — mask fix.** Represent and consume entity×item×op support (held INSERT items, contents for the selected EXTRACT entity, compatible/free slots) instead of the present union item mask.
2. **`fle/rl/observation.py::_write_masks` / `fle/rl/env.py::_place_position` — mask fix.** Evaluate the snapped, rotated full footprint against water, entities, trees/obstacles/cliffs and actual centre reach.
3. **`fle/rl/observation.py::_target_slots` / `_write_masks` — mask fix.** Exclude crude oil and remove invalid/depleted tree targets immediately after terrain drains.
4. **`fle/rl/dqn.py::randomize_actions` / `UCBExplorer.select` and `FleMacroEnv` cooldown state — mask fix.** Feed the exact refused-action blacklist back to selection (or expose equivalent conditional support) so a cooldown action cannot be chosen again.
5. **`fle/rl/observation.py::_write_masks` / `env.py::_invalid_combination` — mask fix.** Admit ROTATE only for rotatable prototypes and directions different from the selected row's current direction.
6. **`fle/rl/env.py::_execute_connect` / `_invalid_combination` — mask fix.** Require distinct endpoints and validate connector compatibility before calling the position-endpoint tool.
7. **`fle/rl/ops.py::execute_action` and `env.py::step` — executor fix.** After approach, drain/revalidate entity contents and, after tool exceptions, still reconcile observed effects so partial success is not labelled refusal.
8. **`fle/rl/env.py::__init__` / `reset` / `step` — executor fix.** Scale the tick cap with the requested step horizon so a nominal 512-step episode is not always truncated around 67–310 steps.
9. **`fle/rl/env.py::step` / `_write_episode_end` — logging fix.** Record the explicit termination/truncation cause.
10. **`fle/rl/env.py::step` and `fle/rl/dqn.py` action logging — logging fix.** Log before/after position, preceding used-head masks (or their admitted indices), decoded args even on cooldown, and actual transferred quantity separately from the clamped call argument.
