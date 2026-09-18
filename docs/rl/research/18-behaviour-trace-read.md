# Behaviour trace read, 2026-09-17

## Scope

This is a read-only analysis of a stable copy of the six traces at 10:49 CEST. “Recent” is the last 20 completed root episodes in metrics order, so it crosses the 10:33 checkpoint restart; “since the last restart” is only the appended 10:33 segment. APS/step is final APS divided by actual decisions, not configured horizon. Excursions are excluded from score comparisons.

## What the best recent policies do

| arm; trace episode | observed strategy and APS trajectory |
|---|---|
| **dqn-per; 27004 s1/e53, APS 447/256** | Crafts and places a furnace by steps 5–6, adds coal then iron, and reaches 43 APS by step 10. It sprawls to **19 concurrent furnaces** (34 placements, 18 pickups; 16 remain), inserting 199 coal, 261 iron ore and 120 copper ore. APS reaches 101/200/402 at steps 77/101/177. The productive front half gives way to rebuilding and research: 48 RESEARCH, 34 PLACE, 18 PICKUP, and repeated zero-transfer inserts. |
| **dqn-per-ucbexplorer; UCB 27001 s1/e31, APS 1,699/205** | Crafts four furnaces immediately, then mostly operates **one furnace at a time**, recycling it 11 times (maximum four concurrent). It inserts 920 iron ore and 122 coal in alternating harvest/feed passes. APS rises almost linearly: 225 at step 20, 800 at 78, 1,218 at 153, 1,699 at 204. All 205 calls are `ok`; only one research action occurs. The 512-step episode ends at the tick cap. |
| **qrdqn-per-ucbexplorer-frontier; epsilon 27002 s3/e11, APS 260/124** | Builds a small line (maximum four furnaces, three final), primarily fuelled by wood: 246 wood, 122 iron ore and 85 stone inserted. It reaches 105 APS at step 56 and 202 at 77, but 12 refusals and repeated zero-transfer inserts limit the finish. No drill contributes. |
| **rainbow-optimisticper-n5; 27003 s1/e54, APS 1,316/171** | Specializes most sharply: one furnace, repeatedly picked up/replaced four times, is fed **1,078 stone and 200 coal**. It has only one craft and one research. APS reaches 201/401/804/1,208 at steps 41/71/115/154 and 1,316 at 170. This is a high-throughput stone-brick loop, not a drill strategy. |

Thus all four scores still come from hand-fed furnaces. B’s best policy is an iron-smelting shuttle; Rainbow is a cleaner stone-smelting shuttle; D is a weaker mixed wood/iron/stone line; the control overbuilds and churns furnaces.

## Rainbow versus dqn-per

Across the last 20 roots, Rainbow makes 81.5% of decisions HARVEST or INSERT (48.8%/32.7%), versus 34.1% for dqn-per (16.3%/17.8%). Control spends 12.3% PLACE, 11.3% WAIT, 11.3% EXTRACT and 10.5% RESEARCH; Rainbow spends 4.3%, 0.9%, 4.5% and 1.0%. Total refusal is **9.11% Rainbow versus 12.76% control**: Rainbow has more tool refusals (8.16% vs 3.05%) but almost eliminates `no_support` (0.95% vs 9.71%). APS/decision is **4.675 versus 0.753**, a 6.2× ratio. Its advantage is therefore substantially a different, concentrated strategy, with lower waste as a secondary gain—not merely the same plan executed more cleanly.

The optimistic sampler is operationally stable but not stationary. From resumed env-step 6.1k to 17.9k, positive-delta fraction moves 0.463→0.583, mean-max-Q 7.18→115.49, and TD loss 0.94→6.33 as returns rise. After 10:33, four points are bounded at positive fraction 0.566–0.587, Q 91.98–106.57 and loss 5.79–6.30. Positive priorities are about twice negative priorities (10.52 vs 5.52 latest), as designed. There is no collapse or oscillatory explosion, although growing Q/loss means “stable” should not be read as converged.

## UCB behavior and forcing

Explorer fractions fall with counts but jump at resumes. B’s 2k-action blocks go 28.4%→15.9% before 05:50 and 21.1%, 23.5%, 16.0%, 18.0% afterward; its current short segment is 24.8%. D goes 41.2%→30.4%, then 27.9%, 24.7–26.0%, and 18.4% before forcing lifts the last block to 29.2%; current is 28.2%. Typical selected bonus versus absolute standardized Q is now 0.115 vs 2.50 (B) and 0.341 vs 2.10 (D), while unseen/forced bonuses remain 2.13–2.27 and can dominate Q.

The documented coefficient has **not** remained decayed across restart: latest `ucb_c` is 0.746/0.748, after reaching 0.640 for B and 0.707 for D. `behaviour_steps` evidently resets although counts resume. Across the complete files B visits rung keys −1,0,2,3,4,5 (five ladder rungs, missing 1); D visits −1 and all six ladder rungs. In the current segment B reaches 0,3,4,5; D only 0.

Exact “n=0 remaining” cannot be recovered: each action row logs `n` only for the chosen pair, and `admitted_indices` only for the chosen operation’s heads, not the full candidate set. Every logged n=0 choice ceases to be zero immediately. Observable current first tries include B rung-3 `(PLACE, drill)`, rung-4 `(INSERT, stone/iron-plate/coal)`, and many rung-5 inserts; no chosen pair remains zero. This is a telemetry gap, not evidence that every admissible pair has been covered.

Forcing did fire: **21 actions on each explorer**. D chose two rung-3 drill PLACEs (both `place_no_supported_offset`), then one rung-4 INSERT wood and 18 rung-5 inserts. Post-restart B chose PLACE drill at rungs 3 and 5 plus 19 INSERT choices. Crucially, many chosen items were stone, plates, furnaces or boilers; the old item clamp executed coal instead. The apparent fuel success is not the sampled policy under the owner’s rule.

## Every drill craft since 05:50

`R/X` means root/excursion; `near` counts the following 30 positions within 10 tiles of an ore tile observed in that same episode. Fuel is coal+wood reconstructed from deltas (`>=` for restored excursions).

| port/session/episode, craft step | fuel; near | next 30 steps |
|---|---:|---|
| 27004 s1/e36 R, 130 | 60; 15/30 | no drill PLACE or INSERT |
| 27000 s1/e18 R, 65 | 75; 26/30 | PLACE +2 and +30 both `ok`, but offsets 136→144 and 128→162 (`place_footprint_or_reach`); no drill INSERT |
| 27001 s2/e0 R, 171 | 15; 30/30 | PLACE +1 `ok`, 128→124 clamp; 19 executed coal inserts, only four actually selected coal; PLACE +19 `ok` at unchanged 138→138 |
| 27002 s2/e7 R, 76 | 65; 16/30 | no drill PLACE or INSERT |
| 27005 s1/e2 X, 41 | >=0; 3/30 | PLACE +16 `no_support`, offset 181, agent 80 tiles from observed ore |
| 27005 s1/e3 X, 41 | >=0; 15/30 | none |
| 27005 s1/e3 X, 42 | >=0; 15/30 | none |
| 27005 s1/e4 X, 41 | >=0; 18/30 | none |
| 27005 s2/e1 X, 89 | >=0; 28/30 | none |
| 27005 s2/e4 X, 62 | >=0; 15/30 | none |
| 27005 s3/e6 X, 47 | >=0; 1/30 | PLACE +1/+2 both `place_no_supported_offset`, offset 232, agent 96 tiles from ore |

Overall, 182/330 positions (55.2%) were within reach, and every window reached ore at least once. Of seven sampled PLACE attempts, three were genuinely out of range, three succeeded only after D12 relocation, and **one** succeeded unchanged. No listed attempt has `approach_tiles`, so D15 did not fire in these windows. Unrelocated placement was feasible often, but the sampled full action was correct only once; D12 materially overstates policy competence.

## Loops, waste, and remaining defects

Since 10:33, per-port shares are:

| port | steps in adjacent PLACE/PICKUP pairs | repeated identical refusals | RESEARCH (longest run) |
|---|---:|---:|---:|
| 27004 | 6.8% | 10.6% (14) | 14.2% (18) |
| 27000 | 7.1% | 6.7% (8) | 5.3% (5) |
| 27001 | 10.6% | 0% | 1.1% (1) |
| 27002 | 3.2% | 0% | 8.1% (6) |
| 27005 | 4.8% | 0% | 4.0% (1) |
| 27003 | 1.6% | 0% | 0.8% (1) |

Three defects remain visible. First, `extract_entity_has_no_items` repeats despite being a deterministic no-support case (80 current control steps), so the op/entity/item combination is not effectively masked or cooled down. Second, drill fuel is exposed to EXTRACT although the tool cannot extract that slot: B’s fuelled drill returns `nothing_to_extract`. Third, CONNECT can consume poles (`items_delta` −1/−6) while the effect predicate reports `predicate_false`; that is a verifier/tool disagreement. The many positive rewards on refused calls are not themselves defects—furnaces advance during elapsed ticks—but `nothing_to_extract` after approach is a continuing stale-content symptom.

## Ranked recommendations

1. **Force full drill actions, without relocation.** At the drill rung, sample PLACE offsets/directions from visible resource-grid offsets; if none exist, preserve and fail the sampled action. After an actual placement, sample only genuinely held fuel items. Never substitute target, anchor or item.
2. **Persist UCB behavior time across resumes.** Continue `c` from checkpoint and cap the forcing window by distinct full `(op, primary, secondary)` trials, preventing restart-driven re-exploration and repeated invalid INSERT variants.
3. **Add a low-exploration exploitation stream for Rainbow.** Preserve its optimistic replay, but reduce late behavior noise so the learned harvest/feed loop is measured and reinforced rather than diluted. Use masks/cooldowns only for proven deterministic refusals such as empty extraction; leave game-contingent failures learnable.
