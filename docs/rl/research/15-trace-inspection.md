# Post-D6 trace inspection: the drill is still behind the quantity head

## Scope and counting

I treated the latest appended segments of arm B and arm D as the post-D6 restart, and aligned the untouched QR/control streams to 18:35 using their resumed learner clocks (approximately learner step 6,100 for QR and 63,900 for control). Episode identifiers below are therefore `port/episode` as written in the files, not merged dashboard indices. Inventory was reconstructed by accumulating `items_delta` from an empty root reset. The loop denominator includes every paired step available at inspection time; the drill-probability denominator uses 74 completed root episodes (80 root trajectories including six in progress).

A “loop step” is a zero-reward step belonging to at least one mechanically reproducible run: an adjacent PLACE/PICKUP pair for the same prototype, a run of at least two identical rejected `op+args`, or at least two consecutive RESEARCH actions. This deliberately does not label every zero-reward action a loop.

## What followed every post-fix gear episode

There were four such episodes, containing six successful gear crafts:

| trace | successful gear step(s) | next 40 steps | provenance and interpretation |
|---|---|---|---|
| B UCB `27001/e0` | 26 (`x1`), 38 (`x1`) | No drill. After step 26, 26/40 actions were CRAFT. Steps 27--32 repeated `iron-gear-wheel x5`; steps 39--41 repeated `transport-belt x5`; all failed because the required iron plate “cannot be crafted (category: smelting).” | The gear choices were raw-Q greedy, not exploratory: standardized Q 3.275 at step 26 and 2.964 at step 38. In the window, 32/40 actions were logged non-exploratory. The primary-pair value kept preferring gear (Q 3.150--3.255 at steps 27--32) while the unmasked greedy quantity made that preference unexecutable. Only two gears were made, so drill never had literal inventory support. |
| D UCB `27005/e4` | 103 (`x1`) | No drill. The next 40 were 11 INSERT, 11 EXTRACT, seven RESEARCH, four HARVEST, four CRAFT and three MOVE. Examples are CRAFT furnace at 106, INSERT furnace into a belt at 108, and rejected EXTRACT plate at 114 (`extract_item_not_in_entity`). | All 40 were marked exploratory by the UCB diagnostic. The gear itself was an unseen pair (`n=0`), standardized Q 0.697, bonus 2.033. Count-driven breadth, not a learned drill suffix, dominated. |
| QR `27003/e32` | 129 (`x1`), 138 (`x1`) | No drill. The first window contains 12 RESEARCH, five CRAFT, five WAIT and four PLACE; steps 142--149 include radar/radar, two 60-second waits, automation, radar/radar. | Both gears were marked exploratory, with high recipe-pair Qs 3.975 and 3.859. The episode only reached two held gears and never simultaneously held the full drill inputs, so the support mask withheld drill while the high-epsilon joint behaviour wandered. |
| QR `27003/e33` | 70 (`x5`) | No drill. It instead did 17 INSERT and six PLACE. Steps 71--73 insert `iron-ore x5`; steps 81--82 try `iron-plate x5` into a furnace already containing ore and receive that exact refusal. | The gear was exploratory and had standardized Q -0.001. Although five gears existed, the trace never simultaneously had three plates and a furnace in player inventory; placing/feeding furnaces consumed or moved the other requirements. |

Thus no policy attempted drill CRAFT in the immediate 40-step suffix. The action logs record Q only for the chosen primary pair, so they cannot supply a counterfactual drill Q while drill is masked. They do show why the alternatives win: B has a strong but quantity-blind gear value; D pays UCB bonuses for broad unseen pairs; QR’s successful gears themselves are exploratory and are followed by familiar furnace/production actions.

## Arm D restored excursions

Port 27005 contains four 64-step excursions, 256 steps total, all restored at source step 103 after the first gear. They did not behave like focused drill suffix searches:

- Excursion `e6` used 26 INSERT, 14 RESEARCH, 13 CRAFT and ten EXTRACT. It did try `CRAFT burner-mining-drill x20` at step 107. That was an unseen rung-2 UCB pair: standardized Q **-0.410**, bonus **+2.037**, `exploratory=true`, but `secondary_heads_sampled=false`. It failed deterministically: “Failed to craft 20x ... iron-plate ... cannot be crafted.” This is the only post-fix drill admission.
- `e8` spent 16 CRAFTs on firearm magazines; steps 104--108 are identical `x5` refusals for missing smelted plate. `e10` mostly inserted, moved and rotated. `e11` placed 18 furnaces and picked up five; steps 109 and 111--114 retry `(22,-200)` against `tree-05`, while steps 124--125 retry `(8,-186)` against `tree-07`.
- No excursion attempted PLACE with a drill. No chest-crafting spree occurred. Under the three requested categories, 33/256 steps (12.9%) were loop steps: six in adjacent place/pickup pairs and 27 RESEARCH steps. Adding identical rejected-action runs raises the union to 36/256 (14.1%); 28 excursion steps belonged to such refusal runs.

The restore itself is sound—the traces continue from the gear state—but the snapshot is too early and UCB spends most of the budget rediscovering unrelated legal primaries.

## Placement, drill probability, and the anchor

Since D6, **zero** actions have had `PLACE placeable=burner-mining-drill`. Consequently the new character/footprint offset support has not been exercised by training. Historical evidence still diagnoses the old bug: the nine pre-fix drill-craft episodes produced 55 drill PLACE attempts, all refused. For example, UCB `27000/e75` repeats target `(30,-66)` 24 times; UCB `27001/e56` gets the same “something is in the way or terrain is unplaceable” refusal at `(26,-78)` 19 times.

Across the 80 observed post-cutoff root trajectories, no inventory reconstruction ever simultaneously reached `iron-gear-wheel>=3`, `iron-plate>=3`, and `stone-furnace>=1`; drill crafts were 0/74 completed episodes (empirical 0%, one-sided 95% upper bound about 4.0%). This literal count is stricter than the recursive one-unit recipe mask: D’s excursion proves the recipe primary can be admitted from recursively craftable ingredients, but the independent quantity head can immediately invalidate it.

Loop shares were: B **21.4%** (epsilon port 15.5%, UCB 27.3%); D **37.3%** (epsilon 32.4%, UCB 42.1%); QR **14.2%**; control **10.0%**. They do concentrate on UCB: 64% of B’s loop steps and 56.5% of D’s occur on the UCB port despite approximately half the steps being there.

The control’s failure is preference, not plate access. In the inspected 13 root trajectories it held at least two plates in three, each reaching 20 plates: `e52` EXTRACTs `iron-plate x20` at step 195, `e54` at 99, and `e59` at 58. It never spends a plate. All 107 CRAFT actions target `stone-furnace` (78 are rejected `x20` requests); none target gear. After `e54:99`, it resumes HARVEST/INSERT, moves the furnace at 107--108, feeds coal/ore at 109--111, and WAIT at 112 earns +15. That is the learned high-score hand-fed furnace basin: plates are terminal inventory, while ore insertion and waiting keep paying APS.

## Ranked, support-only changes

1. **Make quantity support recipe-aware.** For CRAFT, admit `x5/x20` only when the recursive ingredient calculation covers that quantity; do the analogous held/slot-capacity masking for INSERT and EXTRACT. This single support fix removes B’s step 27--32 gear refusals, D’s drill `x20` failure, magazine spam, and the control’s 78 furnace-`x20` refusals.
2. **Guarantee a valid secondary on a first-time UCB pair.** A new `(rung, CRAFT, recipe)` should use the smallest supported quantity (normally `x1`), not merely resample secondary heads with probability 0.3. D paid a +2.037 first-try bonus for drill but paired it with an impossible greedy `x20`; that exploration credit was wasted. If quantity masking cannot land first, key CRAFT counts by `(recipe, quantity)`.
3. **Snapshot when drill CRAFT first enters support, then force one supported try.** The present first-gear snapshot had one gear and produced four broad excursions but only one invalid drill request. A state-derived “drill recipe admitted” frontier is still agent-discovered, yet starts at the actual missing suffix. Keep the 64-step budget initially; spend action one on the unseen supported drill pair before widening to UCB.
4. **Add short support cooldowns as loop breakers.** After a deterministic refusal, mask that exact full action until state changes; after PLACE/PICKUP, briefly mask its immediate inverse; suppress repeated RESEARCH of the same technology absent progress. The 21--37% hybrid-arm loop shares, UCB concentration, and exact tree/quantity retries justify this before increasing excursion budget.
5. **Only then enlarge excursions or add per-rung forced tries.** If a valid forced drill attempt still fails to occur after the better snapshot, allocate 128 steps or one forced unseen primary per rung. More budget now would mostly scale D’s INSERT/RESEARCH/refusal traffic, not test placement.

These changes alter support and acquisition only; none changes rewards, TD targets, or milestone bonuses.
