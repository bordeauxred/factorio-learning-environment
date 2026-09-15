# Environment modelling for factory construction

## Decision and scope

Replace the shared coordinate-and-anchor action tuple with a **parameterized macro-action interface whose legal arguments depend on the operation and all preceding arguments**. HARVEST selects a known harvestable target and includes navigation. PLACE retains prototype, reference, local offset and direction. Entity references are one reference type, alongside player and terrain references. The observation exposes the exact geometry and item relations needed to complete those actions. Reward is the signed change in FLE's automated production score. A step is a completed or explicitly interrupted macro at an authoritative simulation boundary.

This specifies an environment, not a learner. It makes resource acquisition and factory construction expressible through supported actions; it does not establish that an arbitrary learner will discover a productive factory. No algorithms, exploration methods, curricula or training procedures are proposed. This document is the only requested change. No simulator was run. Repository statements below cite the current checkout; historical measurements cite their reports. Proposed contracts are requirements, not claims of existing functionality. Runtime properties not established here are marked **UNVERIFIED**.

The design intent in `docs/rl/DESIGN.md:47-81` is sound about macro-actions, typed inventories, two spatial resolutions and relative placement. Its state-only marginal masks are not sufficient. The earlier critiques identify complete-action support and missing action-relevant relations (`docs/rl/research/08-action-space-rethink.md:46-54`; `docs/rl/research/09-observation-rethink.md:23-39`). Adopt those environment findings without adopting their learner or experimental recommendations.

## 1. What failed

### Action space: the first material transition is almost inaccessible

**The single most responsible defect is HARVEST's target contract: it presents arbitrary nearby coordinates as a resource-acquisition macro but does not move into harvest reach before calling the server.** This cuts off the first item, and therefore every subsequent construction operation. It is more causally important than the empty-entity errors, although those consume more decisions.

The dispatcher calls `harvest_resource(position, quantity)` directly (`fle/rl/action.py:90-97,244-249`). The server checks distance to that position against the actual character's `resource_reach_distance` before resolving resources (`fle/env/tools/agent/harvest_resource/server.lua:460-478`). The client comment claiming fast mode ignores reach is contradicted by this unconditional Lua gate. Its movement/retry logic occurs after the initial call and error handling, so cannot rescue that initial rejection (`fle/env/tools/agent/harvest_resource/client.py:39-45,55-89`). The server gate is verified; **2.7 as the effective live value in every historical run is UNVERIFIED**. It is the audit's default-value assumption and is now also hard-coded in the mask.

| Historical evidence | Environment consequence |
|---|---|
| 0 successful harvests in 6,752 unscripted attempts across nine runs | No observed first material transition. |
| 1,797/1,800 harvest errors say “Nothing within reach to harvest” | Navigation was not part of the acquisition macro. The message also covers missing targets, so the text alone does not distinguish every rejection's branch. |
| Only 21/289 integer offsets fall within radius 2.7 | The shared `[-8,8]²` coordinate space wastes 92.7% of HARVEST offsets even before checking resource presence. This lattice calculation assumes a player-centered integer offset, not arbitrary real-entity anchors. |
| The other three harvests fail on the negative quantity limit | The `-1 = all` bin destroys the remaining opportunities. It reaches `find_entities_filtered(limit=count)`. |
| 10,835/21,540 steps, 50.3%, fail with “real entity anchor required” | Six operations are offered with no real entity in the world. |
| Outcomes are 19.3% `ok`, 30.4% tool-rejected, 50.3% preflight-invalid | A nominally large action space mostly describes refusals. `ok` is not necessarily a productive effect. |
| WAIT uses 73.8% of measured wall time; HARVEST uses 2.1% | Action duration and idle-time support compound the geometry problem. |

Sources: `docs/rl/research/11-cracking-exploration.md:19-122`; negative quantity reaches the filter at `fle/env/tools/agent/harvest_resource/server.lua:259-269`. The user reports 186 PPO episodes with entropy bonus, automated score 0.00 and zero entities built. The stored run confirms 186 training episodes; its last training and evaluation episode records also show zero score and empty built/automatic-item sets (`docs/rl/results/ppo_open_play.jsonl:50989,54074-54075`). This is historical run evidence, not a new experiment or proof about every possible learner.

The offline audit finds resource presence in 362/21,540 fine views, 194 offset windows, and only 95 harvest-reach neighborhoods. At those 95 boundaries, an average of 11 of 289 offered offsets hits reachable ore. These multiply to about `1.05e-5` successful harvests per action, or 0.23 expected in 21,540 actions (`docs/rl/research/11-cracking-exploration.md:136-166`). Improving only the reach mask leaves resource access dependent on a separate accidental walk.

The present checkout has partial repairs. `masks.py` removes far offsets and negative quantities only when explicitly called with `operation="HARVEST"`; it does not test resource presence, actual world distance from a selected anchor, inventory, or whether an operation has a complete legal tuple. The usual observation encoding calls it without an operation. All other argument masks begin as ones, and index zero is required to remain enabled (`fle/rl/masks.py:8-55`; `fle/rl/observation.py:552-563`). Thus these repairs do not satisfy aggressive invalid-action masking. The schema now uses a joint 290-entry offset head and 64 entity rows, not the older independent axes and 512 rows (`fle/rl/schema.py:25-31,133-154`).

### Observation: visible records do not supply the required decisions

The current fine grid preserves five ore identities. The coarse grid has no resource identity, water or obstacles, so resources outside the fine crop cannot guide a regional resource action. Distant entity-relative coordinates clip at 64 tiles, and the player-centered fine grid does not show the neighborhood of every distant entity offered as a placement anchor (`fle/rl/observation.py:381-409,410-466,487-488`).

Buildings occupy only their center cell in the fine grid. Exact placement needs the rotated collision footprint, neighboring ports and terrain clearance. Entity item and fluid identities are parsed into the cache, then reduced to totals in entity features; the fuel feature counts coal specifically. This hides whether a machine contains ore, plates, another ingredient or another fuel (`fle/rl/observation.py:65-71,423-441,494-504`). Tree and rock records lack exact target identity; rocks and cliffs share a terrain tag (`fle/cluster/scenarios/open_world/observation_diff.lua:239-244`). A generic obstacle pixel is not a harvestable-resource description.

The live implementation already selects entities by strata, carries explicit player inventory and technology bits, and excludes the character from real targets. Preserve these repairs. Remaining deficiencies include truncated inventory/flow tables, absent current-research identity in the encoded globals, a permanently zero stock feature, unfilled power lanes, and no action budget feature (`fle/rl/observation.py:287-370,514-551`). The vocabulary loader already rejects technology counts above its capacity; do not repeat the earlier claim that every such overflow is silently accepted (`fle/rl/vocabulary.py:338-339`).

### Reward: several incompatible contracts coexist

The old design uses item-count goal attainment, which is not the requested priced net production objective (`docs/rl/DESIGN.md:83-89`). The current base environment returns zero for every counter-valid transition. The open-play runner ignores that reward and separately queries the automated score (`fle/rl/env.py:377-397`; `tests/benchmarks/run_ppo_open_play.py:280-289`). Directly consuming the environment's return would therefore produce an all-zero reward even if a factory worked.

Gross automatic output counts are also not automated PS: they omit consumption, prices and the manual-crafting input correction. The protocol's automatic counters subtract manual outputs and clamp to zero; they cannot reconstruct the signed score (`fle/cluster/scenarios/open_world/observation_diff.lua:434-450`). A report verifies a fuel transition of -3 followed by production of +40. Negative rewards for inputs are part of the objective, not an invalid-action penalty (`docs/rl/results/production_score.md:19-44`). The reward has a legitimate zero-reward manual bootstrap interval. That fact does not explain why the first harvest itself never executes.

### Step contract: duration, effects and observations disagree

The wrapper unpauses before preflight validation. Even impossible tuples can advance the factory, and WAIT ends after a host polling loop detects the tick target. Non-WAIT clients mix synchronous mutations, synthetic `storage.elapsed_ticks`, wall sleeps and queued work (`fle/rl/env.py:315-357`; `fle/env/tools/agent/move_to/client.py:53-108`; `fle/env/tools/agent/craft_item/client.py:30-61`). Logging true elapsed ticks is necessary but does not remove host-latency-dependent transitions.

Success cannot mean “a truthy return.” An INSERT into a full belt group can return its unchanged target; harvest can fall back to another resource type; SET_RECIPE searches nearby and can return a serialized entity without verifying the requested recipe took effect (`fle/env/tools/agent/insert_item/client.py:56-87`; `fle/env/tools/agent/harvest_resource/server.lua:494-529`; `fle/env/tools/agent/set_entity_recipe/server.lua:8-42`). Historical zero-output CRAFT successes are already repaired for integer returns at `fle/rl/action.py:179-189`; complete effect verification remains missing.

Pausing and draining is already implemented, but draining does not make all cached data current. Quantized inventory signatures can suppress changes even after a check; only the nearest 25 entities receive repeated priority checks, with the rest on a rotating slice (`fle/cluster/scenarios/open_world/observation_diff.lua:152-170,308-367`). This is a partially observed process, not an exact Markov state merely because the result is a fixed float array.

## 2. Action specification

### Typed grammar and support

An action is `Action(boundary_id, operation, arguments)`. Keep the twelve operation names. Replace shared active heads with the following tagged argument records. Arguments omitted by an operation do not exist in its logical action. Padding zero may appear in serialization but is never a selectable value for a required argument.

Let `L(s)` be the set of complete tuples supported at boundary `s`. The operation mask is `op enabled iff some (op,args) is in L(s)`. After any argument prefix, the next mask contains exactly values with at least one surviving complete completion under the declared admission checks. Pairwise marginal unions do not meet this requirement. The interface supplies a deterministic conditional legality object, not a flat ID for every complete tuple. This is an action contract and makes no assumption about how a learner represents or chooses actions.

| Operation | Arguments, in dependency order | Admission and completion |
|---|---|---|
| HARVEST | harvest-target, quantity | A visible typed target with positive manual yield, no unmet mining prerequisite, inventory capacity and an approach route that is not known impossible. Complete when the requested amount is acquired, the selected target is exhausted, or the macro deadline interrupts it. |
| MOVE | destination | Select a visible regional cell, known landmark, or local site. The environment chooses a walkable arrival point and exposes it. Complete on arrival. No laying or leading side effect. |
| CRAFT | output-item, quantity | The existing recursive hand-crafting macro, with enabled recipe closure, character categories and shared ingredient accounting. Complete on actual output receipt, not queue submission. |
| PLACE | prototype, reference, offset, direction | Inventory-backed exact placement with footprint, collision, terrain and orientation support. Includes approach. Complete when that prototype exists at the resolved center and orientation. |
| PICKUP | entity | Recover the selected entity and its recoverable contents subject to inventory capacity. Includes approach. Do not silently pick up a whole connected group. |
| ROTATE | entity, direction | Only supported orientations different from the current orientation. Includes approach and verifies the final orientation. |
| INSERT | entity, slot-role, item, quantity | Exact receiver and destination inventory/line, accepted item and available capacity. Includes approach. Complete on verified positive transfer. |
| EXTRACT | entity, slot-role, item, quantity | Exact source inventory/line with available contents and player capacity. Includes approach. Complete on verified positive transfer. |
| SET_RECIPE | entity, recipe | Machine category and enabled recipe, different from the current recipe. Includes approach and verifies the result. Does not overload a recipe ID as an inserter filter. |
| CONNECT | connector-family, material-profile, source-port, target-port | Compatible distinct ports and supported routing materials. Route and build through FLE's existing macro. Completion means the requested connection exists, with all changed segments reported. |
| RESEARCH | technology | Enabled, unresearched, prerequisites satisfied, different from the current selection. Complete when selection changes. It does not wait for the research to finish. |
| WAIT | duration | Advance simulation for the specified game ticks. Always retain a short duration; long durations require a possible autonomous change as defined below. |

At the empty start, PICKUP, ROTATE, INSERT, EXTRACT, SET_RECIPE and CONNECT have no support. PLACE has none without a held placeable. CRAFT has none without a feasible recipe. RESEARCH may still be legal without labs: choosing a technology is a real state change and the server does not require science infrastructure (`fle/env/tools/agent/set_research/server.lua:5-39`). Do not mask it merely because it yields no immediate score. No operation is unlocked by episode number or prior success.

Admission distinguishes **structurally supported**, **server-certified at this boundary**, and **path/geometry uncertain**. Missing rows, incompatible slots and negative quantities are never uncertain. Pathfinding or collision checks may still reject a structurally supported action. Refresh selected inventories, targets and affected geometry while paused; use batched checks for the offered reference windows and held prototypes. Do not make one network request for each Cartesian-product tuple. Route search remains a chosen-action operation because it is compound work (`docs/rl/research/01-action-space.md:56-80`; `fle/env/tools/agent/connect_entities/client.py:161-198`). Actual cost and certification coverage are **UNVERIFIED**.

### HARVEST acquires the first resource in one decision

`harvest-target` addresses a resource patch, an individual tree, or a mineable rock in the observation. It carries a persistent identity, resource/product IDs, exact representative target position, extent, yield/capacity facts, sample tick and approach status. Ore tiles need not have a player-entity `unit_number`. Use a surface, resource name and persistent patch identity, with exact tile coordinates for execution. Depletion or splitting versions that identity rather than reusing it for another resource.

Execution is: resolve the chosen target; find a non-colliding working position; navigate there; re-read the actual player position and reach; harvest the chosen resource; repeat within the selected patch if necessary. Require `distance(player, next_target) <= actual_resource_reach_distance - epsilon` before every harvest call, with `epsilon=min(0.1 tile, reach/10)`. Use the position actually returned after navigation, not the requested waypoint. `move_to` currently rounds destinations to quarter tiles, requests a path and can return before slow-mode walking is complete, so both arrival and queue completion require checking (`fle/env/tools/agent/move_to/client.py:29-49,91-108`; `fle/env/tools/agent/move_to/server.lua:40-77`).

This preserves the server's reach rule. Reach becomes an internal execution invariant, not a coordinate the policy must guess. It does not guarantee that a path exists. An inaccessible target yields `path_failed`; depletion yields `target_exhausted`; an interrupted trip may yield a changed player position without any harvested item. These outcomes must not be labelled successful resource acquisition.

Do not simply prepend MOVE to the unchanged harvest call and call the contract finished. Lua currently resolves any tree/resource/simple entity within 1.5 tiles, searches around that target, and can fall back to other types. The new adapter must bind the selected type and patch and prevent cross-type fallback (`fle/env/tools/agent/harvest_resource/server.lua:245-279,494-522`). It must honor mining prerequisites and actual accepted inventory output. Uranium, fluids, cliffs and other targets are not admitted as hand-harvestable merely because they have resource or obstacle pixels. Their live manual-yield eligibility is **UNVERIFIED** until the prototype predicate agrees with execution.

Expose all known targets, including those outside the current player crop, through the landmark view described below. Thus a visible stone patch 40 tiles away can produce the first stone from one HARVEST decision. No real machine anchor, blind coordinate guess or separately selected walking action is required.

### Placement keeps relative geometry, without inventing an entity

`reference` is a tagged union: `PLAYER`, `ENTITY(unit,generation)`, or `TERRAIN(tile_x,tile_y)`. Terrain references come from visible local/regional cells and resource landmarks. They are coordinates, never synthetic machine rows eligible for INSERT or CONNECT.

For PLACE, retain integer `dx,dy in [-8,8]` as one 289-cell offset. The selected reference defines a 32×32 one-tile crop. The policy can inspect that crop within the same frozen observation before completing the action. This inspection is an observation accessor, not a simulation step. Entity references use the entity center; terrain references use the selected tile center; PLAYER uses the actual player position. Resolve the final center using the rotated prototype's legal lattice and expose that exact resolved coordinate in the offset/direction support record. Never relocate it after selection. Legal-center rules are exported and checked, not inferred solely from odd/even width. The current arithmetic is at `fle/rl/action.py:33-35,98-114`.

Relative placement is appropriate for “put the inserter one tile east of this machine.” It is inappropriate as HARVEST's universal target language or MOVE's regional geometry. MOVE selects a coarse cell or landmark; the executor resolves a walkable arrival point. HARVEST selects a typed patch; the executor resolves working positions. INSERT and EXTRACT select inventories, not offsets. CONNECT selects ports, not arbitrary entity-center pairs. The coordinate scale follows what the operation means.

All exact supported buildable sites in the offered reference window remain available, including strategically bad sites. Do not restrict drills to iron, orient them automatically toward chests, or require a future receiver for placement. Expose resource overlap and output-port geometry so the agent can make those decisions. Geometry masking removes impossible footprints, not unproductive designs.

Offshore-pump support is required for the power/fluids construction chain. Its existing fast placement forces `exact=false`, while the RL dispatcher rejects it entirely (`fle/env/tools/agent/place_entity/server.lua:193-198`; `fle/rl/action.py:99-101`). Supply exact shoreline site/orientation records and dispatch to that selected site without relocation. This is required new environment work, **UNVERIFIED** in the current tools. Do not advertise full power/fluids coverage while that action remains absent.

### Vocabularies and quantities

Pin complete item, entity, recipe, technology, resource, fluid, inventory-role, port and direction catalogs to a schema/version/mod hash. Observation IDs and action IDs share this manifest. PAD and OTHER are observation encodings only, never executable names. Use the checked-in live export as the starting artifact, not Python enum declaration order. Historical pinned counts are 104 placeables, 251 items, 217 recipes and 196 technologies (`docs/rl/research/08-action-space-rethink.md:44`); these are artifact counts, not universal limits. Current tool adapters still reject names absent from Python enums (`fle/rl/action.py:218-227,258-271`). Exported existence and supported dispatch are separate bits.

Use positive quantity requests **1, 5, 20, 100**, with no ALL sentinel. INSERT/EXTRACT requests above available source or destination capacity are masked. HARVEST quantities require inventory capacity and sufficient known yield when exactly known; exhaustion after partial yield is a recorded partial result. CRAFT quantities mean requested output units, rounded upward to the recipe's indivisible batch yield, with requested, planned and actual output counts all returned. Resource and recipe batch surplus is visible. Unknown or probabilistic yields carry explicit bounds and uncertainty. Recursive CRAFT includes its existing intermediate crafts, but adds neither mining nor smelting (`fle/env/tools/agent/craft_item/server.lua:79-109,130-159`).

Directions are named legal prototype orientations, mapped through the exported Factorio codebook. Four cardinal choices suffice for the currently supported construction families; symmetric duplicates may be canonicalized. Do not encode raw direction integers with an assumed angular denominator. The current entity encoder uses `direction*pi/4`, whereas the placement server explicitly discusses Factorio 2.0's 16-direction representation (`fle/rl/observation.py:489-490`; `fle/env/tools/agent/place_entity/server.lua:238-241`). Agreement of every prototype mapping is **UNVERIFIED**.

Slot roles distinguish fuel, machine input, machine output, module, chest and belt line. Only supported roles appear. The existing INSERT predicate has type-specific branches and nearby-substitute behavior; the new predicate and executor must share exact target/slot semantics (`fle/env/tools/agent/insert_item/server.lua:68-181`). In particular, its variable called `closest_distance` is distance from the requested position to the found entity, not distance from the player. Do not infer a uniform ten-tile player interaction rule from that code.

CONNECT retains automatic belt, pipe and power routing. Material profiles are supported sets of actual connector prototypes, such as a belt tier with its underground variant; do not always map a whole family to one basic prototype as the current adapter does (`fle/rl/action.py:272-278`). Port records include exact positions, direction, fluid identity/compatibility and electrical connection role. FLE already accepts positions, entities and multiple waypoints (`fle/env/tools/agent/connect_entities/client.py:136-177`). This contract initially chooses two ports and lets the existing router choose intermediate geometry. It does not turn a 40-tile connection into 40 policy decisions. Report the route, material use and partial construction. CONNECT keeps FLE's remote routing semantics; its server explicitly does not adhere to ordinary distance rules (`fle/env/tools/agent/connect_entities/server.lua:506-525`). Other local interaction macros absorb approach navigation, and their travel time is charged.

WAIT offers **60 ticks** unconditionally. Offer 300, 900 and 3,600 ticks when autonomous change is possible: production, transport, research work, pending queues, changing power/temperature, or other enabled scenario dynamics. Uncertain dynamics permit longer waits rather than being declared idle. Merely selecting research without labs is not research work. In a certified static empty world only the 60-tick wait remains. This removes the empty world's long-duration waste without declaring that waiting is always invalid. The legal short wait also guarantees nonempty action support. Long waits use actual simulation ticks, not host sleep duration.

## 3. Observation specified by those actions

Use a named, typed observation with **three content groups: tokens, spatial state and global state**. Conditional masks and argument facts are derived boundary data accompanying these groups. The authoritative environment state is the full simulator plus episode clocks, score baselines and macro bookkeeping. The bounded observation remains partially observed; do not claim it is a sufficient statistic for every factory.

### Tokens: entities and non-entity targets

Keep 64 selected real-entity tokens, with no player sentinel consuming a real slot. Fill deterministic strata: 24 nearest to player/focus, 16 by functional diversity, 12 fault-status, 6 spatially dispersed and 6 recently changed. Deduplicate, then fill unused capacity by distance and stable identity. Functional diversity uses exported categories, not a hard-coded iron objective. Preserve a complete unit-keyed cache. Existing 24/16/12/6/5 quotas provide the implementation basis (`fle/rl/observation.py:337-355`).

Each selected token contains identity/generation, prototype/type, exact position and footprint, named direction, recipe, status group, health/max-health, progress, fuel identity and remaining fuel energy, and sparse typed inventories/fluids with capacity and validity. Attach sparse ports and relations: inserter pickup/drop, drill output, belt direction/line, fluid ports and electrical adjacency. A total item count is insufficient for INSERT/EXTRACT. Exact counts and accepted-item support for selected targets are sampled at the decision boundary. Remote summaries may be aged, but their age is explicit and they cannot certify a quantity.

Supply a separate paged landmark table of resource patches, harvestable trees/rocks, regional factory groups and known map cells. Use 64 records per page, stratified by kind and spatial region, with deterministic ordering, total count and page identity. Pages are read-only views of the same frozen boundary snapshot, with no ticks or new information revealed by paging. This prevents a resource from vanishing from the action universe because 64 nearer trees filled a tensor. Exact harvest-target details accompany each offered target. A selected regional factory group can expose its entity page under the same observation-access contract. Every action-addressable entity must have a visible token before action submission.

Do not put arbitrary unit or electric-network IDs into continuous magnitude features. They identify records and connectivity relations. Physical power features are energy/capacity fractions, supply and demand in watts with declared scales, and satisfaction only when measurable. Until validated, use missingness plus known connectivity/status, not fictitious watts. The current protocol explicitly emits `V0` (`fle/cluster/scenarios/open_world/observation_diff.lua:538-540`).

### Spatial state: fine precision wherever placement is offered

Keep a **32×32 fine grid at one tile per cell**, player-centered by default and reference-centered for PLACE. Use 14 channels: known, water, impassable terrain, tree occupancy, mineable-rock occupancy, five named ore layers, transport/inserter footprint, machine footprint, fluid-structure footprint and power/storage footprint. Resource presence and amount bucket are distinct facts. Include sparse exact target coordinates and collision polygons, because a tile grid cannot represent sub-tile centers or every rotated footprint. Per-prototype/direction placement support is a derived bitmap over the 289 offsets, with exact resolved centers and resource/port overlap facts.

Keep a **32×32 coarse grid at four tiles per cell**, covering 128×128 tiles, with 14 channels: known fraction, water fraction, obstacle fraction, five typed ore occupancies, harvestable wood/rock density, functional-entity density, working fraction, no-power fraction, input/fuel-starved fraction and output-blocked fraction. Status fractions use eligible functional entities as their denominator. Long-range landmarks retain exact world coordinates plus distance and bearing, rather than clipping all distant objects to the crop edge.

The persistent sparse tile map supplies these crops at negative as well as positive coordinates. A reference-centered crop is available before choosing its offset, from the same observation snapshot. It cannot query new live facts between argument choices. This resolves the remote-anchor visibility defect without expanding every action to a world-sized coordinate head. If a required footprint extends beyond the crop, supply its complete sparse geometry or mask that particular site; do not assume the unseen part is empty.

Adopt the protocol's existing information rule explicitly: **all generated and delivered chunks are known, even if unvisited**. Full sync already enumerates generated chunks (`fle/cluster/scenarios/open_world/observation_diff.lua:627-635`). Unknown chunks remain unknown. MOVE can target a known boundary cell and reveal subsequently generated chunks; no hidden nearest-resource oracle is part of HARVEST. Resource access probability therefore depends on generated-map coverage, not just player reach. Actual initial generated-region resource coverage is **UNVERIFIED** from the reported fine-grid decode.

### Global state and static facts

Include exact full player inventory by item ID and capacity, researched/enabled bits for all technologies, current research ID/progress, enabled recipes, actual interaction reaches, episode tick, remaining tick budget, daylight/scenario dynamics relevant to WAIT, and last macro outcome with requested/realized quantities and duration. Use dynamic catalog lengths or sparse records rather than silently cutting off kinds at 128 or technologies at 256.

Include produced and consumed flows for every active scored item/fluid, both raw and manual-adjusted where defined, over an explicit **3,600-tick trailing window**. Store timestamped count changes, normalize by the covered ticks during warm-up and expose window coverage. Existing code instead derives rates between slow samples and clips at 60/min (`fle/rl/observation.py:251-265,529-531`). Do not call those rates an exact trailing-minute window. Current score, cumulative raw/manual ledger values and reward endpoints are boundary-exact and do not wait for the flow tier.

Expose static recipe ingredients/products, batch amounts, categories, craft times, unlocks, footprints, mining prerequisites, fuel classes and score prices through the same pinned manifest used for legality. Integer identities and exact counts remain typed integers; fluid quantities and score arithmetic retain source precision. Optional compressed policy magnitudes use declared log scales with an overflow indicator, never silently replace quantities needed by actions. Entity energy is normalized by capacity; health by that prototype's maximum. A missing measurement has a validity bit. Delete zero-filled promises of surveyed stock until an actual coverage-defined stock measurement exists.

### Required protocol delta

Keep the combined entity/terrain drain, named resource map, change events, freshness acknowledgments and full-sync recovery. The live checkout already adds inventory, technology, counters and name-plus-bucket resource signatures; these are not still missing as in the initial PR audit (`fle/cluster/scenarios/open_world/observation_diff.lua:375-382,492-540,590-636`).

Add boundary ID, server epoch, explicit agent/force/surface identity, schema/vocabulary hash, exact score and raw/manual ledger endpoints, exact selected-target inventory/slot support, typed harvest targets and physical geometry. Keep coarse remote drift updates sparse, but synchronously refresh the selected action facts and affected region while paused. A check of an unchanged quantized signature does not refresh an exact count. Missing character identity must be an error, not an apparently valid `(0,0)` player with empty inventory (`fle/cluster/scenarios/open_world/observation_diff.lua:396-431`).

Full sync after overflow replaces caches without changing the episode's time or reward origin. Currently `ObservationClient.apply(full_sync=True)` clears its local episode tick origin, while the server counter origin persists across ordinary full sync (`fle/rl/observation.py:155-176`; `fle/cluster/scenarios/open_world/observation_diff.lua:607-625`). Carry the authoritative episode start tick in every boundary so recovery cannot restart the apparent episode clock. Reset and resynchronization are different operations. This extends the same-call and freshness principles in `docs/rl/research/06-simulator-improvements.md:15-45,71-90`; the exact new payload sizes, latency and lab-scenario availability are **UNVERIFIED**.

## 4. Reward and episode contract

### One signed production objective

With the pinned price table `V`, use the requested objective:

`PS(t) = sum_i V(i) [P_i(t) - C_i(t)]`.

For the automated variant define cumulative manual harvest output `H_i`, manual craft output `O_i`, and manual craft input `I_i`:

`APS(t) = PS(t) - sum_i V(i) H_i(t) - sum_i V(i) [O_i(t) - I_i(t)]`.

All quantities are relative to the reset baselines. This follows the repository scorer's subtraction of harvested value and **net** manual craft value (`fle/env/tools/agent/score/server.lua:335-408`). Subtracting only `H+O` would leave manual inputs charged and change the automated objective. Manual mining/crafting alone should contribute zero automated value apart from the scorer's rounding; subsequent machine consumption and output count normally. Moving items between inventories or placing an entity must not fabricate production.

Set `r_k = APS(t_{k+1}) - APS(t_k)` in the environment itself. Report PS and APS separately. For exact compatibility, use the existing authoritative scorer's rounded APS endpoint and pin its price-table fingerprint; also expose the unrounded diagnostic sum. The scorer includes priced fluids as well as items and floors its totals, so an item-only counter vector is not a replacement for its published accessor (`fle/env/tools/agent/score/server.lua:309-330,391-408`). Do not independently floor per-transition contributions. Match the accessor's endpoint arithmetic.

Preserve negative deltas. Add no building bonus, first-item bonus, readiness potential, positive-only clipping, invalid-action penalty or per-step cost. Over an undiscounted finite episode, `sum_k r_k = APS(T)-APS(0)`. A duration-normalized reward or discounted sum changes that endpoint objective; neither is this environment's contract. Travel and idle time already consume the finite simulation budget. There is no terminal bonus on top of the final delta.

Reward correctness requires exact ledgers for partial actions and every recursive craft. The fast craft path computes batches, then calls its statistics helper with `crafted` item units even though the helper multiplies by recipe amounts as a craft count. Multi-output recipe accounting therefore warrants an explicit correctness check (`fle/env/tools/agent/craft_item/server.lua:112-127,145-147,186-196`). Complete ledger correctness across all tool paths is **UNVERIFIED**, even though the limited drill probe reports calibration. Never conceal disagreement by clamping derived counters. The same authoritative corrected ledger must feed observations and the published scorer.

### Step semantics

Use a finite-horizon semi-Markov environment: actions are parameterized macros with variable duration. The full state transition includes all autonomous factory activity during the macro.

1. Return an immutable paused boundary with exact tick, score, references and support. Resolve and validate the submitted boundary ID and complete tuple before advancing time. A malformed or masked tuple is an API error with no simulation transition; it cannot consume a decision budget as a hidden no-op. Structurally admitted uncertain actions may fail during execution.
2. Execute one macro, including required navigation and existing routing/recursive crafting. Let the factory run during travel and work. Charge a minimum of **one simulation tick** to a valid immediate mutation or execution failure. A path or craft consumes its actual execution ticks, not a fixed duration copied from another operation.
3. Stop at macro completion, explicit partial failure, the **3,600-tick per-macro limit**, or the remaining episode horizon, whichever comes first. The per-macro bound also applies to navigation and routing. Cancel outstanding walking/mining/crafting or build continuations before the next decision, retaining committed effects and returning/refunding uncommitted reservations according to the tool contract. Do not return control with an invisible macro still mutating the world.
4. Pause on the server, refresh changed targets and exact ledgers, drain one coherent boundary, calculate reward, then return. Include `start_tick`, `end_tick`, duration, selected identities, actual placement/route, transferred items, consumed materials, requested/actual quantities and an explicit result code.

The required time implementation is a server-controlled deadline/completion scheduler. Python polling must not decide how many bonus ticks an operation receives. Existing fast tools mutate immediately and sleep for synthetic elapsed ticks; slow harvest dispatch is commented out in Lua (`fle/env/tools/agent/harvest_resource/server.lua:368-379,481-483`). A compliant adapter must schedule those macro effects and simulation advancement consistently, including interruption at the horizon. Calling current fast tools and appending a duration field does not meet this requirement. Exact scheduling, cancellation and equivalence to the intended FLE macro semantics are **UNVERIFIED** and are required environment work.

Result codes distinguish `completed`, `partial`, `path_failed`, `target_changed`, `capacity_changed`, `macro_timeout` and `environment_error`. Partial means some requested effect or travel occurred, with the unmet part explicit. For harvesting, separately record `items_acquired > 0`; moving closer alone is not harvest success. For INSERT, verify transferred count even when the returned entity object is non-null. For ROTATE, track replacement identity if the underlying tool recreates the machine. Always retain actual partial mutations. A failed macro can have nonzero reward because unrelated machines continued producing; that reward belongs to the transition.

### Reset and end conditions

Use one fixed **216,000-tick horizon**, 3,600 game seconds, from a fresh empty-inventory, no-player-built-entities, default-research open-world start. Pin map-generation settings, spawn, agent identity, scenario, mods and score prices in reset metadata. This is a proposed task specification, not evidence that current resets already satisfy it. Restore terrain/resources and ledgers as well as entities; do not carry depleted patches or prior inventory into a nominal fresh episode. Reusing serialized terrain is permitted only when the restored world actually matches that serialization.

Validate a coherent spawn and available map/action information. Do not silently replace a seed because it is difficult. A map on which necessary resources are inaccessible remains identified as such; whether the configured map distribution guarantees a productive construction path is **UNVERIFIED**. In any start with a visible, accessible, manually harvestable target and room for its product, HARVEST must already be supported. If no target is known, MOVE and short WAIT remain, and later generated terrain can expose targets.

There is **no ordinary action-count cutoff**. The old 256-action cap can truncate a factory-building process at different simulated times depending on macro granularity (`docs/rl/DESIGN.md:35`; `fle/rl/env.py:380-388`). At the declared horizon return the final boundary and reward with `terminated=true`, because this is the task's finite horizon. Character death is also terminal in scenarios where death is possible; no silent character recreation inside a macro. A host timeout, transport failure or operator stop is `truncated=true` with the reason and validity of the last boundary, not a scored task completion. Wall-clock watchdogs abort malfunctioning execution rather than awarding production for host delays. There is no early termination for reaching an item threshold and no unreported final free-running production interval.

The action vocabulary supplies a physical bootstrap path: harvest stone, craft/place a furnace, harvest smeltable ore and fuel, insert them, wait for smelting, extract plates, then craft/place and supply production/logistics machinery. Quantities and prerequisites come from the pinned recipes, and geometry remains the agent's choice. This is a reachability argument, not an action script embedded in the environment. A complete successful trajectory under the new contract and horizon is **UNVERIFIED**. The specification must expose every necessary operation, including exact shoreline placement, before claiming broader factory coverage.

## 5. Comparable success probability

Use the audit's event, **at least one successfully acquired resource from a HARVEST action**, not all-operation `ok` and not automatic production. For the old trajectory:

`p_old = (95/21540) * (1/12) * (11/289) * (3/4) = 1.0492e-5`.

These factors are boundary coverage, operation selection, target selection and quantity validity. It is an empirical approximation: resource pixels and the assumed reach do not prove live mining eligibility. Its expectation is 0.226 acquisitions in 21,540 actions (`docs/rl/research/11-cracking-exploration.md:153-166`).

The directly corresponding new factorization is:

`p_new = q_known_target * p_H_given_target * 1_target_support * 1_quantity_support * eta_execution`.

Here `q_known_target` is the fraction of boundaries with an offered, visible harvestable target that navigation can service within the macro/episode budget. It replaces “player already within 2.7 tiles.” `eta_execution` is the residual probability of actually acquiring an item given the supported target and quantity, including path failures, stale/depleted targets, interruption and tool defects. The two support factors are **contractual 1s**, not measured success rates. If uncertain path status is included in the coverage count, charge path failure to `eta` instead; never charge it twice.

For a diagnostic uniform draw over **enabled operation families**, `p_H_given_target=1/K(s)`. This is a probability reference measure, not a proposed exploration procedure. At a normal empty start with a target, `K` is at most four: HARVEST, MOVE, WAIT and possibly RESEARCH. The other eight lack material/entity support. Uniform selection over all complete tuples gives a different answer and must not be called equivalent, since placement may have many more tuples.

The following arithmetic uses `K=4` and an explicitly **UNVERIFIED engineering target** `eta=0.95`:

| Coverage assumption | Per-action harvest probability | Relative to `p_old` | Interpretation |
|---|---:|---:|---|
| Old in-reach coverage only: `q=95/21540` | 0.001047 | 99.8× | About 22.6 expected acquisitions over the old 21,540-boundary count. Assumes those pixels describe eligible resources. |
| Only the old fine-window presence: `q=362/21540` | 0.003991 | 380× | About 86.0 expected acquisitions at that count, if navigation can service these targets. This isolates the benefit of absorbing local travel. |
| Known-map/landmark coverage `q=0.5` | 0.11875 | 11,318× | Sensitivity example, not measured coverage. |
| A verified target is available at this boundary: `q=1` | 0.2375 | 22,637× | Conditional empty-start acquisition probability under this reference draw. With only three enabled operations it is 0.3167. |

Neither new known-map coverage nor `eta` can be measured from the five old fine-grid channels. The initial full map, pathability, tree/rock eligibility, capacities and new macro are missing from that decode. Therefore **the unconditional new success probability is UNVERIFIED**; its exact statement is `E[1_target_available * eta(s)/K(s)]` under the specified diagnostic distribution. A successful first harvest also changes inventory, enabled operations and subsequent boundaries, so 22.6 and 86.0 are fixed-old-coverage counterfactuals, not forecasts for a new 90-minute run.

At a boundary with no known reachable target, harvest probability is zero. The design removes the artificial offset and quantity lotteries and absorbs navigation; it does not conjure resources on an inaccessible map. No wall-clock harvest rate or whole-factory success probability follows from these numbers because the new travel macro has a different duration distribution.

## 6. Delete the harmful parts

| Delete or replace | Reason |
|---|---|
| Shared universal anchor/offset semantics for HARVEST and MOVE | Resource identity and regional destination are the actual arguments. Reach belongs inside acquisition execution. |
| Always-enabled required sentinels and state-only marginal masks | They manufacture complete tuples known to be impossible. No real entity means no entity-only operation. |
| The negative ALL quantity bin | It is not accepted consistently by the tools. Use positive, operation-checked requests. |
| Sampling every exported item/prototype as tool-supported | Python adapter coverage and live prototype existence differ. Unsupported dispatch must be explicit. |
| Empty-world long WAIT durations and host polling as the time contract | They consumed most wall time while producing nothing, and make transition timing depend on the host. |
| Preflight-invalid transitions that burn the episode budget | Invalid client messages are outside legal action support; actual execution failures remain real transitions. |
| Numeric-return or entity-return success without an effect receipt | These inflate success counts and hide unchanged targets, wrong resources and partial work. |
| Nearby substitute targets and silent inexact relocation | The selected entity, resource and placement site must be the executed target. |
| Center-only occupancy, missing regional resource identity, clipped remote coordinates and merged entity item totals | They hide distinctions the actions require. |
| Fixed goal/readiness features tied to an iron drill feeding a chest | They encode one construction pattern and are unnecessary for PS. Remove the mandatory `Y` readiness scan/validation from this environment contract (`fle/cluster/scenarios/open_world/observation_diff.lua:453-495`; `fle/rl/env.py:223-224`). |
| Zero stock/power placeholders without meaningful validity, arbitrary network magnitudes, unscaled joules and guessed direction/status numbers | They convey missing or arbitrary information as physical state. Keep typed validity and exported codebooks. |
| Zero environment reward with benchmark-owned replacement, item-threshold reward and positive-only production surrogates | There must be one reward authority implementing signed automated PS. |
| Fixed action-count task endings, implicit resync clock resets and invisible pending macros | They make nominally identical episodes and boundaries mean different things. |

Do not delete recursive crafting, exact relative placement, automatic routing, persistent named terrain, player inventory, full technology identity, stratified entity selection or the combined diff drain. The acceptance standard for the replacement is concrete: zero required-sentinel or absent-anchor submissions through legal support; a supported visible-target HARVEST can acquire the first item in one decision; every selected site is visible at execution resolution; every result reports its actual effects and ticks; and accumulated reward exactly equals the change in the published automated-score endpoint. These are environment properties to establish, not claims that have been measured for this proposed design.

## Implementation changelog — 2026-09-15, first material transition only

Implemented HARVEST as `(target, harvest_quantity)` in the static MultiDiscrete:
255 addressable patch rows plus an absent sentinel, and positive quantities
1/5/20. Targets come from connected same-name ore tiles in the existing terrain
cache, including distant known chunks. Rows expose resource vocabulary ID,
persistent patch ID, exact representative tile and bounding extent. IDs retain
membership through depletion/splitting and full sync; overflow raises explicitly.
No utility ranking, dynamic action cardinality or semantic admission rules were
added. The existing policy now consumes this small target view. Observation size
is 25,121 floats with the pinned live vocabulary; the changed fingerprint/shapes
require fresh replay/checkpoints. The vocabulary export format is unchanged.

The macro resolves an actual resource inside a selected patch tile, finds a
non-colliding working position, calls move_to, and reads actual server position,
walking queue and effective resource reach before mining. Startup also reads and
validates reach; the hard-coded 2.7 and HARVEST offset mask are deleted. Each bound
Lua call checks surface/name/exact position and epsilon reach, mines only that
resource into the character inventory, and reports accepted item deltas. It
cannot enter the legacy nearby/cross-type fallback. The macro can continue over
the selected patch's observed tiles. Outcomes distinguish path_failed,
target_exhausted, interrupted and tool_rejected from ok, with items_acquired
reported separately; outstanding walking is cancelled before returning.

**Probability for this implementation (supersedes the proposed K=4 example):**
under uniform masked heads, `p = q_known_patch × (1/12) × f_manual × 1 × eta`.
Here f_manual is the fraction of offered patches that are manually harvestable;
the view deliberately retains all resource types, with eligibility enforced by
the simulator at execution. Target-address validity and quantity validity are
both 1 whenever a patch exists. Eta includes path/collision failure, depletion,
capacity and interruption. No operation families were pruned. At a boundary
with only eligible patches this is `0.083333 × eta`, approximately 7,942 times
the old `1.0492e-5` when eta=1. Using only old fine-window coverage as a
counterfactual gives `(362/21540) × (1/12) × 1 × 1 × eta = 0.0014005 × eta`
(about 133.5 times the audit baseline). These are conditional arithmetic, not a
measured new-run probability; q, f_manual and eta remain unmeasured.

Offline regression tests cover a patch 40 tiles away in one decision, navigation
before harvest, actual reach/position rather than returned waypoints, failed and
interrupted trips, depletion identity, startup reach, hook wrappers, generated
RCON Lua syntax, exact Lua binding and accepted inventory receipts. Verbatim
suite output is in `12-astra-env-modelling-tests.txt`. No Docker, Factorio or
training was run. Only a live server can confirm the effective reach value,
Factorio API/mining behavior, path completion and end-to-end acquisition rate.

Still specified but unbuilt: typed tree/rock targets (current terrain records lack
their names), general conditional admission, observation/placement redesign,
reward changes, and authoritative macro deadlines/scheduling. This
patch retains existing move_to timing and fast-harvest timing; it does not add a
watchdog to a blocking move_to. All other sections above remain proposals.
