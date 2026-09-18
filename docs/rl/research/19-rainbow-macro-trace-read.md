# Rainbow macro trace read, 2026-09-17

## Scope

Read-only snapshot at 16:20 CEST: 100 completed episodes and 10,975 decisions (27003/4/5: 45/30/25 episodes). Citations are `port:e/step/line`; line is in `env_<port>.jsonl`. Partial episodes are excluded.

## Policy and economy

The global median APS is 103 (mean 139). The five best and five closest-to-median episodes are:

| set | port:e | APS / player score | successful feed (coal, iron, other) | furnace craft/place |
|---|---:|---:|---:|---:|
| best | 27005:22 | 619 / 3,981 | 113, 435, 0 | 22 / 6 |
| best | 27005:14 | 564 / 2,636 | 111, 281, 0 | 1 / 3 |
| best | 27005:24 | 522 / 2,510 | 161, 285, 20 stone | 1 / 20 |
| best | 27003:41 | 497 / 2,405 | 42, 261, 0 | 1 / 4 |
| best | 27005:23 | 457 / 3,126 | 45, 246, 20 stone | 1 / 5 |
| median | 27003:10; 27004:11; 27005:11; 27003:14; 27003:26 | 102–115 / 328–1,036 | totals 165, 285, 20 copper + 8 stone | 10 / 22 |

All are **stone-furnace** economies. A representative best episode crafts/places/fuels one at steps 2/3/5 (`27005:22/s2/L3169`, `s3/L3170`, `s5/L3172`) and then harvests and hand-feeds it and later furnaces. Across the best five, reconstructed net player inventory is coal 1,293, stone 763, copper ore 195, wood 42, furnaces 22, iron ore 0; successful transfers put 1,508 iron ore, 472 coal and 40 stone into furnaces. Thus production is overwhelmingly iron plates, with at most a small brick line; outputs stay in furnaces (only 65 coal is extracted). Median-five net inventory is iron 124, stone 58, coal 32 and copper 27; only two iron ore are extracted. No plate/brick delta is logged.

There is no `burner-mining-drill` occurrence: none crafted, placed, fuelled, or producing. Consequently **all APS is from hand-fed furnaces**, not mining automation. Best endpoints are directly recorded at `27005:22/L3315`, `27005:14/L2140`, `27005:24/L3616`, `27003:41/L3524`, and `27005:23/L3469`.

## Learning signal and drift

| port | first-20 APS mean/median | last-20 APS mean/median | player-score median, first→last |
|---:|---:|---:|---:|
| 27003 | 23.6 / -1 | 239.5 / 226 | 397.5→1,683.5 |
| 27004 | 49.7 / -1 | 172.4 / 205.5 | 626.5→1,618 |
| 27005 | 119.5 / 43.5 | 220.8 / 224 | 827→2,264 |

Global chronological 20-episode blocks show a sharp strategy change:

| eps | APS mean | H/I/M/W share | PLACE/RESEARCH | no-support / tool-rejected | top refusal |
|---:|---:|---:|---:|---:|---|
| 1–20 | 1.3 | 20.0/8.8/34.4/17.2% | 5.3/6.2% | 4.6/4.4% | `other` |
| 21–40 | 44.0 | 9.6/44.4/11.9/11.2% | 7.4/5.8% | 14.5/14.8% | item not held/accepted |
| 41–60 | 127.3 | 22.9/32.2/17.7/3.1% | 9.6/3.7% | 7.2/21.6% | `other` |
| 61–80 | 215.6 | 41.3/34.4/17.3/0.6% | 2.9/0.8% | 1.0/20.0% | no suitable slot |
| 81–100 | 307.7 | 42.8/24.1/19.2/0.2% | 4.6/1.7% | 3.1/4.6% | no suitable slot |

Metrics show Q/loss growing with returns: at env steps 500/3k/6k/9k/11.1k, mean-max-Q is 0.16/8.97/25.61/66.67/91.50 and TD loss 0.50/0.38/1.73/5.39/7.19 (`metrics.jsonl:L5,L30,L60,L90,L111`). PER positive-delta fraction is 0.094/0.547/0.500/0.531/0.551; latest positive/negative mean priorities are 11.39/5.44, the intended optimistic skew. Rising loss deserves monitoring but accompanies rising Q rather than collapse.

## Conveniences, waste, and defects

Executor navigation occurs on 4,375 non-MOVE steps: HARVEST 2,924, INSERT 1,256, PICKUP 174, EXTRACT 20, PLACE 1 (`27003:0/s0/L1`). Quantity is clamped 728 times (INSERT 565, CRAFT 153, EXTRACT 10; `27003:0/s19/L20`). Cooldown-hit/masked flags are both zero. No frontier-forced, first-try, or secondary-head-forced sample appears. Comparing non-null requested/executed fields finds **zero substitutions** for item, anchor, peer, or offset: no rule violation. The 1,260 one-sided nulls are 595 refused item calls plus all 665 PLACE rows, whose `executed_offset` telemetry is absent even on success; logged targets show no changed choice.

Waste remains material: 5,060/10,975 steps (46.1%) have no item delta and zero reward, including 1,819 MOVE, 1,695 INSERT, 691 WAIT, and 396 RESEARCH. WAIT is 702 (6.4%) and RESEARCH 404 (3.7%) overall. There are 177 immediately adjacent successful PLACE→PICKUP transitions (`27003:8/s62–63/L676–677`). Identical-argument refusal runs total 77, containing 457 repeats beyond the first; the worst is 84 identical wrong-input inserts (`27005:9/s29–112/L1316–1399`).

Visible environment defects (categories may overlap):

- Deterministically maskable INSERT refusals: 1,170 (743 full-slot, 427 furnace already holding another input; `27003:13/s71/L1155`), plus one empty EXTRACT miss (`27005:14/s148/L2137`).
- Predicate/tool disagreement: 43 PLACE offsets admitted then rejected for a tree/rock (`27003:7/s81/L606`); 136 “inventory full” errors report only 12–19/50 occupied (`27003:14/s44/L1222`).
- Stale-content symptoms: 221 invalid `LuaItemStack` reads (`27003:22/s47/L1942`) and 31 macro HARVEST arrivals with nothing present (`27004:5/s40/L569`). Repeated refusals despite zero cooldown telemetry also indicate cooldown failure.

## Ranked recommendations

1. Add only deterministic, observation-backed masks for incompatible furnace input, visibly full slots, and empty extraction; repair exact-action cooldown persistence/telemetry. Do not mask contingent placement.
2. Add a Rainbow behavior-policy first-try/diversity phase keyed by ladder rung and complete sampled action, with ore-visible drill offsets and held fuel sampled as in D17—never substituted.
3. Use behavior-policy count/recency penalties (not reward shaping or masks) against valid PLACE/PICKUP churn, repeated RESEARCH, and WAIT, while retaining a low noisy-exploration floor.
