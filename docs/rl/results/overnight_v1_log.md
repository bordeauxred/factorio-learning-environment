## 01:53 check 1
- Arms alive: ctrl-v1 (27004) 4 eps, boot-v1 (27002+27005) 8 eps, ctrl-v1-s2 (27000) 4 eps; random-v1 census done, 12 eps, 2/12 positive (max 8) on env v1.
- Last-20 APS medians: ctrl-v1 2.0 (2/4 positive, max 8), boot-v1 -1, ctrl-v1-s2 -1. Firsts: ctrl-v1 furnace fuelled and fed at ep1; boot-v1 chest only so far.
- D2 follow-up not landed (its fake-env acceptance still running); UCB arm pending. No relaunches needed.
## 02:23 check 2
- ctrl-v1 10 eps, 4/10 positive, max 17, plates extracted at ep6; ctrl-v1-s2 11 eps, 0/11 (5/11 fuelled+fed furnaces); boot-v1 20 eps, 0/20, never fuelled a furnace.
- boot-v1 was degenerate: 61% RESEARCH, 0 INSERT, greedy heads with eps floor 0.02. Resume failed (D2 changed the bootdqn module layout mid-run; checkpoint keys mismatch). Restarted boot-v1 fresh on current code with --eps-floor 0.10; old run moved to runs/boot-v1-old.
- D2 still editing dqn.py (02:03); no UCB flags yet. Plots regenerated into docs/rl/results/overnight_v1/.
## 02:53 check 3
- ctrl-v1 17 eps, 6/17 positive (max 17); ctrl-v1-s2 19 eps, 1/19 positive (max 40); boot-v1 (fresh, eps floor 0.10) 6 eps, 0/6, furnace placed at ep4.
- All processes alive, logs fresh. No UCB flags yet (D2 follow-up running).
- Eval slot: boot has no checkpoint yet; running greedy eval of ctrl-v1's latest checkpoint on 27003 instead.
## 03:05 D2 follow-up landed; arms reconfigured
- Launched ucb-v1 on 27000+27001 (dqn, count-UCB c=1.5, eps floor 0.02, PER a=0.5 b0=0.4 cap p99, n-step 5, gamma 0.999, lr 1e-4, target 2000, replay 40k, batch 64, archive 25%, action logs, 60k steps). ctrl-v1-s2 stopped (19 eps, 1/19 positive).
- Launched ctrl-v1m on 27003: identical hyperparameters, epsilon 1.0 -> 0.05 over 15k (matched control). Periodic greedy eval suspended overnight; all 6 servers now train.
- ctrl-v1 (v0 hyperparameters) continues on 27004 as the learner-only baseline; boot-v1 continues on 27002+27005.
## 03:23 check 4
- ctrl-v1 30 eps, 11/30 positive, max 41, mean max-Q 13; ucb-v1 8 eps, 2/8 positive, plates extracted already in its first episode (forced first try of each supported pair); ctrl-v1m 4 eps, 0/4; boot-v1 20 eps, 0/20, fuelled at ep18 and fed at ep3 but never both.
- All 4 learners alive, 6 servers at 70-92% CPU. Plots regenerated.
- Watch item: host free memory is low (four learners, ctrl-v1 holds a 100k replay); swap checked this round.
- Memory: swap 24.5/25.6 GB used but resident learner sizes are 0.3-0.5 GB and system-wide free memory is 39%; the swap is idle apps and other sessions, not the runs. No action.
## 03:53 check 5
- ctrl-v1 45 eps, 18/45 positive, last-20 median 0, max 51, mean max-Q 20; ctrl-v1m 14 eps, 8/14 positive, last-20 median 15, max 58 (strongest early curve, epsilon still 0.76); ucb-v1 26 eps, 3/26, max 30; boot-v1 34 eps, 3/34, max 23, first fuelled-and-fed at ep13.
- All 4 learners alive, logs fresh. No relaunches.
- No gears crafted in any arm yet (the v0 wall); plates extracted in ctrl-v1, ctrl-v1m and ucb.
## 04:23 check 6
- ctrl-v1 60 eps, 32/60 positive, last-20 median 7, max 133; ctrl-v1m 26 eps, 13/26, median 11, max 92; ucb-v1 46 eps, 7/46, max 59; boot-v1 48 eps, 11/48, max 65.
- NEW RUNGS: gears crafted (ctrl-v1m ep23, ucb ep22), inserter placed (ctrl-v1m ep16, ucb ep22), belt placed (ctrl-v1m ep22, ucb ep38). No drill yet. Traces saved to docs/rl/results/overnight_v1/first_gear_traces.md.
- All 4 learners alive. Plots regenerated. Eval slot skipped: all six servers are training (greedy eval deferred to the morning).
## 04:53 check 7
- ctrl-v1 73 eps, 41/73 positive, last-20 median 28, max 180; boot-v1 65 eps, 26/65, last-20 median 14, max 274 (biggest single episode so far); ctrl-v1m 39 eps, 21/39, median 2, max 92; ucb-v1 64 eps, 13/64, median -1, max 59.
- All 4 learners alive. No new rungs (gears in ctrl-v1m and ucb only; no drill anywhere).
- Reading: UCB reaches deeper rungs earliest but exploits worst (its bonus keeps pulling toward untried pairs); boot with the 0.10 floor has overtaken it on APS.
## 05:23 check 8
- boot-v1 81 eps, 38/81 positive, last-20 median 68, max 393 (now the leading arm); ctrl-v1 84 eps, 52/84, median 32, max 180; ctrl-v1m 52 eps, 30/52, median 26, max 92; ucb-v1 82 eps, 29/82, median 5, max 59.
- Ladder: gears crafted in 8/82 ucb episodes (repeating, not a fluke) and 2/52 ctrl-v1m; inserters placed 4/82 and 3/52; still no drill crafted anywhere. Plots regenerated.
- All 4 learners alive. ctrl-v1 reaches its 30k-step budget around 05:55; relaunch with --resume planned at the next check.
## 05:53 check 9
- NEW RUNG: ctrl-v1m crafted a burner mining drill at ep54 (first drill in the project). Not yet placed. Trace appended to first_gear_traces.md.
- boot-v1 97 eps, 53/97 positive, last-20 median 91, max 393; ctrl-v1 98 eps, 66/98, median 69, max 206 (25.0k of 30k steps); ctrl-v1m 64 eps, 38/64, median 40; ucb-v1 100 eps, 43/100, median 14, max 78.
- All 4 alive. Eval slot skipped again (all servers training). ctrl-v1 finishes its budget in ~25 min; resume at next check.
## 06:23 check 10
- boot-v1 111 eps, 66/111 positive, last-20 median 140, max 393; ctrl-v1 109 eps, 77/109, median 103, max 261 (27.9k/30k steps, still running); ctrl-v1m 76 eps, 45/76, median 31, max 121; ucb-v1 118 eps, 60/118, median 26, max 90.
- NEW: ucb-v1 crafted a drill at ep115 (second arm to do so). Gears now in 10/118 ucb episodes. No drill placed anywhere yet.
- All 4 alive; plots regenerated.
## 06:53 check 11
- ctrl-v1 finished its 30k budget (117 eps, 85/117 positive, last-20 median 135, max 301); resumed from checkpoint-final.pt with --total-steps 60000 (v0 hyperparameters kept; replay not persisted by resume).
- boot-v1 125 eps, 80/125, last-20 median 222, max 393; ctrl-v1m 89 eps, 53/89, median 26, max 121; ucb-v1 140 eps, 67/140, median 9, max 90 (drill crafted at ep62 per the earliest-first scan).
- 3 learners alive before the resume, 4 after. No drill placed anywhere yet.
## 07:23 check 12
- boot-v1 139 eps, 93/139 positive, last-20 median 153, max 596 (new project max); ctrl-v1 (resumed) 127 eps, 95/127, median 143, max 400; ctrl-v1m 102 eps, 62/102, median 43, max 121; ucb-v1 162 eps, 82/162, median 2, max 90.
- Drill crafted: ucb 5/162, ctrl-v1m 1/102; drill placed: none anywhere. UCB explores deepest but its APS median has fallen back to 2.
- All 4 alive; plots regenerated; eval slot skipped (all servers training).
## 07:53 check 13
- ctrl-v1 138 eps, 103/138 positive, last-20 median 108, max 1189 (new project max); boot-v1 153 eps, 105/153, median 143, max 596; ctrl-v1m 115 eps, 71/115, median 43 (29.2k/30k steps, resume due next check); ucb-v1 184 eps, 95/184, median 8.
- No new rungs. All 4 alive.
- Record episode (ctrl-v1 ep17 after resume, APS 1189): 9 furnaces, 87 harvests, 87 iron-ore inserts, 9 coal inserts, 128 INSERT ops total; APS rose almost linearly through the episode. A hand-fed iron-plate smelting line scaled up, no drill.
## 08:23 check 14
- ctrl-v1m finished its 30k budget (118 eps, 73/118 positive, last-20 median 40, max 121); not resumed. Its server 27003 now runs the deferred greedy evaluations: latest boot-v1, ctrl-v1 and ucb-v1 checkpoints, 4 episodes each.
- ctrl-v1 148 eps, 111/148, last-20 median 177, max 1189; boot-v1 168 eps, 119/168, median 143, max 596; ucb-v1 206 eps, 109/206, median 21, max 116.
- 3 learners alive (ucb 52k/60k, boot 42k/60k, ctrl-v1 38k/60k). Plots regenerated.
## 08:53 check 15
- ctrl-v1 156 eps, 119/156 positive, last-20 median 593 (accelerating after the resume), max 1189; boot-v1 182 eps, 133/182, median 171, max 701, drill crafted at ep172 and inserter placed at ep181 (new for boot); ucb-v1 228 eps, 126/228, median 25.
- Greedy eval (4 episodes each, port 27003): ctrl-v1@38k = 574 in every episode; boot@42k = 38 in every episode; ucb@52k = -1 in every episode. Identical repeats: the env is deterministic for a deterministic policy, so greedy eval episodes are replicas, not samples.
- 3 learners alive (ucb 57.8k/60k, will finish soon; boot 45.8k; ctrl-v1 40k/60k).
- 08:56: 27003 idle after eval; launched ctrl-v1-s2b (v0 hyperparameters + PER, seed 2, 60k) as a second seed of the best arm.
## 09:23 check 16
- ucb-v1 finished its 60k budget (237 eps, 135/237 positive, last-20 median 23, max 116). Relaunched as ucb-v1-c05 on 27000+27001: resumed from checkpoint-final.pt (nets, counts, archive) with --ucb-c 0.5, 100k budget, i.e. the exploitation-leaning hybrid from the report's recommendation 1.
- ctrl-v1 164 eps, 127/164, last-20 median 661, max 1189; boot-v1 197 eps, 148/197, median 181, max 960; ctrl-v1-s2b (seed 2 of ctrl) 11 eps, 6/11 positive.
- 3 learners alive before the relaunch, 4 after. Plots regenerated.
## 09:53 check 17
- ctrl-v1 171 eps, 134/171 positive, last-20 median 797, max 1491 (new project max); boot-v1 210 eps, 161/210, median 366, max 1243; ctrl-v1-s2b 23 eps, 15/23, median 28, gears crafted and belt placed already at ep12; ucb-v1-c05 20 eps since resume, 18/20 positive, median 59 (vs 23 for ucb at c=1.5): the hybrid exploits immediately.
- All 4 alive. No drill placed anywhere.
## 10:23 check 18
- ctrl-v1 178 eps, 141/178 positive, last-20 median 958, max 1491; boot-v1 223 eps, 174/223, median 429, max 1545 (new project max); ucb-v1-c05 40 eps, 38/40, median 77, max 364; ctrl-v1-s2b 34 eps, 24/34, median 44, inserter placed at ep30.
- All 4 alive; plots regenerated; eval slot skipped (all servers training). Still no drill placed in any arm.
## 10:53 check 19
- ctrl-v1 185 eps, 148/185 positive, last-20 median 1006, max 1666 (new max); boot-v1 236 eps, 187/236, median 792, max 1545 (58.4k/60k, resume due next check); ucb-v1-c05 60 eps, 57/60, median 155, max 497; ctrl-v1-s2b 46 eps, 35/46, median 61.
- All 4 alive. No new rungs; drill still never placed.
## 11:23 check 20
- boot-v1 finished its 60k budget (243 eps, 194/243 positive, last-20 median 731, max 1545); resumed from checkpoint-final.pt with --total-steps 100000, same configuration.
- ctrl-v1 191 eps, 154/191, last-20 median 1243, max 1666; ucb-v1-c05 80 eps, 77/80, median 225, max 498; ctrl-v1-s2b 57 eps, 45/57, median 51.
- 3 learners alive before the resume, 4 after. Plots regenerated.
- 11:38: user asked to kill poor runs; stopped ctrl-v1-s2b (57 eps, last-20 median 51). 27003 free.
- 11:48: research memo landed at docs/rl/research/13-algorithms-for-open-play.md; dispatched D3 (rung-conditioned UCB, frontier archive sampler, explorer/exploiter per port, stochastic eval).
## 11:53 check 21
- ctrl-v1 197 eps, 160/197 positive, last-20 median 1196, max 1762 (new max); boot-v1 252 eps, 203/252, median 652; ucb-v1-c05 96 eps, 91/96, median 225.
- 3 learners alive (ctrl-v1-s2b stopped at the user's request). 27003 idle: running one greedy episode each of the latest ctrl-v1 and boot-v1 checkpoints (deterministic, so one episode suffices).
- D3 (rung-conditioned UCB, frontier archive, explorer/exploiter, stochastic eval) in progress at Codex.
## 12:05 day plan v2 launched (memo section 5)
- Stopped boot-v1 (252 eps, last-20 median 652, 62k steps) and ucb-v1-c05 (96 eps, median 225, 85k steps); checkpoints kept. ctrl-v1 continues on 27004 as the anchor (v0 hp + PER, 50k steps, median 1196).
- armA-v2 on 27003: fresh plain DQN+PER, eps 1.0->0.1 over 18k, uniform archive 25%. armB-v2 on 27000(epsilon 0.2->0.05 over 5k)+27001(rung-UCB c 0.75->0.15 over 40k, eps floor 0.02), uniform archive 25%. armC-v2 on 27002+27005: as B but frontier archive, 50% of batch. Common: PER a=0.6 b=0.4, n=3, gamma 0.99, replay 100k, target 1000, batch 64, 57,600 steps each, seed 3.
- Decision metric: root episodes with drill placed, fuelled and producing; repeat rate afterwards; greedy median APS >= 80% of arm A.
## 12:23 check 22
- v2 arms after ~20 min: armA 7 eps (3/7 positive, max 44), armB 10 eps (1/10, max 25), armC 7 eps (1/7, max 83); all already fuel-and-feed furnaces and extract plates by ep4-5. ctrl-v1 anchor 203 eps, last-20 median 1329, max 1762.
- All 4 learners alive; plots regenerated with the v2 arms.
- Metric caveat: armC reports automated_ore_produced=5 with drill_placed=0, so that detector fires on non-drill ore increases (probably EXTRACT); treat drill_placed and drill_fuelled as the real counters until fixed.
## 12:53 check 23
- 
- All learners alive: 1. D4 (QR-DQN + NoisyNets, --rainbow preset) in progress at Codex.
## 12:53-13:05 incident: disk full
- The disk hit 100% (it was at 874/926 GB before our runs; checkpoints with embedded archives are 50-98 MB every 2k steps). ctrl-v1, armB-v2 and armC-v2 died on checkpoint writes at ~12:45; armA-v2 survived.
- Freed 23 GB: deleted Codex acceptance dirs in /tmp and pruned checkpoints to the newest 1-3 per run (17.9 GB). One truncated checkpoint (armB 6000) removed. Now 25 GB free; a background pruner keeps 2 checkpoints per run every 10 min.
- Resumed: ctrl-v1 from checkpoint-52000 (100k budget), armB-v2 from 4000, armC-v2 from 2000 (replay and ~10-20 min of transitions lost on each).
## 13:20 Rainbow arm launched
- D4 landed: QR-DQN (51 quantiles, rank-4 residual parameterization), factorized NoisyNets, --rainbow preset (PER a=0.5 b=0.4, n=3, gamma 0.99, target 1000, lr 6.25e-5); 77 offline tests. On FakeMacroEnv: rainbow 22.96 vs random 14.04 at 5k steps.
- Stopped armA-v2 (14 eps; duplicate of the anchor's configuration). rainbow-v2 on 27003: --rainbow, noisy exploration (eps floor 0.02), uniform archive 25%, replay 100k, 57,600 steps, seed 3. Anchor ctrl-v1 stays the plain-DQN reference on 27004; armB/armC continue.
## 13:23 check 24
- ctrl-v1 (anchor, resumed at 52k) 210 eps, last-20 median 1242, max 1762; armB-v2 30 eps, 3/30 positive, max 154 (post-resume counters: 6 root eps, no gears yet); armC-v2 19 eps, 2/19, max 83; rainbow-v2 1 ep (just started).
- All 4 learners alive; disk 23 GB free with the pruner running. Eval slot skipped (27003 now hosts rainbow). No drill placed in any arm.
## 13:53 check 25
- ctrl-v1 214 eps, last-20 median 1068, max 1762; armB-v2 38 eps, 5/38 positive, max 279; armC-v2 23 eps, 2/23, max 83; rainbow-v2 5 eps, 1/5, max 12.
- All 4 alive, 22 GB free, plots regenerated. No gears or drills in any v2 arm yet (v2 arms have 4k-8k steps each since the disk incident).
- Throughput note: armC advances about half as fast as armB (4.0k vs 7.8k steps in the same wall time); the frontier sampler is likely the cost. Watch this: if it persists, C gets fewer decisions per hour than B for the same budget.
## 14:23 check 26
- ctrl-v1 217 eps, last-20 median 979 (drifting down from 1430 since the resume lost its replay), max 1762; armB-v2 48 eps, 9/48 positive, max 279, first gear at ep47; armC-v2 30 eps, 4/30, max 250; rainbow-v2 9 eps, 1/9, mean max-Q 0.34 (slow start with lr 6.25e-5).
- All 4 alive, 21 GB free, plots regenerated. Frontier sampler in C is drawing as designed (deepest rung present = plates, 1600 of 3200 archive draws, 1498 from the pre-attainment window).
- No drill placed anywhere. Anchor decline after the resume is the cost of losing replay on the disk-full crash; keep watching whether it recovers.
## 14:53 check 27
- ctrl-v1 221 eps, last-20 median 922; armB-v2 61 eps, 20/61 positive, max 279, 1 gear (ep53 in the merged index); armC-v2 40 eps, 7/40, max 250, no gear; rainbow-v2 16 eps, 1/16, mean max-Q 0.45.
- All 4 alive, 20 GB free. Eval slot skipped (27003 busy). Decision point for B/C set at 17:00 (kill C if neither has placed a drill and B and C look alike); rainbow lr revisit at 16:00.
## 15:23 check 28
- ctrl-v1 226 eps, last-20 median 898; armB-v2 77 eps, 31/77 positive, last-20 median 2, max 279, gears in 2 episodes, belt placed at ep75; armC-v2 52 eps, 9/52, max 250, no gear; rainbow-v2 24 eps, 1/24, mean max-Q 0.33 after 6.1k steps.
- Rainbow decision taken early: rainbow-v2 stopped (not learning at lr 6.25e-5 on this budget); relaunched as rainbow-v2b with the anchor's lr 1e-4 and PER alpha 0.6, QR + noisy kept, so the comparison isolates the distributional and noisy components.
- All learners alive, 19 GB free, plots regenerated. No drill placed anywhere.
## 15:53 check 29
- ctrl-v1 232 eps, last-20 median 1125 (recovering), max 1762; armB-v2 94 eps, 37/94 positive, last-20 median 2, gears in 2 eps, belts placed; armC-v2 64 eps, 11/64, max 250, plates only (conversion counters: 0 plates since restart); rainbow-v2b 7 eps, 1/7, mean max-Q 0.19 at 1.9k steps.
- All 4 alive, 19 GB free. No drill placed anywhere. Eval slot skipped (27003 busy with rainbow).
- Note for the 17:00 decision: C's conversion counters show 0 plate extractions since its restart although its ladder shows plates at ep37 (counter reset at resume); judge B vs C on the ladder file, not the counters.
## 16:23 check 30
- ctrl-v1 237 eps, last-20 median 1395 (recovered), max 1762; armB-v2 109 eps, 41/109 positive, gears 2 eps, belts placed; armC-v2 77 eps, 14/77, belt placed at ep42, no gear; rainbow-v2b 16 eps, 1/16, mean max-Q 0.42 at 4k steps (flat).
- Rainbow diagnosis: two configurations (lr 6.25e-5 and 1e-4) both flat and below random, unlike the plain control at the same step count. Stopped rainbow-v2b; launched qr-v2 on 27003: QR-DQN (51 quantiles) with the anchor's epsilon schedule and no noisy layers, to attribute the failure to noisy exploration or the quantile head.
- All learners alive, 19 GB free, plots regenerated. No drill placed anywhere.
## 16:53 check 31 and the 17:00 decision
- ctrl-v1 242 eps, last-20 median 1395; armB-v2 125 eps, 49/125 positive, gears 2 eps, belts; armC-v2 88 eps, 16/88, no gear, 60% of B's throughput; qr-v2 8 eps, 6/8 positive, max 75, mean max-Q 4.9 at 2.3k steps.
- Rainbow diagnosis closed: QR head + epsilon learns like the plain control; the two flat Rainbow runs shared only the noisy-network exploration, so NoisyNets (sigma0 0.5, fresh noise per step) are the failing component here. qr-v2 continues on 27003 as the value-axis arm.
- Decision: armC-v2 stopped (B >= C on positives, rungs and throughput; neither placed a drill). armD-v2 launched on 27002+27005: QR head + B's explorer/exploiter behaviour (rung-UCB + epsilon), uniform archive, seed 4.
## 17:10 frontier return (D5) landed and live-checked
- D5: FleMacroEnv.snapshot()/reset(options={"restore"}) plus --frontier-return in the learner; 84 offline tests. Live smoke on 27003: snapshot 0.11 s, restore 0.83 s, player position, inventory and hash all restored exactly, excursion tagged, budget honoured. Entity restore not exercised (no entities in the smoke).
- qr-v2 resumed on 27003. armD-v2 restarted fresh (was 15 min old) on 27002+27005 with --frontier-return --return-prob 0.5 --return-budget 64: when its explorer crafts a drill, half its next episodes start from that state. armB-v2 stays as the clean explorer/exploiter comparison.
## 17:23 check 32
- ctrl-v1 247 eps, last-20 median 1412, max 1929 (new max); armB-v2 138 eps, 59/138 positive, last-20 median 25, gears in 4 eps, belts; armD-v2 3 eps (fresh, frontier return armed, 0 excursions yet); qr-v2 13 eps, 9/13 positive, max 75, but its process is gone (metrics 14 min old) - investigating and relaunching.
- 3 learners alive, 18 GB free, plots regenerated. No drill placed anywhere.
- 17:27: qr-v2 relaunch failed earlier on a shell quoting error (--resume passed as one word); relaunched correctly from checkpoint-2000.
## 17:53 check 33
- ctrl-v1 249 eps, last-20 median 1342, max 1929; armB-v2 145 eps, 63/145 positive, last-20 median 24, max 510 (new for B); qr-v2 18 eps, 14/18 positive, max 78 (the quantile head is on the plain control's trajectory); armD-v2 8 eps, 0/8, frontier return armed, no drill yet so 0 excursions.
- All 4 alive, 17 GB free. No drill placed anywhere; B has gears in 4 episodes but no drill craft since its restart.
## 18:23 check 34
- 
- Learners alive: 4. Plots regenerated. D6 (character-tile and footprint offset mask, forced-pair secondary exploration, return-rung, per-port horizon) in progress at Codex; the drill wall is diagnosed as a masking gap (55 refused placements at the player's tile).
## 18:24 D6 landed; drill-capable arms restarted on the new code
- D6: player tile excluded from the PLACE offset mask (footprint expansion already existed), --forced-arg-eps (secondary heads resampled for first-time or just-refused pairs), --return-rung, --max-steps-by-port; 88 offline tests.
- armB-v2 resumed from runs/armB-v2/checkpoint-28000.pt with --forced-arg-eps 0.3 --max-steps-by-port 256,512 (explorer port 512 steps), 80k budget. armD-v2 resumed from runs/armD-v2/checkpoint-2000.pt with the same plus --frontier-return --return-rung gears_crafted. Anchor and qr-v2 untouched (references).
## 18:53 check 35
- ctrl-v1 258 eps, last-20 median 1171, max 1929; armB-v2 165 eps, 81/165 positive, last-20 median 87 (note: its explorer port now runs 512-step episodes, so B's medians mix horizons from here), gears 1 since restart; armD-v2 24 eps, 1/24, no gear yet, 0 excursions; qr-v2 34 eps, 26/34 positive, last-20 median 55, max 173, gears crafted and belt placed at ep27 with plain epsilon exploration.
- All 4 alive, 16 GB free. No drill placed anywhere yet; the D6 fix has been live for 20 minutes and no drill has been crafted since.
## 19:23 check 36
- ctrl-v1 261 eps, last-20 median 1171 (metrics 5.7 min old, process alive; watch); armB-v2 173 eps, 88/173 positive, median 98 (mixed horizons); armD-v2 31 eps, 4/31, gears + inserter + belt at ep29 but 0 excursions recorded despite --return-rung gears_crafted (investigate next check if still 0); qr-v2 41 eps, 33/41, median 73, max 173.
- All 4 alive, 27 GB free (pruner), plots regenerated. No drill crafted since the D6 fix (drill rung untested in training so far).
- 19:24: armD excursions are running after all: 25 excursion-tagged steps on the UCB port (27005) after its gear episode; the metrics counter lags. ctrl-v1 metrics advanced (65.8k), not stuck.
## 19:53 check 37
- ctrl-v1 264 eps, last-20 median 1200; armB-v2 178 eps, 92/178 positive, median 98; armD-v2 37 eps, 7/37, gears+inserter+belt at ep32, 2 excursions run (130 excursion steps) from the gear state, no drill crafted in them yet; qr-v2 47 eps, 39/47, median 95, max 173.
- All 4 alive, 26 GB free. No drill crafted or placed in any arm since the D6 fix.
## 23:56 check 38
- ctrl-v1 290 eps, last-20 median 1438, max 1929; armB-v2 248 eps, 159/248 positive, last-20 median 574 (mixed 256/512 horizons), max 983; armD-v2 116 eps, 50/116, gears at ep65, inserter and belt, excursions running; qr-v2 104 eps, 83/104, median 102, max 330, DRILL CRAFTED at ep103 (on pre-D6 code, so the placement fix did not apply to it).
- All 4 alive, 20 GB free, plots and dashboard regenerated.
## 00:14 D7 landed; drill-capable arms restarted on it
- D7 (94 tests): quantity-aware CRAFT/INSERT/EXTRACT support with clamping to the largest supported quantity (logged as requested vs executed), UCB first-time pairs at the smallest supported quantity, --return-rung drill_admitted (snapshot when the drill craft first becomes admissible, first excursion action = that craft), refusal cooldowns and research no-op masks (default on).
- Restarted: armB-v2 from runs/armB-v2/checkpoint-46000.pt, armD-v2 from runs/armD-v2/checkpoint-20000.pt with --return-rung drill_admitted, qr-v2 from runs/qr-v2/checkpoint-26000.pt (epsilon held at 0.1). Anchor untouched. CONNECT restoration (D8) dispatched.
## 00:20 D8 landed: schema macro-v2 with CONNECT restored; fresh v3 arms
- D8 (100 tests): CONNECT is op 12 (heads: entity, peer, connector over the held connector items), executed with Position endpoints so the BeltGroup crash cannot occur; refusals are ordinary zero-reward steps. v1 checkpoints cannot resume on v2.
- Stopped armB-v2 (46k), armD-v2 (20k), qr-v2 (26k). Started fresh on macro-v2 with D7 guards: armB-v3 (plain DQN, epsilon+rung-UCB, 512-step explorer) on 27000+27001; armD-v3 (QR head + same + frontier return at drill_admitted) on 27002+27005; qr-v3 (QR, epsilon 1.0->0.1 over 18k) on 27003. 100k budgets. Anchor ctrl-v1 stays on v1 as the score reference.
## 00:23 check 39
- v3 arms 4 minutes old (launched 00:19 after the schema switch): env logs filling, no metrics record yet (first at step 100). Anchor ctrl-v1 process at 0.3% CPU; liveness being checked.
- 4 learner processes present, 20 GB free.
## 00:53 check 40
- Anchor ctrl-v1 299 eps, last-20 median 1535 (new high), max 1929. v3 arms 30 min in: armB-v3 14 eps, 5/14 positive, plates at ep13; qr-v3 7 eps, 5/7 positive, belt placed at ep3; armD-v3 10 eps, 0/10 (QR head starts slow, as before).
- CONNECT is live: first attempt executed through the position-endpoint path and was refused by the tool ("Transport belts cannot be connected directly from a StoneFurnace"), i.e. a normal zero-reward step, no crash.
- All 4 alive, 20 GB free, plots and dashboard regenerated.
- 01:20: wrap-up written to overnight_v1.md; dashboard rebuilt.
## 01:23 check 41
- Anchor 303 eps, last-20 median 1508, max 1929. v3 arms 1 h in: armB-v3 28 eps, 8/28 positive, plates at ep20; qr-v3 14 eps, 10/14, belt at ep3; armD-v3 20 eps, 0/20, first fuelled-and-fed furnace at ep18 (QR head warming up slowly again).
- All 4 alive, 21 GB free. No gear crafted yet in any v3 arm.
## 01:26 anchor retired; plain DQN restarted on the current env
- ctrl-v1 stopped at 76k steps (303 eps, last-20 median 1508, max 1929). It ran the 12:55 environment code (v1 schema, none of the day-2 fixes) because a running process never picks up file changes. Checkpoints stored in runs/_kept/.
- ctrl-v3 started on 27004: plain DQN + PER, v0 hyperparameters, epsilon 1.0->0.1 over 18k, on macro-v2 with all guards. All four arms now share the current environment.
## 01:45 own trace read of the v3 arms (12k steps on current code)
- Three bugs in our layer (not in FLE tools): (a) item-head union mask lets INSERT/EXTRACT pick items the anchor lacks; the quantity clamp then hits 0 and we log 396 refusals without calling the tool; (b) the exact-action cooldown is a post-selection guard, so 555 steps were logged tool_rejected/cooldown_masked and wasted; (c) ROTATE is offered on furnaces and chests: 394 refusals + 178 no-effects, the top waste in every arm. Also 207 entity ops on vanished anchors and INSERT of furnace items into furnaces.
- D9 dispatched: held-only INSERT item mask + per-anchor item clamp, cooldowns written into the per-head masks, rotatable set from the vocab, anchor-validity clamp, target-type acceptance for INSERT. Codex audit (16-env-trace-audit.md) running in parallel for anything I missed.
## 02:05 Codex audit landed (research/16-env-trace-audit.md)
- Confirms the three bugs I found and adds: PLACE mask tests one tile only (56 refusals on trees, rocks, cliffs, water, and snapped centres out of reach); crude oil and stale trees offered as harvest targets (71); CONNECT allowed the same entity as source and peer; EXTRACT contents change during approach (19); a HARVEST that acquired items was logged as refused because the tool raised afterwards; the 216k tick cap truncates every "512-step" explorer episode at 67-310 steps; cooldowns cost 632 steps (one MOVE repeated 57 times). Rewards telescope exactly (0 mismatches); no op has p95 wall > 10 s; no deaths.
- D10 dispatched for the remaining items (footprint/reach, harvestable targets, distinct CONNECT endpoints, post-approach revalidation and ok-after-exception, tick cap scaling, end_reason and position/mask logging), running concurrently with D9 on disjoint items.
## 01:41 check 42; D9 landed
- ctrl-v3 2 eps (fresh, both positive); armB-v3 33 eps, 8/33 positive, max 57; armD-v3 24 eps, 0/24; qr-v3 18 eps, 14/18 positive, last-20 median 13, max 221, one gear. All 4 alive, 19 GB free, plots and dashboard regenerated.
- D9 landed (115 tests): held-only item mask with per-anchor item clamp, rotatable set from the vocab with anchor clamp, live-anchor clamp, INSERT acceptance by target type, cooldowns written into the masks. Arms keep running on pre-D9 code until D10 lands, then all restart together.
## 02:00 D9+D10 landed; live smoke clean; all arms restarted as v4 on the fixed environment
- Live smoke (2 x 128 random steps on 27004): ok rate 0.99/0.96, 4 refusals total (2 FLE insert-tool full-slot bug, 2 tree collisions invisible to the clamp), 0 rejected crafts, 0 cooldown-wasted steps, PLACE ok 92%/90%, CRAFT ok 100%.
- v3 arms stopped (armB 33 eps, armD 24, qr 18 with one gear, ctrl 2); last checkpoints kept in runs/_kept. Started fresh v4 arms, same four configurations (B: plain+explorer/exploiter; D: QR+explorer/exploiter+frontier return at drill_admitted; QR: quantile epsilon; ctrl: plain epsilon), seeds 9-12, 100k budgets, schema macro-v2, 125 offline tests.
## 02:02 check 43
- Clock correction: the system time is 02:02; several earlier headings tonight were stamped up to an hour ahead by my estimate. Ordering is right, wall times are not.
- v4 arms launched 02:00 (the earlier launch at "01:41" was these same four processes; the pkill in the v4 launch replaced the v3 set). All four at ~70% CPU in startup, RCON ports 27000/27003 answer, no connection logged yet. `docker ps` itself is slow to respond, so the Docker daemon is under load; watching for connection or crash.
## 02:23 check 44
- v4 arms 20 min in, 3,564 steps total: ok rate 0.962 (v3 was 0.81-0.89), 88 refusals, 46 no_support, 0 cooldown hits. Remaining refusals are legitimate game outcomes or FLE tool behaviour: furnace already holds another ore (11), wood into a furnace slot (8), path not found (13), the insert tool's LuaItemStack full-slot error (5), extract of coal the observation saw but the tool did not (9).
- All 4 alive, 16 GB free (pruner running), plots and dashboard regenerated.
## 02:55 D11 landed; Rainbow arm on the fixed environment
- D11 (129 tests): asymmetric PER priority p = (|delta| * (K if delta>0 else 1))^alpha (Osband-style optimism in the sampler, not the target), per-episode return bonus multiplying an episode's priorities by 1 + B*clip(APS/1000), n-step truncation at episode and excursion boundaries verified.
- qr-v4 stopped (3 eps). rainbow-v4 on 27003: QR-51 + NoisyNets + PER a=0.6 with optimism K=3 and return bonus B=2, n-step 5, gamma 0.99, lr 1e-4, eps floor 0.02, 100k budget, seed 13. If Rainbow moves this time, on the fixed env with a long budget, it is the configuration to scale; if it stays flat with noisy on, the NoisyNets verdict stands.
## 02:53 check 45
- v4 arms 50 min in: armB-v4 8 eps, 2/8 positive, max 218 (a 512-step explorer episode); armD-v4 6 eps, 1/6, max 119; ctrl-v4 7 eps, 5/7, plates at ep6; rainbow-v4 1 ep (just started). Every arm fuels and feeds a furnace within its first two episodes on the fixed environment.
- All 4 alive, 16 GB free, plots and dashboard regenerated. No gears yet (expected before ~5k steps).
- 02:58: host reported low memory and killed two of my hung docker-status commands. Free memory 46%, learners 0.4-0.8 GB each; the pressure is 23.8/24.6 GB swap from idle apps and other sessions, not the runs. Training processes unaffected.
## 03:23 check 46
- v4 arms ~80 min in: rainbow-v4 6 eps, 4/6 positive, last-20 median 44, max 55 (noisy exploration learning this time, unlike both broken-env runs); ctrl-v4 12 eps, 9/12, median 10, max 51, plates at ep6; armD-v4 11 eps, 3/11, max 119, plates at ep10; armB-v4 14 eps, 3/14, max 218.
- All 4 alive, 6 servers answer, 15 GB free. No gears yet in any v4 arm.
## 03:53 check 47
- rainbow-v4 11 eps, 8/11 positive, last-20 median 43, max 92, plates at ep9 (still the best early curve); ctrl-v4 17 eps, 11/17, median 8, max 56; armD-v4 16 eps, 5/16, max 206, plates at ep14; armB-v4 19 eps, 5/19, max 218, plates at ep10 (metrics 4.3 min old; liveness checked).
- 4 learners, 6 servers answer, 15 GB free. No gears yet.
## 04:23 check 48
- rainbow-v4 16 eps, 12/16 positive, last-20 median 44, max 92; ctrl-v4 21 eps, 13/21, median 10, max 103; armD-v4 20 eps, 9/20, max 219; armB-v4 24 eps, 7/24, max 218.
- 4 learners, 6 servers answer, 14 GB free, plots and dashboard regenerated. No gears in any v4 arm at 4-5k steps each.
## 04:53 check 49
- FIRST GEARS on the fixed environment: armD-v4 gears at ep16 then inserter and belt placed at ep23; ctrl-v4 gears and belt at ep22 (plain epsilon, no exploration machinery). rainbow-v4 21 eps, 14/21 positive, median 44, max 300; armB-v4 30 eps, 13/30, max 280; armD-v4 24 eps, 11/24, median 18; ctrl-v4 26 eps, 16/26, median 17.
- 4 learners, 6 servers answer, 14 GB free. No drill crafted yet.
## 05:23 check 50
- FIRST DRILL on the fixed environment: armD-v4 crafted a drill at ep29 (2 episodes with drills), after gears at ep16 and inserter+belt at ep27. Not yet placed. armD-v4 31 eps, 15/31 positive, median 29, max 335; rainbow-v4 27 eps, 16/27, median 28, max 332; ctrl-v4 31 eps, 20/31, median 29, gears 2 eps; armB-v4 35 eps, 15/35, max 320.
- 4 learners, 6 servers answer, 14 GB free, plots and dashboard regenerated.
## 05:30 first drill placements refused on the fixed env; live probe dispatched
- armD-v4 frontier return worked: 2 excursions restored the drill-admissible state, each opened with the forced drill craft, both crafted a drill. Both PLACE drill attempts (2-3 tiles from the player, footprint judged free by our clamp) were refused: "something is in the way or terrain is unplaceable". Our occupancy model disagrees with the server's.
- ctrl-v4 paused (checkpoint runs/ctrl-v4/checkpoint-8000.pt kept) to free 27004 for a live placement probe (D12): 5x5 grid of drill placements around the character, both snapping conventions, 4 directions, with the true blocker queried from the game for each refusal. Fix to follow in fle/rl only.
## 05:50 the drill wall, finally: mining drills need ore under them
- Live probe (research/17-drill-placement-probe.md): 200 drill placements on verified empty grass, 200 refused; the same tool placed drills on iron ore at every snapping and direction. Factorio's manual build check requires a compatible resource inside the drill's mining area. Every drill placement any agent ever attempted (day 1 to now) was on bare ground and could never succeed. Nothing occupied those tiles; our clamp modelled collision but not this positive requirement. Also: the tool moves the character aside, so the player tile is placeable and is no longer masked.
- Fix: the PLACE clamp requires cached ore within the mining area for drills (2x2 burner, 5x5 electric, pumpjack on oil) and moves an unsupported request to the nearest offset over ore; 130 tests. All four arms restarted from their latest checkpoints on the fix (B 5.8k, D 6.9k, rainbow 7k, ctrl 8k steps).
## 05:53 check 51
- All four arms resumed 3 minutes ago on the drill-placement fix (ore-coverage clamp); the two historical refused drill placements are the only drill PLACE records so far. rainbow-v4 32 eps, 20/32 positive, last-20 median 63, max 332; ctrl-v4 32 eps, 21/32, median 29; armB-v4 and armD-v4 producing steps after resume. 4 learners, 6 servers answer, 14 GB free.
## 06:23 check 52
- armD-v4 38 eps, 22/38 positive, last-20 median 118, max 777 (best fresh-arm episode; frontier return + QR head); armB-v4 44 eps, 22/44, max 396 (a 512-step explorer episode), median -1 (mixed horizons); ctrl-v4 37 eps, 25/37, median 26; rainbow-v4 37 eps, 22/37, median 24, max 332.
- 4 learners, 6 servers answer, 14 GB free. No drill PLACE attempted since the ore-coverage fix (30 min ago); the two records are the pre-fix refusals.
## 06:53 check 53
- armD-v4 45 eps, 29/45 positive, last-20 median 92, max 777, deepest drill crafted; rainbow-v4 42 eps, 27/42, median 55; ctrl-v4 42 eps, 30/42, median 42, max 170; armB-v4 49 eps, 24/49, max 396.
- 4 learners, 6 servers answer, 14 GB free. Still no drill PLACE attempt since the ore-coverage fix (1 h); the only two drill placements on record are the pre-fix refusals.
- 07:00 found why the ore-coverage fix is unexercised: the UCB explorer's primary key for PLACE is None on all 232 explorer PLACE actions (the macro-v2 placeable head was never mapped), so PLACE drill is never an unseen pair and gets no first-try forcing; after 6 drill crafts in excursions the explorer placed furnaces, chests and a pipe. D13 dispatched (map PLACE primary to the placeable head; verify every op's primary; drop stale None keys on resume).
- 07:20 D13 landed (133 tests; PLACE keyed on the placeable head, primaries validated for every op, stale None keys dropped on resume). armD-v4 resumed from runs/armD-v4/checkpoint-8000.pt on it; other arms unaffected (their behaviour does not use UCB primaries).
## 07:23 check 54
- armD-v4 50 eps, 34/50 positive, last-20 median 89, max 777; ctrl-v4 48 eps, 36/48, median 69, max 170; rainbow-v4 47 eps, 31/47, median 62, max 332; armB-v4 57 eps, 32/57, median 31, max 399. All four medians rising.
- 4 learners, 6 servers answer, 13 GB free. No drill crafted since armD's D13 resume, so no drill placement on ore attempted yet; the two records remain the pre-fix refusals.
## 07:53 check 55: FIRST DRILL PLACED
- armB-v4 (plain DQN, epsilon port 27000) episode 42: crafted a drill and PLACED it on ore at step 67, then again at 95 and 107 after picking it up; final APS 497 from the furnace line. The ore-coverage clamp moved every request onto ore (requested offset 136 -> executed 144, etc.). The drill was never fuelled (no INSERT into it; the policy laid belts, picked the drill up, waited), so no automated ore yet. First drill placements in the project.
- armD-v4 59 eps, 43/59 positive, last-20 median 146, max 777; ctrl-v4 53 eps, 41/53, median 97; armB-v4 68 eps, 43/68, median 88, max 538; rainbow-v4 54 eps, 38/54, median 80, max 457. 4 learners, 6 servers answer, 13 GB free.
## 08:23 check 56
- armB-v4 79 eps, 54/79 positive, last-20 median 184, max 942 (new fresh-arm max); armD-v4 68 eps, 52/68, median 148, drills crafted 8, placed 0; ctrl-v4 59 eps, 47/59, median 119, max 213; rainbow-v4 60 eps, 44/60, median 136, max 464. All medians still rising.
- 4 learners, 6 servers answer, 12 GB free. Drill fuelled: 0 everywhere.
- 08:30 diagnosis of the fuel rung: in armB's drill episode the agent held no coal after placing the drill (coal extracted earlier had been consumed), so INSERT coal into the drill was never coverable; armD's UCB explorer has selected PLACE drill as an unseen pair 3 times since the fixes (bonus +2) but its post-resume episodes have not yet held a drill at a placement step. Both are budget, not defects.
- 08:35 logging note: v4 step records carry items_delta but no inventory snapshot (dropped when the D10 logging fields were added); ladder and fuel analyses must reconstruct inventories from deltas. Not a training issue.
- 08:40 fuel-rung diagnosis (inventories reconstructed from deltas): armD holds a crafted drill in 445 steps and fuel alongside it in 283, yet PLACE drill was chosen 3 times all run and INSERT into a drill never. UCB's one forced first try per (rung, op, primary) pair is the designed rate and it is too slow for a 2-step conjunction at the deepest rung. D14 dispatched: a frontier forcing window of T steps after reaching the drill rung during which under-tried pairs are chosen uniformly (behaviour only), plus a frontier snapshot at the drill craft with PLACE drill as the first excursion action.
- 09:05 D14 landed (137 tests): --frontier-force-tries 24 (after reaching the drill rung, under-tried pairs at that rung are chosen uniformly for 24 steps; behaviour only) and a frontier snapshot at the drill craft whose first excursion action is PLACE drill. armD-v4 resumed from runs/armD-v4/checkpoint-12000.pt on it.
## 08:53 check 57
- rainbow-v4 67 eps, 51/67 positive, last-20 median 352 (up from 136 an hour ago), max 760; armB-v4 90 eps, 65/90, median 230, max 942; armD-v4 76 eps, 60/76, median 204, max 1024 (new fresh-arm max); ctrl-v4 66 eps, 54/66, median 122, max 245.
- 4 learners, 6 servers answer, 15 GB free. Drill fuelled: 0 everywhere. armD's frontier forcing window has not fired yet since the resume (no drill rung reached in its new episodes).
## 09:23 check 58
- rainbow-v4 74 eps, 58/74 positive, last-20 median 454, max 870 (best median of all fresh arms); armB-v4 101 eps, 76/101, median 428, max 1169 (new fresh-arm max); armD-v4 84 eps, 68/84, median 191, max 1024; ctrl-v4 73 eps, 61/73, median 133, first drill crafted.
- 4 learners, 6 servers answer, 14 GB free, plots and dashboard regenerated. Drill fuelled 0 everywhere; armD's forcing window has fired 0 actions since the resume.
- 09:30 forcing-window check: the explorer's rung does advance (rung 3 = drill crafted seen 441 times over the run) but no transition into rung 3 has happened in the 1,500 actions since the D14 resume, so the window has had nothing to fire on. No defect; waiting.
## 09:53 check 59
- rainbow-v4 80 eps, 64/80 positive, last-20 median 587, max 870; armB-v4 112 eps, 87/112, median 574, max 1388 (new fresh-arm max); armD-v4 92 eps, 76/92, median 195, max 1024; ctrl-v4 79 eps, 67/79, median 138.
- 4 learners, 6 servers answer, 15 GB free. Drill fuelled 0 everywhere; armD's forcing window still awaiting its first drill-rung transition since the resume.
## 10:23 check 60
- rainbow-v4 87 eps, 71/87 positive, last-20 median 726, max 1316; armB-v4 123 eps, 98/123, median 576, max 1699 (new fresh-arm max); ctrl-v4 86 eps, 74/86, median 154, max 447; armD-v4 104 eps, 87/104, median 125, 9th drill crafted; its forcing window has fired for the first time (2 actions).
- 4 learners, 6 servers answer, 12 GB free, plots and dashboard regenerated. Drill fuelled 0 everywhere.
- 10:30 forcing window verified: after the 9th drill craft the explorer chose PLACE drill twice, both no_support "place_no_supported_offset": no offset within build reach covers ore because the agent stands at its furnaces, away from any patch. PLACE has no approach step (HARVEST and the entity ops do). D15 dispatched: for resource-requiring placeables, walk to the nearest compatible ore tile within 160 tiles, then place (navigation absorbed into the macro, same invariant as HARVEST).
- 10:45 arms renamed by learner and exploration (dqn-per, dqn-per-ucbexplorer, qrdqn-per-ucbexplorer-frontier, rainbow-optimisticper-n5); run directories unchanged; mapping in README and overnight_v1.md.
## 10:33 D15 landed: PLACE approaches ore for drills; all arms restarted
- D15 (140 tests): resource-requiring placeables (drills, pumpjack) walk to the nearest compatible ore within 160 tiles before placing when no offset within reach covers ore; logged as clamp_reason place_approach_to_resource with approach_tiles. All four arms resumed from their latest checkpoints (dqn-per runs/ctrl-v4/checkpoint-22000.pt, dqn-per-ucbexplorer runs/armB-v4/checkpoint-20000.pt, qrdqn-per-ucbexplorer-frontier runs/armD-v4/checkpoint-16000.pt, rainbow-optimisticper-n5 runs/rainbow-v4/checkpoint-16000.pt). dqn-per-ucbexplorer also gains the 24-step forcing window.
## 11:05 owner ruling: no relocation of sampled actions
- The ore-coverage clamp (moves a drill to the nearest offset over ore) and the approach-to-ore (walks to ore before placing) substitute the executor's choice for the policy's and break the MDP. Rule restated: masks remove only deterministic refusals; navigation only to a target the policy chose (HARVEST slot, entity anchor); infeasible actions are zero-reward steps the policy learns from, as the LLM agents do with place_entity errors. D16 dispatched: remove both relocations, reclassify the item/anchor/CONNECT clamps as no_support (no substitution), keep quantity clamp behind a default-on flag as a documented judgement call, verify the grid's resource channel aligns with the offset head so placement over ore is learnable (harvest at a patch, then place within reach).
## 11:40 FIRST FUELLED DRILL (dqn-per-ucbexplorer, explorer port 27001, episode 47)
- Drill placed on ore at step 172 (under the relocation code), coal inserted into it at steps 173-187 and 191-197 (19 inserts), final APS 1375 (arm max). Rewards after fuelling include +18/+19 on INSERT steps and +19 on an EXTRACT with no hand harvest in between, consistent with drill output being smelted; verifying against the drill's entity status before calling it automated ore. Tainted for the ladder (relocation), but the policy chose the fuel step on its own.
- 11:55 correction on the fuelled drill: the drill in dqn-per-ucbexplorer ep47 was fuelled 19 times but picked up at step 189 and re-placed; the +18/+19 rewards that followed came from a boiler craft and furnace output, and the working-drill detector never fired. Drill fuelled: yes (2 arms, both under relocation code); automated ore from a drill: not yet.
- 11:55 dashboard plots regenerated with descriptive labels (tests/benchmarks/refresh_plots.sh now does plots + dashboard in one call; used by the checks from here).
## 10:53 check 61
- rainbow-optimisticper-n5 92 eps, 76/92 positive, last-20 median 784, max 1316; dqn-per-ucbexplorer 129 eps, 104/129, median 623, max 1699, drill crafted/placed/fuelled 2/2/1; qrdqn-per-ucbexplorer-frontier 110 eps, 93/110, median 114, drill 9/1/1; dqn-per 90 eps, 78/90, median 170.
- 4 learners, 6 servers answer, 14 GB free, plots and dashboard regenerated. Drill counts are on relocation code and count as tainted until the D16 restart. No automated ore from a drill yet.
- 11:15 plots now carry environment-version boundary markers per arm; version table added to overnight_v1.md. Decision: after D16 lands, all arms start FRESH (not resume) for a single-environment comparison.
## 11:20 D16 landed; all arms started FRESH on the relocation-free environment
- D16 (140 tests): no relocation anywhere. PLACE uses the exact sampled offset (footprint/reach failures are no_support, a drill on grass reaches the tool and is refused with zero reward); item, anchor, rotatable and CONNECT substitutions are off (no_support instead); quantity clamp kept behind --clamp-quantity default on as a documented judgement call; cooldowns kept; grid resource channel verified to align with the offset head.
- Relocation-era runs moved to runs/*-relocation-era. Fresh runs, seeds 21-24, 100k budgets, in runs/dqn-per, runs/dqn-per-ucbexplorer, runs/qrdqn-per-ucbexplorer-frontier, runs/rainbow-optimisticper-n5. Dashboard and plots now read these; the relocation-era Rainbow curve is kept as a grey reference. This is the single-environment comparison and the baseline for the compute run.
## 11:30 Codex behaviour trace read (research/18) landed; two corrections
- The "fuelled drill" in the relocation era was the item clamp's doing: of 19 coal inserts into the drill, the policy sampled coal only 4 times; the executor substituted coal for stone, plates, furnaces and boilers. Not the agent's achievement. Also: unrelocated drill placement would have been feasible in 55% of post-craft positions, but only 1 of 7 sampled placements was correct as sampled.
- UCB's coefficient reset to 0.75 on every resume (behaviour time not persisted), so each restart re-explored; the forcing window retried identical invalid INSERT variants. Remaining defects: repeated no_support EXTRACT on empty entities, EXTRACT of drill fuel offered though the tool refuses it, CONNECT predicate missing pole consumption. D17 dispatched for all of these, plus ore-aware offset SAMPLING (behaviour side, executed as sampled) for forced drill placements.
- Rainbow's edge confirmed: 4.7 APS per decision vs 0.75 for dqn-per, from a concentrated one-furnace stone-brick shuttle; its optimistic sampler is stable (positive-delta share 0.57, priorities 2:1) though Q and loss still rise.
## Check 62, 11:32 (first check of the fresh arms; 12 minutes in)
- All four learners alive, all six env logs under 1 min old. Episodes: dqn-per 4 (median 10, max 33, furnace fuelled and fed at ep1), dqn-per-ucbexplorer 5 (all -1, furnace fuelled and fed at ep1), qrdqn-per-ucbexplorer-frontier 4 (all -1, furnace fed at ep1), rainbow-optimisticper-n5 3 (max 2, furnace fuelled and fed at ep0). Nothing to read yet; first meaningful medians after ~30 episodes per arm (~2.5 h).
- Dashboard gained an archived history section (pre-11:20 curves regenerated into results/overnight_v1_relocation_era/); README repointed to the renamed run directories. Plots refreshed 11:31.
- Codex D17 (UCB c persistence, ore-aware forced offsets without relocation, empty-EXTRACT/drill-fuel masks, CONNECT predicate, candidate telemetry) still running.
## 11:40 D17 landed; all four arms restarted fresh on it
- Codex D17 reviewed: UCB c persisted per port, forcing by distinct trials, ore-aware sampling of forced drill offsets (executed as sampled; greedy untouched), empty-slot and drill-fuel EXTRACT masks with immediate cooldown, CONNECT predicate accepts pole consumption, admitted/n0 pair counts in action logs. 147 tests pass, ruff clean, fake-env resume check shows behaviour steps continuing.
- No checkpoints existed (700 to 1100 env steps), so killed all four and relaunched fresh at 11:40 with the same seeds and flags; D16 fragments moved to runs/_scratch/*-d16-15min. Comparison stays within one environment version.
## Check 63, 11:53 (13 minutes after the D17 restart)
- All four learners alive, six env logs fresh, no tracebacks. Steps since 11:40: dqn-per 403, dqn-per-ucbexplorer 383 (two ports), qrdqn-per-ucbexplorer-frontier 297, rainbow-optimisticper-n5 254. One episode each; Rainbow's first scored 31, dqn-per 0, the UCB arms -1. Furnace fuelled and fed at ep0 for dqn-per and Rainbow.
- D17 telemetry confirmed in the UCB action logs (admitted_pair_count present). Too early for medians or ladder comparison; next plot refresh at 12:13.
- Refusal profile after D17: Rainbow's 19 empty-entity EXTRACT refusals in 262 steps are all distinct full actions (item and quantity vary), so the exact-action cooldown is doing its job and there are no identical repeats; the residual comes from the independent entity head picking an empty slot while another slot has items, a zero-reward step the policy must learn. UCB arm's 36 INSERT rejections are furnace-content conflicts (stone into a copper-ore furnace and the reverse), game-contingent and learnable.
## 12:22 owner decision: stop the four-arm comparison, run Rainbow macro vs Rainbow bare
- Killed dqn-per, dqn-per-ucbexplorer, qrdqn-per-ucbexplorer-frontier and the one-server Rainbow (30 min on D17, no checkpoints yet; fragments in runs/_scratch/*-d17-30min). Started rainbow-optimisticper-n5-macro on 27003,27004,27005, seed 24, 300k steps.
- Codex D18 dispatched: `--action-regime bare` (no move_to in HARVEST, no approach in entity ops or CONNECT, out-of-reach targets and slots masked as deterministic refusals). The bare twin starts on 27000,27001,27002 when it lands.
- Throughput question answered for the owner at 12:05: 2.2 steps/s over six servers, 1 to 2 s per tool call, CPU-saturated Mac (load 22 on 14 cores; an unrelated toy_lit.py process uses two cores). Recommended a 48 to 64 vCPU x86 Linux VM without GPU.
## 16:23 status (D18 landed at ~16:15; bare Rainbow launched 16:18)
- rainbow macro actions: 99 episodes, 11.1k steps, APS median 104, max 619, 63/99 positive; furnace fuelled and fed 68/99; no plates extracted, no drill. All episodes end on the one-game-hour tick cap at a median of 115 decisions.
- rainbow bare actions: started 16:18 on 27000-27002 after 151 tests pass; first steps pending. D18 report: MOVE travels 16 tiles per action in 8 directions; target head carries dx, dy, distance to 160 tiles, so walking to ore is representable.
- Program section written to overnight_v1.md ("The program as of 16:25"): runs, achievements, convenience inventory, state and action space, episode definition. Codex trace read of the macro run dispatched (research/19).
## Check 64, 16:23 (covers the eight scheduled checks queued between 12:43 and 16:13, which did not run while the session was busy with D18)
- Both learners alive, all six env logs fresh, no tracebacks. rainbow macro actions: 102 episodes, 11.3k steps, last-20 APS median 225, max 619, 66/102 positive, checkpoint-8000 kept; ladder unchanged (furnace fuelled and fed, chest placed; no plates, no drill). rainbow bare actions: 100 steps, 0 episodes, all MOVE/WAIT/RESEARCH so far.
- Per-server decision budget differs: port 27003 has 45 episodes at ~85 decisions each, 27005 has 26 at ~146, from the same learner, because episodes end on the one-game-hour tick cap and the servers run at different speeds. Recorded under "Episode definition" in overnight_v1.md; to be fixed by defining the episode explicitly (one game hour, at most 256 decisions) and reporting decisions per episode.
- Plots and dashboard refreshed 16:23. Codex trace read of the macro run (research/19) still running.
## Check 65, 16:24 (queued check, one minute after 64)
- Both learners alive, logs fresh. Macro unchanged since 16:23. Bare: 195 steps, 125 MOVE (median 15.6 tiles per MOVE, as designed), 38 WAIT, 30 RESEARCH, and the first 2 HARVEST successes yielding 40 items; the character has wandered up to 254 tiles from spawn. All 195 steps ok, so the bare masks are admitting only feasible actions.
## 16:39 D19 landed (not deployed; owner: let the runs go)
- Cooldown root cause confirmed and fixed offline: the support fingerprint included furnace contents, which change every step while smelting, so the exact-action cooldown was invalidated immediately (zero cooldown hits in 11k steps). New fingerprint: player tile, inventory, entity unit set, plus only the refused action's relevant slot state. Also: observation-backed INSERT op mask (no visible entity accepts any held item given its input), PLACE executed_offset telemetry. 156 tests pass, ruff clean. Kept for the next resume point or the cloud box; both Rainbows continue on D18.
- FLE tool behaviour documented by Codex: the insert tool labels any zero-insert as "Inventory is full" and reads a stack without valid_for_read (fle/env/tools/agent/insert_item/server.lua); not ours to change, reported upstream-worthy.
- Owner has ~25k USD of Google Cloud credits; compute recipe written to docs/rl/specs/compute-recipe.md (c3d-highcpu-60, 24 servers, smoke test first, then 3 seeds per regime).
## Check 66, 16:53
- Both learners alive, logs fresh, no tracebacks, disk 9.4 GB free. rainbow macro actions: 114 episodes, 12.5k steps, last-20 APS median 268 (up from 225), max 709 (new best), 78/114 positive, checkpoint-12000; ladder unchanged (no plates extracted, no drill). rainbow bare actions: 17 episodes, 1.6k steps, all APS 0 or -1; episodes with furnaces placed exist but none fuelled and fed yet.
- Plots and dashboard refreshed 16:53. Owner asked whether the reward is automated or raw production score: automated only (score()[1]); assessment given, no change made, decision deferred to 20k bare steps.
## Check 67, 17:23
- Both learners alive, logs fresh, no tracebacks. rainbow macro actions: 127 episodes, last-20 APS median 457 (268 at 16:53), max 862 (new best), 91/127 positive; ladder unchanged. rainbow bare actions: 36 episodes, 2k steps, checkpoint-2000, all APS 0 or -1, no successful INSERT yet; player score median has fallen to 0 (harvesting is being unlearned, consistent with the -1 charge for hand work).
- Plots and dashboard refreshed 17:23. Compute memory note updated by a parallel session: on-demand c3d-highcpu-60 is 2.24 USD/h in us-central1, Spot quota is 0, sizing to come from a measured throughput sweep rather than the guessed 24 servers.
## Check 68, 17:53
- Both learners alive, logs fresh, no tracebacks. rainbow macro actions: 142 episodes, 14k steps, last-20 APS median 553 (457 at 17:23), max 1045 (first four-digit episode), 106/142 positive, checkpoint-14000; ladder unchanged. rainbow bare actions: 57 episodes, 4k steps, checkpoint-4000, all APS 0 or -1.
- Bare has collapsed to inaction: last 900 steps are 49% MOVE, 30% WAIT, 21% RESEARCH, zero HARVEST, zero PLACE, zero INSERT. The -1 charge on hand work has taught it not to touch anything; with nothing positive ever seen, the noisy-net exploration is not reaching a fuelled furnace. This is the predicted failure of the automated-score reward without absorbed navigation. No change made; decision rests with the owner (bare with player-score reward as a third arm was the proposal).
- No plot refresh this check (last at 17:23); next at 18:13.
## Check 69, 18:23
- Both learners alive, logs fresh, no tracebacks, disk 10 GB free. rainbow macro actions: 155 episodes, 16k steps, last-20 APS median 702 (553 at 17:53), max 1075, 119/155 positive, checkpoint-16000; ladder unchanged (hand-fed furnaces only, no plates extracted, no drill). rainbow bare actions: 79 episodes, 6k steps, checkpoint-6000, every episode APS 0 or -1, player score 0 in the last 20; inaction collapse persists.
- Plots and dashboard refreshed 18:23. Awaiting the owner's decision on the bare arm (keep as negative result, or restart with player-score reward as a third arm).
## Check 70, 18:53
- Both learners alive, logs fresh, no tracebacks. rainbow macro actions: 171 episodes, 18k steps, last-20 APS median 883 (702 at 18:23), max 1085, 135/171 positive, checkpoint-18000; ladder unchanged. rainbow bare actions: 100 episodes, 8k steps, checkpoint-8000, 0/100 positive, inaction collapse persists.
- No plot refresh this check (last 18:23); next at 19:13.
## Check 71, 19:23
- Both learners alive, logs fresh, no tracebacks, disk 10 GB free. rainbow macro actions: 186 episodes, last-20 APS median 965 (883 at 18:53), max 1197, 150/186 positive, checkpoint-18000 (20000 due shortly); ladder unchanged. rainbow bare actions: 121 episodes, 0/121 positive; one recent episode reached player score 328 (a new high for bare), with 11 harvests, 12 crafts, 7 placements and one successful INSERT, and a second recent episode with one INSERT; the inserts were not a fuel-plus-ore pair, so APS stayed at -1. Exploration on port 27000 is not fully collapsed.
- Plots and dashboard refreshed 19:23.
## Check 72, 19:53
- Both learners alive, logs fresh, no tracebacks. rainbow macro actions: 202 episodes, 20k steps, last-20 APS median 1013 (965 at 19:23; first four-digit median), max 1265, 165/202 positive; checkpoint-20000 copied to runs/_kept/rainbow-macro-checkpoint-20000.pt. Ladder unchanged. rainbow bare actions: 143 episodes, 0/143 positive, checkpoint-8000.
- No plot refresh this check (last 19:23); next at 20:13.
## Check 73, 20:23
- Both learners alive, logs fresh, no tracebacks, disk 8.7 GB free. rainbow macro actions: 215 episodes, 22k steps, last-20 APS median 1045 (1013 at 19:53), max 1265, 178/215 positive, checkpoint-22000; ladder unchanged. rainbow bare actions: 163 episodes, 12k steps, 0/163 positive; best player score rose to 423, so exploration still touches resources without closing the fuel-plus-ore loop.
- Plots and dashboard refreshed 20:23.
## Check 74, 20:53
- Both learners alive, logs fresh, no tracebacks. rainbow macro actions: 231 episodes, ~23.5k steps, last-20 APS median 1085 (1045 at 20:23), max 1265, 194/231 positive; ladder unchanged. rainbow bare actions: 185 episodes, 14k steps, checkpoint-14000, 0/185 positive.
- No plot refresh this check (last 20:23); next at 21:13.
## Check 75, 21:23
- Both learners alive, logs fresh, no tracebacks. rainbow macro actions: 246 episodes, 24k steps, last-20 APS median 1122 (1085 at 20:53), max 1265 unchanged for four checks, 209/246 positive, checkpoint-24000; ladder unchanged. rainbow bare actions: 207 episodes, 16k steps, checkpoint-16000, 0/207 positive.
- Plots and dashboard refreshed 21:23. Disk free fell 8.7 to 6.9 GB in one hour; investigating the source before it becomes a repeat of the disk-full incident.
## Check 76, 21:53
- Both learners alive, logs fresh, no tracebacks. rainbow macro actions: 259 episodes, 26k steps, last-20 APS median 1164 (1122 at 21:23), max 1314 (new best), 222/259 positive, checkpoint-26000; ladder unchanged. rainbow bare actions: 229 episodes, 18k steps, checkpoint-18000, 0/229 positive.
- Disk stable at 6.9 GB free over the last half hour, so the earlier 1.8 GB drop was a one-off outside our runs (our two run directories total 830 MB). No plot refresh this check; next at 22:13.
- 21:56 addendum: the disk drop is swap. Swap is 24.9 of 26.6 GB used; the disk scan was killed by the OS for low memory. Learners are 670 MB RSS each, Docker 1.6 GB; the rest is other applications swapped out. Heavy ad-hoc analysis on this machine is off until the box arrives; checks stay to the light status scan and plot refresh.
- 2026-09-17 evening, throughput study run on Google Cloud and reported in `docs/rl/results/vm_throughput.md`. The c3d-highcpu-60 plan above is superseded: C3D/C4D/C4 are capped at 24 vCPUs per region by `CPUS_PER_VM_FAMILY`, so it was never buildable. Measured winner is `h4d-standard-192` at 6.44 CHF per million steps, 128 servers (sizing rule 0.7 x vCPUs, 1.4 vCPU per environment). SMT does carry a Factorio server, about a third of a core. The learner, not the environment, is the ceiling: 16.15 updates/s at 2 torch threads against 238 steps/s from 128 servers. Episodes on native hardware end on the 256-step cap, not the tick cap that bound on the Mac, so episode length is not comparable across the move.
## Check 77, 22:23
- Both learners alive, logs fresh, no tracebacks. rainbow macro actions: 275 episodes, 28k steps, last-20 APS median 1176 (1164 at 21:53), max 1314, 238/275 positive, checkpoint-28000; ladder unchanged. rainbow bare actions: 250 episodes, 20k steps reached, 0/250 positive; best player score 697 (exploration still harvests occasionally); checkpoint-20000 copied to runs/_kept/rainbow-bare-checkpoint-20000.pt as the artifact of the negative result.
- 20k-step decision mark for the bare arm reached: no positive episode. Owner's call stands (keep as negative result, or restart bare with player-score reward as a third arm). Plots and dashboard refreshed 22:23. Disk 7.7 GB free, swap 24.9 of 25.6 GB.
## Check 78, 22:53
- Both learners alive, logs fresh, no tracebacks. rainbow macro actions: 288 episodes, 28k+ steps, last-20 APS median 1192 (1176 at 22:23), max 1453 (new best), 251/288 positive; ladder unchanged. rainbow bare actions: 270 episodes, 22k steps, checkpoint-22000, 0/270 positive.
- Disk: free fell 7.7 to 5.8 GB in 30 min. Cause located: Docker's disk image (Docker.raw) is 290 GB of actual data on a 926 GB disk, and swap grew another 1 GB. Reversible relief applied: gzipped the large logs of finished runs (358 MB freed, 6.1 GB free at 22:56); current runs and _kept untouched. `docker system df` running to see what in Docker is reclaimable; no prune without knowing what it removes. If free space drops under 2 GB before that, the fallback is `docker builder prune` and `docker image prune` (re-downloadable content only), then a note here.
- No plot refresh this check (last 22:23); next at 23:13.
## Check 79, 23:23
- Both learners alive, logs fresh, no tracebacks. rainbow macro actions: 303 episodes, 30k steps, last-20 APS median 1250 (1192 at 22:53), max 1453, 266/303 positive; checkpoint-30000 copied to runs/_kept/rainbow-macro-checkpoint-30000.pt; ladder unchanged. rainbow bare actions: 291 episodes, 24k steps, 0/291 positive; one recent episode reached player score 3183 (previous best 697), inspected below.
- Plots and dashboard refreshed 23:23. Disk 6.0 GB free (6.5 at 23:03); swap-driven fluctuation, no action.
- Bare is waking up, not collapsing: eight of the last ~35 episodes have player scores 558 to 3183, from sustained hand harvesting (72 HARVEST in one episode gaining 1,061 coal; 47 HARVEST gaining 613 coal; several with 180 to 340 iron ore). One episode on 27001 crafted 43 furnaces and placed 12 without inserting. The policy has learned to walk to resources with MOVE and harvest them, and is now crafting and placing furnaces; the missing link is still fuel plus ore into one placed furnace. APS remains 0 or -1 on every episode.
## Check 80, 23:53
- Both learners alive, logs fresh, no tracebacks, disk 5.9 GB free. rainbow macro actions: 317 episodes, 32k steps, last-20 APS median 1256 (1250 at 23:23), max 1453, 280/317 positive, checkpoint-32000; ladder unchanged. rainbow bare actions: 312 episodes, 24k+ steps, 0/312 positive.
- Bare's harvesting burst has receded: last 21 episodes are 50% MOVE, 30% RESEARCH, 17% WAIT, 3% HARVEST, no INSERT; player scores top out at 576. Consistent with hand work being charged -1 under the automated score: the policy samples it, gets nothing, and backs off again.
- No plot refresh this check (last 23:23); next at 00:13.
## Check 81, 2026-09-18 00:23
- Both learners alive, logs fresh, no tracebacks, disk 6.0 GB free. rainbow macro actions: 332 episodes, 34k steps, last-20 APS median 1281 (1256 at 23:53), max 1453, 295/332 positive, checkpoint-34000; ladder unchanged. rainbow bare actions: 331 episodes, 26k steps, checkpoint-26000, 0/331 positive.
- Plots and dashboard refreshed 00:23.
## Check 82, 00:53
- Both learners alive, logs fresh, no tracebacks. rainbow macro actions: 346 episodes, ~35k steps, last-20 APS median 1301 (1281 at 00:23), max 1453, 309/346 positive; ladder unchanged. rainbow bare actions: 353 episodes, 28k steps, checkpoint-28000, 0/353 positive.
- Disk 5.0 GB free (6.0 at 00:23); swap-driven, watching; fallback (Docker build-cache and image prune) at 2 GB. No plot refresh this check; next at 01:13.
## Check 83, 01:23
- Both learners alive, logs fresh, no tracebacks. rainbow macro actions: 359 episodes, 36k steps, last-20 APS median 1312 (1301 at 00:53), max 1453, 322/359 positive, checkpoint-36000; ladder unchanged. rainbow bare actions: 373 episodes, 30k steps, checkpoint-30000, 0/373 positive.
- Plots and dashboard refreshed 01:23. Disk 4.7 GB free (5.0 at 00:53).
## Check 84, 01:53
- Both learners alive, logs fresh, no tracebacks. rainbow macro actions: 374 episodes, 38k steps, last-20 APS median 1301 (1312 at 01:23; first flat-to-down step), max 1457 (new best by 4), 337/374 positive, checkpoint-38000; ladder unchanged. rainbow bare actions: 393 episodes, 32k steps, checkpoint-32000, 0/393 positive.
- Disk 8.6 GB free (4.7 at 01:23): swap shrank; the earlier drops were swap growth, confirmed. No plot refresh this check; next at 02:13.
## Check 85, 02:23
- Both learners alive, logs fresh, no tracebacks, disk 8.6 GB free. rainbow macro actions: 387 episodes, ~39k steps, last-20 APS median 1300 (1301 at 01:53; plateau holding), max 1457, 350/387 positive; ladder unchanged. rainbow bare actions: 413 episodes, 34k steps, checkpoint-34000, 0/413 positive.
- Plots and dashboard refreshed 02:23.
## Check 86, 02:53
- Both learners alive, logs fresh, no tracebacks, disk 8.3 GB free. rainbow macro actions: 400 episodes, 40k steps, last-20 APS median 1347 (1300 at 02:23), max 1457, 363/400 positive; checkpoint-40000 copied to runs/_kept/rainbow-macro-checkpoint-40000.pt; ladder unchanged. rainbow bare actions: 432 episodes, 34k+ steps, 0/432 positive.
- No plot refresh this check (last 02:23); next at 03:13.
## 03:13 owner: stop all; something else runs overnight
- Both learners stopped at 03:13. Final: rainbow macro actions 408 episodes, 40k steps, last-20 APS median 1347, all-episode median 731, max 1457, 371/408 positive, hand-fed furnaces only (no plates extracted, no drill). rainbow bare actions 442 episodes, 34k+ steps, 0/442 positive, learned to walk to and harvest resources and to craft and place furnaces but never fuelled and fed one.
- Checkpoints kept in runs/_kept/: rainbow-macro at 20k, 30k, 40k (final), rainbow-bare at 20k and final (34k). Run directories intact with env, action and metric logs. Docker servers left running for the owner's overnight job. Half-hourly checks stopped.
