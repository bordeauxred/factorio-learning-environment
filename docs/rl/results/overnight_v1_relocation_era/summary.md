# First curves: summary

| run | half | final APS | player PS | invalid rate | episodes with APS > 0 |
|---|---|---|---|---:|---:|
| dqn-per | first | n=45 median 22.0 [q1 -1.0, q3 56.0] max 170.0 | n=45 median 1093.0 [q1 853.5, q3 1293.0] max 2304.0 | 0.02 | 33/45 |
| dqn-per | second | n=46 median 144.0 [q1 111.0, q3 214.8] max 548.0 | n=46 median 1903.0 [q1 1562.2, q3 2634.2] max 5719.0 | 0.04 | 46/46 |
| dqn-per | all | n=91 median 96.0 [q1 17.0, q3 162.0] max 548.0 | n=91 median 1416.0 [q1 1064.0, q3 1954.0] max 5719.0 | 0.02 | 79/91 |
| rainbow-optimisticper-n5 | first | n=46 median 47.0 [q1 -1.0, q3 80.0] max 332.0 | n=46 median 1564.5 [q1 913.2, q3 2339.2] max 6206.0 | 0.04 | 30/46 |
| rainbow-optimisticper-n5 | second | n=46 median 466.0 [q1 282.0, q3 778.8] max 1316.0 | n=46 median 4833.0 [q1 3633.2, q3 5814.0] max 8059.0 | 0.04 | 46/46 |
| rainbow-optimisticper-n5 | all | n=92 median 114.0 [q1 43.2, q3 467.0] max 1316.0 | n=92 median 2760.5 [q1 1247.2, q3 4948.5] max 8059.0 | 0.04 | 76/92 |
| dqn-per-ucbexplorer | first | n=65 median 40.0 [q1 -1.0, q3 136.5] max 942.0 | n=65 median 918.0 [q1 471.0, q3 1747.5] max 4642.0 | 0.04 | 40/65 |
| dqn-per-ucbexplorer | second | n=65 median 574.0 [q1 261.0, q3 940.5] max 1699.0 | n=65 median 2660.0 [q1 1867.5, q3 3785.0] max 6408.0 | 0.04 | 65/65 |
| dqn-per-ucbexplorer | all | n=130 median 221.0 [q1 39.8, q3 574.2] max 1699.0 | n=130 median 1814.5 [q1 697.5, q3 2849.0] max 6408.0 | 0.04 | 105/130 |
| qrdqn-per-ucbexplorer-frontier | first | n=56 median 110.0 [q1 -1.0, q3 204.8] max 1024.0 | n=56 median 833.0 [q1 322.0, q3 2046.2] max 8609.0 | 0.06 | 40/56 |
| qrdqn-per-ucbexplorer-frontier | second | n=56 median 103.0 [q1 55.0, q3 197.5] max 363.0 | n=56 median 1399.0 [q1 860.0, q3 1851.5] max 2707.0 | 0.04 | 55/56 |
| qrdqn-per-ucbexplorer-frontier | all | n=112 median 107.0 [q1 27.5, q3 200.5] max 1024.0 | n=112 median 1079.5 [q1 578.2, q3 1930.0] max 8609.0 | 0.05 | 95/112 |
| dqn-per-v1env | first | n=152 median 25.5 [q1 1.2, q3 108.8] max 1189.0 | n=152 median 1050.5 [q1 588.0, q3 1778.8] max 7479.0 | 0.29 | 115/152 |
| dqn-per-v1env | second | n=152 median 1192.0 [q1 782.8, q3 1495.5] max 1929.0 | n=152 median 5005.0 [q1 3991.0, q3 5918.5] max 8076.0 | 0.13 | 151/152 |
| dqn-per-v1env | all | n=304 median 375.0 [q1 23.0, q3 1194.2] max 1929.0 | n=304 median 2938.5 [q1 1028.5, q3 5151.8] max 8076.0 | 0.20 | 266/304 |
| random | first | n=6 median -1.0 [q1 -1.2, q3 1.2] max 8.0 | n=6 median 947.0 [q1 792.2, q3 1064.0] max 1085.0 | 0.15 | 1/6 |
| random | second | n=6 median -1.0 [q1 -2.0, q3 0.8] max 6.0 | n=6 median 999.5 [q1 839.8, q3 1336.5] max 1677.0 | 0.16 | 1/6 |
| random | all | n=12 median -1.0 [q1 -1.8, q3 -1.0] max 8.0 | n=12 median 999.5 [q1 818.0, q3 1078.0] max 1677.0 | 0.16 | 2/12 |
