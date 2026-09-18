# First curves: summary

| run | half | final APS | player PS | invalid rate | episodes with APS > 0 |
|---|---|---|---|---:|---:|
| ppo | first | n=165 median -1.0 [q1 -1.0, q3 -1.0] max 23.0 | n=165 median 137.0 [q1 62.0, q3 297.0] max 780.0 | 0.43 | 3/165 |
| ppo | second | n=166 median 0.0 [q1 -1.0, q3 0.0] max 0.0 | n=166 median 3.0 [q1 0.0, q3 62.0] max 237.0 | 0.32 | 0/166 |
| ppo | all | n=331 median -1.0 [q1 -1.0, q3 0.0] max 23.0 | n=331 median 62.0 [q1 0.0, q3 144.0] max 780.0 | 0.37 | 3/331 |
| dqn | first | n=79 median -1.0 [q1 -1.0, q3 42.0] max 116.0 | n=79 median 541.0 [q1 344.0, q3 710.0] max 1904.0 | 0.28 | 33/79 |
| dqn | second | n=80 median 162.0 [q1 94.2, q3 388.0] max 723.0 | n=80 median 882.0 [q1 603.8, q3 2353.8] max 4378.0 | 0.29 | 76/80 |
| dqn | all | n=159 median 54.0 [q1 -1.0, q3 162.0] max 723.0 | n=159 median 653.0 [q1 467.0, q3 978.0] max 4378.0 | 0.28 | 109/159 |
| random | first | n=75 median -1.0 [q1 -1.0, q3 -1.0] max 0.0 | n=75 median 586.0 [q1 439.0, q3 732.0] max 1197.0 | 0.28 | 0/75 |
| random | second | n=75 median -1.0 [q1 -1.0, q3 -1.0] max 0.0 | n=75 median 611.0 [q1 474.0, q3 777.0] max 1156.0 | 0.29 | 0/75 |
| random | all | n=150 median -1.0 [q1 -1.0, q3 -1.0] max 0.0 | n=150 median 598.5 [q1 447.0, q3 753.8] max 1197.0 | 0.28 | 0/150 |
