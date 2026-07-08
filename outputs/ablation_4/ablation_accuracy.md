| variant | setting | dis_pen | adaptive | best_epoch | recall@20 | ndcg@20 | hit@20 | recall@50 | ndcg@50 | hit@50 | delta_recall@20 | delta_ndcg@20 | delta_recall@50 | delta_ndcg@50 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | full NeuMFDICE | 0.01 | True | 14 | 0.1551 | 0.1038 | 0.5341 | 0.2812 | 0.1447 | 0.7394 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| no-dis | remove discrepancy penalty | 0.0 | True | 5 | 0.1583 | 0.1066 | 0.5413 | 0.2860 | 0.1484 | 0.7393 | 0.0033 | 0.0028 | 0.0047 | 0.0037 |
| no-adaptive | disable adaptive training | 0.01 | False | 10 | 0.1465 | 0.0979 | 0.5194 | 0.2695 | 0.1383 | 0.7216 | -0.0086 | -0.0058 | -0.0117 | -0.0065 |
| low-dis | weaker discrepancy penalty | 0.001 | True | 5 | 0.1569 | 0.1054 | 0.5448 | 0.2871 | 0.1479 | 0.7427 | 0.0018 | 0.0016 | 0.0058 | 0.0031 |
| high-dis | stronger discrepancy penalty | 0.05 | True | 6 | 0.1427 | 0.0935 | 0.5034 | 0.2650 | 0.1342 | 0.7131 | -0.0124 | -0.0103 | -0.0162 | -0.0105 |
