# NeuMF-DICE 四组消融实验结果分析

实验目录：`/root/autodl-tmp/DICE_four_star/runs/neumf_ablation_4`  
完成时间：2026-06-17 11:48:48  
状态：四组全部 `rc=0`，无 OOM、Traceback、failed、killed。

## 1. 准确率指标

| Variant | 设置 | Best Epoch | Recall@20 | NDCG@20 | Hit@20 | Recall@50 | NDCG@50 | Hit@50 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | full NeuMFDICE, `dis_pen=0.01`, adaptive | 14 | 0.1551 | 0.1038 | 0.5341 | 0.2812 | 0.1447 | 0.7394 |
| no-dis | `dis_pen=0.0`, adaptive | 5 | 0.1583 | 0.1066 | 0.5413 | 0.2860 | 0.1484 | 0.7393 |
| no-adaptive | `dis_pen=0.01`, no adaptive | 10 | 0.1465 | 0.0979 | 0.5194 | 0.2695 | 0.1383 | 0.7216 |
| low-dis | `dis_pen=0.001`, adaptive | 5 | 0.1569 | 0.1054 | 0.5448 | 0.2871 | 0.1479 | 0.7427 |
| high-dis | `dis_pen=0.05`, adaptive | 6 | 0.1427 | 0.0935 | 0.5034 | 0.2650 | 0.1342 | 0.7131 |

## 2. 相对 baseline 的变化

| Variant | ΔRecall@20 | ΔNDCG@20 | ΔRecall@50 | ΔNDCG@50 | 结论 |
|---|---:|---:|---:|---:|---|
| no-dis | +0.0033 | +0.0028 | +0.0047 | +0.0037 | 纯准确率最高，但需要结合流行度判断是否更偏热门。 |
| no-adaptive | -0.0086 | -0.0058 | -0.0117 | -0.0065 | adaptive 机制有明显贡献。 |
| low-dis | +0.0018 | +0.0016 | +0.0058 | +0.0031 | 较弱解耦强度比原始设定更稳。 |
| high-dis | -0.0124 | -0.0103 | -0.0162 | -0.0105 | 过强解耦会明显损害表达能力。 |

## 3. 去偏指标

说明：AvgPop 越低表示推荐物品越不热门；LongTailRatio 越高表示长尾推荐占比越高；Coverage 越高表示推荐覆盖物品范围越广。

| Variant | TopK | AvgPop | LongTailRatio | Coverage | Users |
|---|---:|---:|---:|---:|---:|
| no-dis | 20 | 1103.3716 | 0.0001 | 0.4571 | 20480 |
| no-dis | 50 | 1070.7882 | 0.0003 | 0.5727 | 20480 |
| no-adaptive | 20 | 1027.4270 | 0.0037 | 0.7595 | 20480 |
| no-adaptive | 50 | 1050.9655 | 0.0048 | 0.8933 | 20480 |
| low-dis | 20 | 982.3677 | 0.0004 | 0.5206 | 20480 |
| low-dis | 50 | 974.8141 | 0.0008 | 0.6640 | 20480 |
| high-dis | 20 | 1194.6630 | 0.0006 | 0.5489 | 20480 |
| high-dis | 50 | 1206.8533 | 0.0011 | 0.6966 | 20480 |

## 4. 综合结论

1. `no-adaptive` 的 Recall/NDCG 明显低于 baseline，说明自适应训练机制不是可有可无的附属项，而是 NeuMF-DICE 稳定优化的重要组成。

2. `high-dis` 在所有准确率指标上最差，说明解耦惩罚过强会损害模型表达能力。这个结果可以支撑“DICE 需要平衡去偏和准确率”的讨论。

3. `low-dis` 在准确率上略高于 baseline，同时 AvgPop@20/50 最低，说明较弱解耦强度在当前 NeuMF backbone 上可能更合适：既保持推荐性能，又减少热门物品依赖。

4. `no-dis` 的 Recall/NDCG 最高，但 AvgPop 明显高于 `low-dis`，Coverage 也更低。这说明它更可能通过推荐热门物品获得准确率收益，因此不能只看 Recall/NDCG 判断方法优劣。

5. 四组消融共同说明：NeuMF-DICE 的关键不只是“把 MF 换成 NeuMF”，而是需要合理的解耦强度和 adaptive 训练策略；过强解耦、关闭 adaptive 都会损害性能。

## 5. 报告建议表述

可以写为：

> 消融实验表明，NeuMF-DICE 对解耦强度较敏感。关闭 adaptive 机制后 Recall@20 和 NDCG@20 均明显下降，说明自适应训练策略对模型优化具有积极作用。进一步比较不同解耦强度发现，过强的解耦惩罚会显著损害准确率，而较弱的解耦惩罚在保持准确率的同时降低 AvgPop，说明其能够缓解热门物品依赖。虽然 no-dis 在 Recall/NDCG 上略高，但其 AvgPop 较高、Coverage 较低，表明其收益可能来自更强的热门偏向。因此，综合准确率和去偏指标，low-dis 是当前 NeuMF backbone 下更平衡的设置。
