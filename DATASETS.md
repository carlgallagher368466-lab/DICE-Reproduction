# 数据集说明

本项目使用三类数据：MovieLens-10M 派生的三类测试集、CiaoDVD 社交推荐数据、Epinions 社交推荐数据。

## 1. MovieLens-10M / Three-star splits

处理后数据已放在：

- `DICE/data/ml10m/output/`
- `DICE/data/three_star/ml10m_longtail_test/output/`
- `DICE/data/three_star/ml10m_uniform_test/output/`
- `DICE/data/three_star/ml10m_head_test/output/`

其中每个 `output` 目录包含 `train_record.csv`、`val_record.csv`、`test_record.csv`、稀疏矩阵 `*.npz`、流行度文件 `popularity*.npy` 和重编号映射文件。

三类测试集由 `DICE/tools/make_three_star_splits.py` 构建：

- `longtail`：提高长尾物品在测试集中的占比，用于观察去偏方法对冷门物品的推荐能力。
- `uniform`：按更均匀的物品分布构造测试集，降低热门物品主导性。
- `head`：偏向热门物品测试，用于与长尾场景对照。

## 2. CiaoDVD

原始压缩包保留在：

- `datasets/ciao/CiaoDVD.zip`

Ciao 预处理脚本为：

```bash
python DICE/tools/prepare_ciao_social.py \
  --rating-path path/to/ratings.txt \
  --trust-path path/to/trust.txt \
  --output-dir DICE/data/ciao/output
```

该脚本会读取评分与信任关系，重新编号用户和物品，按训练/验证/测试划分交互记录，并生成 `social_edges.csv`、`social_adj.npz` 等社交图文件。当前 GitHub 整理包保留了 Ciao 原始压缩包和实验结果汇总；若需要重新训练 Ciao，请先按上述命令生成 `DICE/data/ciao/output/`。

## 3. Epinions

轻量原始压缩包保留在：

- `datasets/epinions/epinions_with_rating_timestamp_txt.zip`

处理后数据已放在：

- `DICE/data/epinions/output/`

Epinions 预处理脚本为：

```bash
python DICE/tools/prepare_epinions_social.py \
  --rating-path path/to/rating_with_timestamp.txt \
  --trust-path path/to/trust.txt \
  --output-dir DICE/data/epinions/output
```

该脚本会把评分记录和 trust 网络转成 DICE/iDICE 可直接读取的交互矩阵、训练/验证/测试 CSV、用户/物品映射、流行度文件和社交邻接矩阵。

## 4. GitHub 大文件说明

为避免仓库过大，本整理包没有放入训练 checkpoint `*.pth`、远程服务器完整归档和超大的 `Epinions.rar`。复现实验不依赖这些权重文件，重新运行训练命令即可生成新的 checkpoint。
