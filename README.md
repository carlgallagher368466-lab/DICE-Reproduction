# 解耦学习与因果嵌入的推荐系统去偏复现与扩展

本仓库是课程项目的 GitHub 整理版，基于 WWW 2021 论文 *Disentangling User Interest and Conformity for Recommendation with Causal Embedding* 的 DICE 代码进行复现和扩展。项目包含原 DICE 思路复现、MovieLens-10M 三类测试集、NeuMF/NGCF/VAE backbone 扩展、四组消融实验，以及基于 CiaoDVD/Epinions trust 关系的 iDICE 社交影响扩展。

## 目录结构

```text
.
├── DICE/                         # 主要源码目录
│   ├── src/                      # 训练、模型、评估、配置
│   ├── tools/                    # 数据预处理、结果提取、指标统计脚本
│   ├── tests/                    # 本地单元测试
│   ├── data/                     # 已处理数据，含 train/val/test
│   └── requirements_reproduction.txt
├── datasets/                     # 原始数据压缩包或轻量来源文件
├── outputs/                      # 实验结果、汇总表、图表、报告
├── DATASETS.md                   # 数据来源与构建步骤
└── .gitignore
```

## 环境配置

建议使用 Linux + CUDA GPU 环境。基础复现和小规模测试可以在 CPU 上运行，但 NGCF/VAE/iDICE 完整训练建议使用 24GB 显存级别 GPU。

```bash
cd DICE
python -m venv .venv
source .venv/bin/activate
pip install -r requirements_reproduction.txt
```

如果使用 Conda：

```bash
conda create -n dice-repro python=3.8 -y
conda activate dice-repro
pip install -r DICE/requirements_reproduction.txt
```

## 数据准备

整理包已包含主要处理后数据：

- `DICE/data/ml10m/output/`
- `DICE/data/three_star/ml10m_longtail_test/output/`
- `DICE/data/three_star/ml10m_uniform_test/output/`
- `DICE/data/three_star/ml10m_head_test/output/`
- `DICE/data/epinions/output/`

数据来源和构建方法见 `DATASETS.md`。如需从原始数据重新生成：

```bash
cd DICE
python tools/make_three_star_splits.py --help
python tools/prepare_ciao_social.py --help
python tools/prepare_epinions_social.py --help
```

## 运行训练

所有模型统一通过配置文件运行：

```bash
cd DICE/src
python app.py --flagfile ./config/ml10m_longtail_neumfdice.cfg
```

常用配置示例：

```bash
# 三星/四星：MovieLens-10M 三类测试集 + NeuMF-DICE
python app.py --flagfile ./config/ml10m_longtail_neumfdice.cfg
python app.py --flagfile ./config/ml10m_uniform_neumfdice.cfg
python app.py --flagfile ./config/ml10m_head_neumfdice.cfg

# NGCF 扩展
python app.py --flagfile ./config/ngcf/ml10m_longtail_ngcfdice.cfg

# VAE 扩展
python app.py --flagfile ./config/vae/ml10m_longtail_vaedice.cfg

# iDICE / Epinions 社交影响扩展
python app.py --flagfile ./config/idice/epinions_idice.cfg
python app.py --flagfile ./config/idice/epinions_idice_no_social.cfg
python app.py --flagfile ./config/idice/epinions_idice_high_social.cfg
```

配置文件默认输出到 `DICE/src/runs/output/`。如需改到服务器数据盘，可修改配置文件中的 `--output`。

## 结果与报告

主要结果位于：

- `outputs/ngcf_full_50/`：NGCF 扩展结果
- `outputs/vae_full_50/`：VAE 扩展结果
- `outputs/ablation_4/`：四组消融实验
- `outputs/idice_ciao/`：CiaoDVD iDICE 汇总与可视化
- `outputs/idice_epinions/`：Epinions iDICE 汇总
- `outputs/five_star_final_report/`：最终五星报告、图表和报告源码

最终报告文件：

- `outputs/five_star_final_report/DICE_解耦学习推荐系统去偏_研究复现报告_五星完整版.pdf`
- `outputs/five_star_final_report/DICE_解耦学习推荐系统去偏_研究复现报告_五星完整版.docx`

## 本项目的主要改动

- 在原 DICE 框架中扩展 NeuMF、NGCF、VAE 三类 backbone。
- 保留 interest/conformity 解耦结构，并分别实现 Base、IPS、DICE 对照。
- 增加 iDICE，使用 trust 网络建模 social influence 分支。
- 构建 MovieLens-10M longtail/uniform/head 三类评估数据。
- 增加 CiaoDVD/Epinions 社交推荐数据预处理脚本。
- 增加 Coverage、AvgPop、长尾比例等去偏辅助指标与可视化结果。

## GitHub 上传建议

本整理包已排除 checkpoint、远程归档和缓存文件。由于处理后数据仍有数百 MB，首次推送可能较慢。如果 GitHub 拒绝大体积仓库，可以把 `DICE/data/` 移到 Release 附件或网盘，并在 `DATASETS.md` 中保留下载和构建说明。

常用上传命令：

```bash
git init
git add .
git commit -m "Initial course project reproduction package"
git branch -M main
git remote add origin https://github.com/<your-name>/<repo-name>.git
git push -u origin main
```
