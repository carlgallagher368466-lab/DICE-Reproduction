# Three-Star Reproduction Guide

This guide targets the rubric's three-star requirement: reproducing experiments
where the train/test construction strategy is changed.

## Data Variants

The script `tools/make_three_star_splits.py` generates DICE-compatible variants:

- `ml10m_longtail_test`: test and skew-training samples favor low-popularity items.
- `ml10m_head_test`: test and skew-training samples favor high-popularity items.
- `ml10m_uniform_test`: test and skew-training samples use uniform sampling.

Each variant writes the same filenames expected by DICE:

- `train_coo_record.npz`, `train_skew_coo_record.npz`
- `val_coo_record.npz`, `test_coo_record.npz`
- `popularity.npy`, `popularity_blend.npy`
- `train_coo_adj_graph.npz`, `train_skew_coo_adj_graph.npz`,
  `train_blend_coo_adj_graph.npz`

Regenerate ML-10M variants from the repository root:

```bash
python tools/make_three_star_splits.py --dataset ml10m --source-root data --destination-root data/three_star
```

## Local Validation

Run:

```bash
python -m pytest tests/test_three_star_splits.py -q
```

Expected result:

```text
4 passed
```

## Training Commands

Recommended GPU environment:

```bash
conda create -n dice python=3.8 -y
conda activate dice
pip install -r requirements_reproduction.txt
```

Install the `torch` and `dgl` builds that match the remote machine's CUDA
version. The original code imports DGL at module load time, so DGL is required
even when the first smoke run uses MF rather than LightGCN.

Run from `src/`.

GPU:

```bash
visdom -port 33336
python app.py --flagfile ./config/ml10m_longtail_mf.cfg
python app.py --flagfile ./config/ml10m_longtail_ips.cfg
python app.py --flagfile ./config/ml10m_longtail_dice.cfg
python app.py --flagfile ./config/ml10m_uniform_mf.cfg
python app.py --flagfile ./config/ml10m_uniform_ips.cfg
python app.py --flagfile ./config/ml10m_uniform_dice.cfg
python app.py --flagfile ./config/ml10m_head_mf.cfg
python app.py --flagfile ./config/ml10m_head_ips.cfg
python app.py --flagfile ./config/ml10m_head_dice.cfg
```

CPU smoke test only:

```bash
python app.py --flagfile ./config/ml10m_longtail_mf.cfg --use_gpu=False --cg_use_gpu=False --epochs=1 --num_workers=0 --batch_size=256
```

Full CPU reproduction is not recommended because this environment has
`torch 2.9.1+cpu`, no CUDA device, and no compatible local DGL installation.

## Result Table Template

| Variant | Model | Recall@20 | NDCG@20 | Recall@50 | NDCG@50 | Notes |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Longtail test | MF | | | | | |
| Longtail test | IPS | | | | | |
| Longtail test | DICE | | | | | |
| Uniform test | MF | | | | | |
| Uniform test | IPS | | | | | |
| Uniform test | DICE | | | | | |
| Head test | MF | | | | | |
| Head test | IPS | | | | | |
| Head test | DICE | | | | | |

## Interpretation

Use `data/three_star/*/output/three_star_summary.json` to report that the split
strategy changed the distribution:

- Longtail test has lower mean item popularity in the test set.
- Head test has higher mean item popularity in the test set.
- Uniform test is the neutral comparison.

The expected analysis is not simply whether DICE wins every cell. Focus on
whether DICE is more stable under distribution shift, especially when the test
set is less dominated by popular items.
