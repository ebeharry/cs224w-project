# CS224W Project: Bot Detection on MGTAB

Graph-based and baseline bot detection on the MGTAB Twitter dataset.

## Setup

```bash
conda env create -f environment.yml
conda activate cs224w
```

## Layout

- `src/` — data loading, models, and run scripts (`run_baseline.py`, `run_mlp.py`, `run_gnn.py`)
- `data/mgtab/` — MGTAB dataset files
- `results/` — result tables and plots
- `tests/` — tests

## Acknowledgements

This project builds on [MGTAB](https://github.com/GraphDetec/MGTAB). `src/dataset.py`, `src/utils.py`, and `src/MGTAB-ML.py` are taken directly from that repository.
