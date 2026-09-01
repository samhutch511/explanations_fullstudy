# Preregistered Analyses — Explanation as Rational Communication

Self-contained analysis bundle for the preregistered study comparing a rational-communication
model of explanation against a Halpern & Pearl baseline and three ablations.

Everything lives in [`preregistered_analyses.ipynb`](preregistered_analyses.ipynb). The notebook
loads participant logs, builds the world models for both grid maps, fits five models, and runs the
three confirmatory analyses specified in the preregistration (`explanations_prereg_0825.pdf`) and
the implementation spec (`analysis_implementation_spec.md`).

## ⚠️ Required data is not in this repository

**The notebook will not run from a fresh clone.** Raw participant data is excluded by
[`.gitignore`](.gitignore) because it is human-subjects data. You need to obtain these five files
and place them at these paths before running anything:

```
pilot_logs/log_orig_remote_20260707_103155.jsonl
pilot_logs/log_l1_remote_20260707_103155.jsonl
pilot_logs/log_l2_remote_20260707_103155.jsonl
pilot_logs/log_l3_remote_20260707_103155.jsonl
data/log.jsonl                                   # original CogSci dataset
```

Without them the notebook raises `FileNotFoundError` at the "LOAD ORIG PARTICIPANT DATA" cell.
The paths are set in the `CONFIG` cell (`LOG_FILE_ORIG`, `LOG_FILE_L1/L2/L3`) and, for the old
dataset, in the cross-dataset correlation cell (`_LOG_OLD`). Contact the project owner
(samhutch@mit.edu) for access.

Everything downstream of model fitting **is** committed — the `.npy` model arrays, the fitted
results in `results/`, the persisted fold indices, and the out-of-sample prediction tables. So the
statistics and figures can be re-derived from `results/` without the raw logs; only re-fitting
requires them.

## Models

| | Description | Free params |
|---|---|---|
| **Full** (M1) | Rational communication: belief inference + future resolution + social cost | 7 |
| **No-inference** (M3) | Ablation: no inference about the listener's beliefs | 4 |
| **No-future** (M4) | Ablation: no future-confusion resolution (γ = 0) | 6 |
| **No-social** (M5) | Ablation: no social cost of mentioning known conditions | 6 |
| **HP goodness** (M2) | Halpern & Pearl (2005b) explanatory-goodness baseline | 5 |

Fit to 60 scenarios across two maps: 15 "Orig" scenarios on a 6×9 grid, plus 45 hub scenarios
(L1/L2/L3, 15 each) on a 9×10 grid. Each L-set is designed to discriminate the Full model from one
specific ablation. The first 3 rounds per participant are practice and are excluded.

## Analyses

1. **Within-set generalization** — leave-one-scenario-out CV within each 15-scenario set
   (15 folds × 5 models × 4 sets = 300 fits), then 1000 bootstrap resamples over the pooled
   out-of-sample predictions.
2. **Across-type generalization** — leave-one-set-out CV (4 folds × 5 models = 20 fits), pooled to
   a 60-scenario out-of-sample table, then 1000 unstratified bootstrap resamples.
3. **Novel-scenario BSCV** — `N_SPLITS = 100` random 45/15 train/test partitions, refitting on every
   split (500 fits).

Point estimates are **medians** (not means); intervals are 2.5/97.5 percentiles. Model comparison
uses `win_rate` and paired differences — **no t-tests or p-values**, since bootstrap replicates are
not independent observations. The preregistered decision rule is `median R²(M1) > median R²(Mk)`
**and** `win_rate ≥ 0.95` (looser for Analysis 3, see spec §4.4).

`SEED = 20260825`, `N_BOOT = 1000`. All fold and resample indices are persisted to
`results/folds/` so results are exactly reproducible and a changed model does not change the splits.

## Setup

```bash
pip install "jax[cpu]" optax matplotlib numpy scipy memo-lang
```

Verified with Python 3.12.12, JAX 0.8.2, optax 0.2.6, NumPy 2.4.0, SciPy 1.16.3,
matplotlib 3.10.8, memo 1.2.7.

## Runtime — read before running

Run cells top to bottom. Total cost is dominated by Analysis 3.

| Cell group | Fits | Approx. compute |
|---|---|---|
| Setup + world models | — | a few minutes |
| Analysis 1 | 300 | ~2 h |
| Analysis 2 | 20 | ~10 min |
| **Analysis 3** | **500** | **~21 h** |
| §5 parameter fits | 25 | ~35 min |

The bottleneck is XLA compilation, not optimization. Every split has a different orig/hub scenario
mix, so `run_fitting_mixed` builds a fresh closure and JAX re-traces a very large unrolled graph on
nearly every fit — individual compiles of 2–3 minutes appear in the logs. Warm-starting from the
all-60 fit (spec §4.3) is the obvious optimization and has not been implemented.

Two practical warnings:

- **Nothing is checkpointed mid-loop.** Results are written only at the end of each analysis. If the
  kernel dies during Analysis 3, all completed fits are lost.
- **Progress bars measure wall-clock time.** If the machine sleeps, elapsed time and ETA are inflated
  and so is `timing.mean_seconds_per_fit` in `analysis3_results.json`. Do not use that field to
  choose `N_SPLITS` without checking it against actual CPU time. On macOS, run `caffeinate -i` to
  prevent idle sleep (it does **not** survive closing the lid).

## Outputs

```
results/
├── analysis1_results.json          # per set: median R², CIs, win rates, paired diffs
├── analysis2_results.json          # pooled stats, per-fold R² table, RMSE
├── analysis3_results.json          # median R², CIs, win rates, timing, param percentiles
├── parameters.csv                  # M1 parameters × {Orig, L1, L2, L3, All 60}
├── folds/                          # persisted bootstrap and split indices (+ hashes)
├── predictions/                    # pooled out-of-sample prediction tables (Analyses 1–2)
└── *.png                           # calibration scatter and median-R² bar charts
```

Prediction tables are kept separately so they can be re-scored with a different metric without
refitting. Figures are gitignored; re-run the figure cell to regenerate them.

## Known limitations

The committed `results/` files are from a pilot sample, not the preregistered target of 50
responses per scenario. Treat them as interim; they are not the confirmatory result.
