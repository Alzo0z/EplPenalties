# EPL Penalty Predictor

Machine-learning project on **6 seasons of English Premier League penalty kicks (628 cleaned rows)**, built end-to-end:

- Two prediction tasks — **Shooter** (will the penalty be scored?) and **Keeper** (which corner will the shooter aim at?).
- Three algorithms compared on each task — **Logistic Regression, SVM (linear + RBF), MLP** — all hyperparameter-tuned with 3-fold cross-validation.
- An interactive **Streamlit goal-mouth simulator** with shooter and keeper views, model picker, and live club-logo display.

## Quick start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Build the processed dataset + train all 6 models
python -m src.data
python -m src.train

# 3. Run the demo
streamlit run app/main.py
```

The first run also auto-downloads 28 club logos from Wikipedia into `app/assets/logos/`.

## Results

### Shooter task — binary (Scored vs Missed), test-set metrics

| Model | Accuracy | ROC-AUC | F1 |
|---|---|---|---|
| LogReg | 0.57 | 0.55 | 0.69 |
| **SVM** | 0.61 | **0.58** | 0.72 |
| MLP    | 0.69 | 0.56 | 0.80 |

ROC-AUC is the right metric — the baseline "always predict Scored" hits 82 % accuracy but offers no discrimination.

### Keeper task — 6-class angle, test-set metrics

| Model | Accuracy | Macro-F1 | Top-2 acc |
|---|---|---|---|
| LogReg | 0.22 | 0.20 | 0.38 |
| SVM    | 0.20 | 0.17 | 0.40 |
| **MLP** | 0.21 | 0.18 | **0.46** |

Random baseline = 17 % accuracy / 33 % top-2. All three models beat random; MLP wins on top-2 which is what powers the heatmap visualization in the demo.

See `report/figures/` for all plots and `report/report.md` for the full discussion.

## Project layout

```
epl-penalty-ml/
├── data/
│   ├── raw/Epl Penalties.xlsx
│   └── processed/{penalties.parquet, shooters.csv, gks.csv}
├── notebooks/
│   ├── 01_eda.ipynb           # exploratory analysis + figure rendering
│   ├── 02_models.ipynb        # training pipeline + result inspection
│   └── make_notebooks.py      # regenerates the notebooks programmatically
├── src/
│   ├── data.py                # load, clean, leak-free feature engineering
│   ├── features.py            # ColumnTransformer + task feature lists
│   ├── evaluate.py            # metric + plot helpers (CM, ROC, bars)
│   └── train.py               # trains all 6 models, saves artifacts
├── models/                    # *.joblib + *_results.csv
├── app/
│   ├── main.py                # Streamlit entry point
│   ├── components/
│   │   ├── goalmouth.py       # HTML/CSS goal renderer + corner buttons
│   │   └── logos.py           # Wikipedia logo fetcher with badge fallback
│   └── assets/logos/          # cached club crests (PNG)
└── report/
    ├── figures/               # all PNGs from notebooks + training
    ├── report.md              # written report
    └── slides.md              # slide deck (Marp / Reveal compatible)
```

## Methodology highlights

- **Random 80/20 stratified split** with `random_state=42` for reproducibility.
- **Leak-free shooter and GK rate features** computed via leave-one-out at row level — a player's training-time conversion rate excludes the current row to prevent target leakage.
- **Imbalance handling:**
  - Shooter (binary, 82/18): `class_weight='balanced'` for LogReg/SVM; SMOTE for MLP (which doesn't support `class_weight`).
  - Keeper (6-class, mildly imbalanced): SMOTE on training fold for all three models.
- **Hyperparameter tuning:** `GridSearchCV` (3 folds) over small, sensible grids per algorithm — `C` for LogReg/SVM, `kernel` for SVM, `alpha` for MLP.
- **Honest evaluation:** primary metric per task is the one that handles imbalance correctly (ROC-AUC for binary; macro-F1 + top-2 for 6-class).

## Demo screenshots

The Streamlit app has three tabs:
- **Shooter view** — click a corner, see P(scored) for your chosen algorithm, with a heatmap-styled goal-mouth.
- **Keeper view** — model predicts a shot heatmap, you dive to a corner, save/goal verdict with a running score.
- **Model comparison** — full results tables, ROC overlay, confusion matrices for every model.

Run the app and screenshot each tab — drop them in `report/figures/screenshots/` and reference them in the slide deck.
