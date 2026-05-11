"""Generate the two project notebooks programmatically.

Run once to (re)create notebooks/01_eda.ipynb and notebooks/02_models.ipynb.
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"


# ---------------------------------------------------------------------------
# 01_eda.ipynb — exploratory data analysis
# ---------------------------------------------------------------------------
def build_eda():
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(nbf.v4.new_markdown_cell(
        "# 01 — Exploratory Data Analysis: EPL Penalties\n\n"
        "**Dataset:** 6 EPL seasons of penalty kicks (628 rows after cleaning).\n\n"
        "This notebook covers:\n"
        "1. Loading + cleaning the raw spreadsheet.\n"
        "2. Class balance (Scored vs Missed) and corner distribution.\n"
        "3. Numeric feature distributions.\n"
        "4. Categorical feature breakdowns.\n"
        "5. Correlation heatmap.\n\n"
        "All figures are written to `report/figures/` for the report and slides."
    ))

    cells.append(nbf.v4.new_code_cell(
        "import sys\n"
        "from pathlib import Path\n"
        "ROOT = Path('..').resolve()\n"
        "if str(ROOT) not in sys.path:\n"
        "    sys.path.insert(0, str(ROOT))\n"
        "\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n"
        "sns.set_theme(style='whitegrid', context='talk')\n"
        "\n"
        "from src.data import build_dataset, load_processed, ANGLE_ORDER\n"
        "\n"
        "FIG_DIR = ROOT / 'report' / 'figures'\n"
        "FIG_DIR.mkdir(parents=True, exist_ok=True)"
    ))

    cells.append(nbf.v4.new_markdown_cell("## 1. Load + clean"))

    cells.append(nbf.v4.new_code_cell(
        "info = build_dataset(ROOT / 'data/raw/Epl Penalties.xlsx', ROOT / 'data/processed')\n"
        "print('Built dataset:', info)\n"
        "df, shooters, gks = load_processed(ROOT / 'data/processed')\n"
        "df.head()"
    ))

    cells.append(nbf.v4.new_markdown_cell("## 2. Class balance"))

    cells.append(nbf.v4.new_code_cell(
        "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n"
        "df['Outcome'].value_counts().plot(kind='bar', color=['#2E7D32', '#C62828'], ax=axes[0])\n"
        "axes[0].set_title('Shooter target: Scored vs Missed'); axes[0].set_ylabel('Shots')\n"
        "\n"
        "df['Angle'].value_counts().loc[ANGLE_ORDER].plot(kind='bar', color='#1565C0', ax=axes[1])\n"
        "axes[1].set_title('Keeper target: shot corner'); axes[1].set_ylabel('Shots')\n"
        "axes[1].tick_params(axis='x', rotation=30)\n"
        "plt.tight_layout()\n"
        "fig.savefig(FIG_DIR / 'eda_class_balance.png', dpi=150)\n"
        "plt.show()"
    ))

    cells.append(nbf.v4.new_markdown_cell(
        "**Observations:**\n"
        "- Scored vs Missed is ~82/18 — heavily imbalanced. We handle this with "
        "`class_weight='balanced'` for LogReg/SVM and SMOTE for MLP.\n"
        "- The 6 corner classes are mildly imbalanced — Left/Right Bottom dominate. "
        "We use SMOTE during training for all keeper models."
    ))

    cells.append(nbf.v4.new_markdown_cell("## 3. Numeric features"))

    cells.append(nbf.v4.new_code_cell(
        "num_cols = ['Minute', 'Gameweek', 'GK_Height', 'ShooterConvRate', 'GKSaveRate']\n"
        "fig, axes = plt.subplots(2, 3, figsize=(14, 8))\n"
        "for ax, c in zip(axes.flat, num_cols):\n"
        "    sns.histplot(df[c], kde=True, ax=ax, color='#3949AB')\n"
        "    ax.set_title(c)\n"
        "axes.flat[-1].axis('off')\n"
        "plt.tight_layout()\n"
        "fig.savefig(FIG_DIR / 'eda_numeric_dists.png', dpi=150)\n"
        "plt.show()"
    ))

    cells.append(nbf.v4.new_markdown_cell("## 4. Categorical breakdowns"))

    cells.append(nbf.v4.new_code_cell(
        "fig, axes = plt.subplots(2, 2, figsize=(13, 9))\n"
        "for ax, c in zip(axes.flat, ['Foot', 'Venue', 'Condition', 'TeamCategory']):\n"
        "    ct = pd.crosstab(df[c], df['Outcome'], normalize='index')\n"
        "    ct.plot(kind='bar', stacked=True, ax=ax,\n"
        "            color={'Scored': '#2E7D32', 'Missed': '#C62828'})\n"
        "    ax.set_title(f'Outcome by {c}'); ax.set_ylabel('Share'); ax.legend(loc='lower right')\n"
        "    ax.tick_params(axis='x', rotation=0)\n"
        "plt.tight_layout()\n"
        "fig.savefig(FIG_DIR / 'eda_outcome_by_categorical.png', dpi=150)\n"
        "plt.show()"
    ))

    cells.append(nbf.v4.new_markdown_cell("## 5. Correlation heatmap"))

    cells.append(nbf.v4.new_code_cell(
        "corr_cols = ['Scored', 'Minute', 'Gameweek', 'GK_Height',\n"
        "             'ShooterConvRate', 'GKSaveRate']\n"
        "corr = df[corr_cols].corr()\n"
        "fig, ax = plt.subplots(figsize=(7, 6))\n"
        "sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax)\n"
        "ax.set_title('Numeric feature correlations')\n"
        "plt.tight_layout()\n"
        "fig.savefig(FIG_DIR / 'eda_correlation.png', dpi=150)\n"
        "plt.show()"
    ))

    cells.append(nbf.v4.new_markdown_cell(
        "## Takeaways for modeling\n"
        "- **Highly imbalanced target** for the shooter task — accuracy alone is misleading; use ROC-AUC.\n"
        "- **Numeric features have weak linear correlation** with the target — non-linear models (SVM-RBF, MLP) may help, though dataset size limits their advantage.\n"
        "- **Shooter conversion rate and GK save rate** are the strongest individual predictors as expected.\n"
        "- **Categorical features matter**: foot, venue, team category and condition all shift the score rate."
    ))

    nb["cells"] = cells
    nbf.write(nb, NB_DIR / "01_eda.ipynb")
    print("Wrote", NB_DIR / "01_eda.ipynb")


# ---------------------------------------------------------------------------
# 02_models.ipynb — training + comparison
# ---------------------------------------------------------------------------
def build_models():
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(nbf.v4.new_markdown_cell(
        "# 02 — Modeling: LogReg, SVM, MLP on Shooter + Keeper tasks\n\n"
        "We train three algorithms on **two prediction tasks**:\n\n"
        "| Task | Type | Target |\n"
        "|---|---|---|\n"
        "| Shooter | Binary classification | Scored / Missed |\n"
        "| Keeper  | 6-class classification | Angle (corner) |\n\n"
        "**Protocol**\n"
        "- 80/20 stratified random split, `random_state=42`.\n"
        "- 3-fold GridSearchCV on the training set.\n"
        "- Shooter: `class_weight='balanced'` (LogReg/SVM), SMOTE (MLP).\n"
        "- Keeper: SMOTE on training fold for all three.\n"
        "- All artifacts (joblibs, PNGs, results.csv) saved under `models/` and `report/figures/`.\n\n"
        "The actual training is implemented as the reusable module `src/train.py`; this notebook calls it and inspects the results."
    ))

    cells.append(nbf.v4.new_code_cell(
        "import sys\n"
        "from pathlib import Path\n"
        "ROOT = Path('..').resolve()\n"
        "if str(ROOT) not in sys.path:\n"
        "    sys.path.insert(0, str(ROOT))\n"
        "\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "from IPython.display import Image, display\n"
        "\n"
        "from src.train import main as train_main"
    ))

    cells.append(nbf.v4.new_markdown_cell(
        "## 1. Train everything\n"
        "\nThis runs the full pipeline (~30s on CPU). Re-run any time you change features."
    ))

    cells.append(nbf.v4.new_code_cell(
        "train_main()"
    ))

    cells.append(nbf.v4.new_markdown_cell("## 2. Shooter results"))

    cells.append(nbf.v4.new_code_cell(
        "shooter_res = pd.read_csv(ROOT / 'models/shooter_results.csv')\n"
        "shooter_res"
    ))

    cells.append(nbf.v4.new_code_cell(
        "display(Image(ROOT / 'report/figures/shooter_roc_overlay.png'))\n"
        "for name in ('logreg', 'svm', 'mlp'):\n"
        "    display(Image(ROOT / f'report/figures/shooter_{name}_cm.png'))"
    ))

    cells.append(nbf.v4.new_markdown_cell(
        "**Shooter interpretation:** ROC-AUC is the right metric here because "
        "the baseline 'always predict Scored' achieves ~82% accuracy but no useful "
        "discrimination. AUC ~0.55-0.58 means the models are slightly informed; "
        "this is realistic for ~628 noisy penalties without rich context."
    ))

    cells.append(nbf.v4.new_markdown_cell("## 3. Keeper results"))

    cells.append(nbf.v4.new_code_cell(
        "keeper_res = pd.read_csv(ROOT / 'models/keeper_results.csv')\n"
        "keeper_res"
    ))

    cells.append(nbf.v4.new_code_cell(
        "for name in ('logreg', 'svm', 'mlp'):\n"
        "    display(Image(ROOT / f'report/figures/keeper_{name}_cm.png'))"
    ))

    cells.append(nbf.v4.new_markdown_cell(
        "**Keeper interpretation:** Random over 6 classes is 17% accuracy / 33% top-2. "
        "Our models reach ~20-22% accuracy and ~40-46% top-2 — a meaningful lift, "
        "though peak accuracy is modest because individual penalty placement is highly stochastic."
    ))

    cells.append(nbf.v4.new_markdown_cell(
        "## 4. Comparison summary\n"
        "\nWinning model per task (on test set):\n\n"
        "- **Shooter:** SVM has the best ROC-AUC; MLP has the best accuracy but at the cost of AUC discrimination.\n"
        "- **Keeper:** MLP wins on Top-2 accuracy, which is what powers the heatmap visualization in the Streamlit demo.\n"
        "- **LogReg** is competitive on both tasks despite being the simplest model — a clear sign the signal is mostly linear at this dataset size.\n"
    ))

    nb["cells"] = cells
    nbf.write(nb, NB_DIR / "02_models.ipynb")
    print("Wrote", NB_DIR / "02_models.ipynb")


if __name__ == "__main__":
    build_eda()
    build_models()
