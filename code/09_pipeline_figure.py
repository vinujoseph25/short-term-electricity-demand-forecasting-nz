"""Pipeline diagram (Figure 3.1)."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from common import *
set_style()
fig, ax = plt.subplots(figsize=(6.6, 3.3)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 5)
def box(x, y, w, h, title, body, c):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc="white", ec=c, lw=1.4))
    ax.text(x + w / 2, y + h - .17, title, ha="center", va="top", fontsize=8, weight="bold", color=INK)
    ax.text(x + w / 2, y + h - .48, body, ha="center", va="top", fontsize=6.1, color=MUTED, linespacing=1.25)
def arrow(x1, y1, x2, y2): ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="->", color=MUTED, lw=1))
box(0.1, 3.1, 2.2, 1.7, "Raw data", "24 EMI Grid_export files\n(175,210 rows, 223 POCs)\n3 Open-Meteo city files\nNZ holiday calendar", COLORS[0])
box(2.7, 3.1, 2.2, 1.7, "01 Build series", "sum POCs, kWh/500 → MW\nTP → true UTC → 48-slot\nlocal grid (DST fix)\nweather TZ alignment", COLORS[0])
box(5.3, 3.1, 2.2, 1.7, "02–03 EDA + features", "re-run EDA on clean data\n25 features (lags, calendar,\nweather), origin = end of d−1\n3 target formulations", COLORS[0])
box(7.9, 3.1, 2.0, 1.7, "Chronological split", "train/CV: 8 Sep 2024–\n31 May 2026 (631 d)\ntest: Jun–Aug 2026\n(92 d, used once)", COLORS[1])
box(0.1, 0.4, 3.0, 2.1, "04–05 Tune (RO1)", "6 expanding rolling-origin folds\nPersistence D-1 / D-7\nRidge (α grid)\nLightGBM (40 random configs)\nLSTM, Transformer (12 each)", COLORS[2])
box(3.5, 0.4, 3.0, 2.1, "06–07 Ablations (RO2, RO3)", "target: abs / Δ-origin / Δ-slot\nloss: L1 / PW-MAE (mean, top-10%)\nλ = 2, 3, 5; weather none /\npersist / oracle; city proxy;\nfeature groups; train length", COLORS[2])
box(6.9, 0.4, 3.0, 2.1, "08 Final test (once)", "retrain on all training days\n5 seeds; score winter 2026\nDM tests, block bootstrap CI\nerror analysis, SHAP, costs\nsaved pipelines (joblib/.pt)", COLORS[1])
arrow(2.3, 3.95, 2.7, 3.95); arrow(4.9, 3.95, 5.3, 3.95); arrow(7.5, 3.95, 7.9, 3.95)
arrow(6.4, 3.1, 1.6, 2.5); arrow(6.4, 3.1, 5.0, 2.5); arrow(3.1, 1.45, 3.5, 1.45); arrow(6.5, 1.45, 6.9, 1.45); arrow(8.9, 3.1, 8.4, 2.5)
savefig(fig, "fig_pipeline")
