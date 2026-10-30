"""Proposed forecasting-service architecture (application conversion figure)."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from common import *
set_style()
fig, ax = plt.subplots(figsize=(6.6, 3.0)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 4.6)
def box(x, y, w, h, t, b, c):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc="white", ec=c, lw=1.3))
    ax.text(x + w / 2, y + h - .15, t, ha="center", va="top", fontsize=7.6, weight="bold", color=INK)
    ax.text(x + w / 2, y + h - .45, b, ha="center", va="top", fontsize=6.3, color=MUTED, linespacing=1.25)
def arr(a, b, c, d): ax.annotate("", (c, d), (a, b), arrowprops=dict(arrowstyle="->", color=MUTED, lw=1))
box(0.1, 2.6, 2.1, 1.8, "Inputs (daily, 23:30)", "metered GXP demand d−1\n(EMI / operator SCADA)\nweather forecast day d\n(3 cities, hourly)\nholiday calendar", COLORS[0])
box(2.6, 2.6, 2.3, 1.8, "Validation + features", "schema & range checks\nDST-aware 48-slot grid\n25 features, same code\nas training (common.py)", COLORS[0])
box(5.3, 2.6, 2.1, 1.8, "Inference", "saved pipeline:\nscalers + Transformer\n(5 seeds, 232 KB each)\n< 1 ms per forecast", COLORS[2])
box(7.8, 2.6, 2.1, 1.8, "Delivery", "REST API (JSON, 48 MW)\ndashboard for operators\nCSV to market systems\npeak alert if > threshold", COLORS[2])
box(1.2, 0.2, 3.4, 1.8, "Monitoring", "daily MAE, TD-MAE, peak error\nvs persistence & Ridge fallbacks\ninput drift: PSI on demand,\ntemperature; alert on breach", COLORS[1])
box(5.4, 0.2, 3.4, 1.8, "Retraining", "monthly expanding-window refit\n+ early retrain on drift alert\nchampion–challenger on last\n8 weeks before promotion", COLORS[1])
arr(2.2, 3.5, 2.6, 3.5); arr(4.9, 3.5, 5.3, 3.5); arr(7.4, 3.5, 7.8, 3.5); arr(8.8, 2.6, 4.0, 2.0); arr(4.6, 1.1, 5.4, 1.1); arr(7.1, 2.0, 6.35, 2.6)
savefig(fig, "fig_service")
