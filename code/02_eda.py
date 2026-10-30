"""Step 2: re-run the EDA on the processed (local-clock, DST-corrected) data."""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from common import *
set_style(); g = load_grid(); res = {}
g["dt"] = pd.to_datetime(g.date); g["month"] = g.dt.dt.month
g["winter"] = g.month.isin([6, 7, 8]); g["split"] = np.where(g.date >= TEST_START, "test", "train")
res["overall"] = g.mw.describe().round(1).to_dict()
res["by_split"] = g.groupby("split").mw.describe().round(1).to_dict(orient="index")
res["winter_mean"], res["nonwinter_mean"] = g[g.winter].mw.mean(), g[~g.winter].mw.mean()
res["weekday_mean"], res["weekend_mean"] = g[g.dt.dt.dayofweek < 5].mw.mean(), g[g.dt.dt.dayofweek >= 5].mw.mean()
res["max_ts"] = str(g.loc[g.mw.idxmax(), "ts"]); res["min_ts"] = str(g.loc[g.mw.idxmin(), "ts"])
dp = g.groupby("date").apply(lambda x: x.slot.values[x.mw.values.argmax()], include_groups=False)
res["modal_peak_slot"] = int(dp.mode()[0]); res["peak_slot_share_evening"] = float(np.mean((dp >= 34) & (dp <= 39)))
res["share_peak_in_winter_top_decile"] = float(g[g.mw >= g.mw.quantile(.9)].winter.mean())
# year-on-year comparison (same months)
yoy = g.assign(yr=np.where(g.dt < "2025-09-01", "Sep24-Aug25", "Sep25-Aug26")).groupby("yr").mw.agg(["mean", "max"])
res["yoy"] = yoy.round(1).to_dict(orient="index")
# temperature correlations: corrected alignment vs naive file-row alignment used in A1
h = g.assign(hr=g.ts.dt.floor("h")).groupby("hr")[["mw", "temp_avg3", "temp_popw", "temp_auc"]].mean()
res["r_temp_avg3_corrected"] = h.mw.corr(h.temp_avg3); res["r_temp_popw"] = h.mw.corr(h.temp_popw); res["r_temp_akl"] = h.mw.corr(h.temp_auc)
raw = {}
for c in ["Auckland", "Wellington", "Christchurch"]:
    w = pd.read_csv(os.path.join(ROOT, "data", f"{c}_weather_2024-09_2026-08.csv"), skiprows=3); raw[c] = w.iloc[:, 1].values
naive = (raw["Auckland"] + raw["Wellington"] + raw["Christchurch"]) / 3
hh = h.mw.values[: len(naive)]
res["r_temp_naive_A1_alignment"] = float(np.corrcoef(hh, naive[: len(hh)])[0, 1])
# lag correlation of half-hourly demand
y = g.mw.values
res["acf"] = {k: float(np.corrcoef(y[k:], y[:-k])[0, 1]) for k in [1, 2, 48, 96, 336]}
# Tiwai
res["tiwai_mean"], res["tiwai_min"], res["tiwai_max"] = g.tiwai_mw.mean(), g.tiwai_mw.min(), g.tiwai_mw.max()
res["tiwai_share"] = g.tiwai_mw.sum() / g.mw.sum()
td = g.groupby("date").tiwai_mw.mean(); res["tiwai_days_below_450"] = int((td < 450).sum())
# anomaly screen: daily abs change vs same weekday last week
dm = g.groupby("date").mw.mean(); wk = (dm - dm.shift(7)).dropna(); z = (wk - wk.median()) / (1.4826 * (wk - wk.median()).abs().median())
res["anomalous_days_|z|>4"] = {d: round(v, 1) for d, v in z[abs(z) > 4].items()}
# demand vs temperature curve (binned) to choose HDD base
b = h.assign(tb=(h.temp_popw // 1)).groupby("tb").mw.mean(); res["binned_temp_curve"] = b.round(0).to_dict()
save_json(res, "eda.json"); print(json.dumps(res, indent=1, default=float))

# ---- figures
fig, ax = plt.subplots(figsize=(6.6, 2.4))
dd = g.groupby("dt").mw.mean(); ax.plot(dd.index, dd.values, color=COLORS[0], lw=1)
ax.axvspan(pd.Timestamp(TEST_START), pd.Timestamp(TEST_END), color="#e6e5e0", zorder=0)
for vs, ve in FOLDS: ax.axvline(pd.Timestamp(vs), color=MUTED, lw=.6, ls=":")
ax.text(pd.Timestamp("2026-06-05"), dd.max() * .995, "test (winter 2026)", fontsize=7, color=MUTED, va="top")
ax.set_ylabel("Daily mean demand (MW)"); savefig(fig, "fig_eda_daily"); plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.5))
prof = g.groupby(["winter", "slot"]).mw.mean().unstack(0)
x = np.arange(48) / 2
axs[0].plot(x, prof[False], color=COLORS[0], label="Sep–May"); axs[0].plot(x, prof[True], color=COLORS[1], label="Jun–Aug (winter)")
axs[0].set_xlabel("Local time (h)"); axs[0].set_ylabel("Mean demand (MW)"); axs[0].legend(); axs[0].set_xticks(range(0, 25, 6))
axs[1].scatter(h.temp_popw, h.mw, s=1, alpha=.15, color=COLORS[0], rasterized=True)
bb = h.assign(tb=(h.temp_popw // 1 + .5)).groupby("tb").mw.mean(); axs[1].plot(bb.index, bb.values, color=COLORS[1], lw=2, label="1 °C bin mean")
axs[1].axvline(HDD_BASE, color=MUTED, ls="--", lw=.8); axs[1].set_xlabel("Population-weighted temperature (°C)"); axs[1].set_ylabel("Hourly demand (MW)"); axs[1].legend()
fig.tight_layout(); savefig(fig, "fig_eda_profile_temp"); plt.close(fig)

fig, ax = plt.subplots(figsize=(4.2, 2.3))
lags = np.arange(1, 337); ac = [np.corrcoef(y[k:], y[:-k])[0, 1] for k in lags]
ax.plot(lags / 2, ac, color=COLORS[0]); ax.set_xlabel("Lag (hours)"); ax.set_ylabel("Autocorrelation"); ax.set_xticks([0, 24, 48, 72, 96, 120, 144, 168])
savefig(fig, "fig_eda_acf"); plt.close(fig)
