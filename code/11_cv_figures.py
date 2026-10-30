"""Step 11: CV summary tables and figures, per-fold comparison (RO1/RO3) and target x weighting ablation (RO2)."""
import matplotlib.pyplot as plt, pickle
from common import *
set_style(); out = {}
main = pd.read_pickle(os.path.join(TAB, "cv_tabular_main.pkl"))
abl = {"LightGBM": pd.read_pickle(os.path.join(TAB, "ablation_lgbm.pkl"))}
for a in ["LSTM", "Transformer"]:
    import glob
    ps = sorted(glob.glob(os.path.join(TAB, f"ablation_{a}*.pkl")))
    if ps: abl[a] = pd.concat([pd.read_pickle(p) for p in ps]).drop_duplicates(subset=["exp", "fold", "seed"])
# per-architecture target x weight summary (mean over folds of seed-mean)
summ = []
for arch, r in abl.items():
    r = r[r.exp.str.startswith("A|")]
    for exp, g in r.groupby("exp"):
        f = g.groupby("fold")[["MAE", "RMSE", "MAPE", "R2", "PW_MAE", "TD_MAE", "TD_under", "PeakMAE", "PeakTimeErr_h"]].mean()
        sd_seed = g.groupby("seed").MAE.mean().std() if g.seed.nunique() > 1 else np.nan
        _, t, w = exp.split("|")
        summ.append(dict(arch=arch, target=t, weights=w, **{c: f[c].mean() for c in f.columns}, MAE_fold_sd=f.MAE.std(),
                         PW_fold_sd=f.PW_MAE.std(), TD_fold_sd=f.TD_MAE.std(), MAE_seed_sd=sd_seed, fold_MAE=f.MAE.round(1).tolist(), fold_PW=f.PW_MAE.round(1).tolist(), fold_TD=f.TD_MAE.round(1).tolist()))
S = pd.DataFrame(summ)
for arch in S.arch.unique():
    b = S[(S.arch == arch) & (S.target == "abs") & (S.weights == "none")].iloc[0]
    for c in ["MAE", "PW_MAE", "TD_MAE"]: S.loc[S.arch == arch, c + "_chg%"] = 100 * (S.loc[S.arch == arch, c] / b[c] - 1)
    # how many folds improved PW-MAE vs baseline
    S.loc[S.arch == arch, "folds_PW_better"] = [int(np.sum(np.array(x) < np.array(b.fold_PW))) for x in S.loc[S.arch == arch, "fold_PW"]]
    S.loc[S.arch == arch, "folds_TD_better"] = [int(np.sum(np.array(x) < np.array(b.fold_TD))) for x in S.loc[S.arch == arch, "fold_TD"]]
S.to_pickle(os.path.join(TAB, "ablation_summary.pkl")); print(S.drop(columns=["fold_MAE", "fold_PW", "fold_TD"]).round(1).to_string())
# corrected resampled t-test (Nadeau & Bengio) across the 6 folds for each variant vs abs/none (PW-MAE)
from scipy import stats
ti = json.load(open(os.path.join(TAB, "tabular_info.json")))
n_tr = np.array([f["train_days"] for f in ti["folds"]]); n_te = np.array([f["val_days"] for f in ti["folds"]])
def crt(a, b):
    d = np.array(a) - np.array(b); k = len(d); var = d.var(ddof=1)
    t = d.mean() / np.sqrt((1 / k + np.mean(n_te / n_tr)) * var); return t, 2 * stats.t.sf(abs(t), k - 1)
tests = {}
for arch in S.arch.unique():
    b = S[(S.arch == arch) & (S.target == "abs") & (S.weights == "none")].iloc[0]
    for _, r in S[S.arch == arch].iterrows():
        if r.target == "abs" and r.weights == "none": continue
        tests[f"{arch}|{r.target}|{r.weights}"] = dict(PW=crt(r.fold_PW, b.fold_PW), TD=crt(r.fold_TD, b.fold_TD), MAE=crt(r.fold_MAE, b.fold_MAE))
out["crt_vs_abs"] = tests; save_json(out, "cv_stats.json")
# ---- figure: per-fold MAE of main models (best target of each arch shown as standard abs/none)
fig, ax = plt.subplots(figsize=(6.6, 2.6)); x = np.arange(1, 7)
series = [("Persistence D-1", main[main.model == "Persistence D-1"].per_fold_MAE.iloc[0]), ("Ridge", main[main.model == "Ridge"].per_fold_MAE.iloc[0])]
for arch in S.arch.unique(): series.append((arch, S[(S.arch == arch) & (S.target == "abs") & (S.weights == "none")].fold_MAE.iloc[0]))
for k, (n, v) in enumerate(series): ax.plot(x, v, marker="o", ms=4, color=COLORS[k], label=n)
ax.set_xticks(x, ["F1 Jun–Jul 25\n(winter)", "F2 Aug–Sep", "F3 Oct–Nov", "F4 Dec–Jan", "F5 Feb–Mar", "F6 Apr–May 26"]); ax.set_ylabel("Validation MAE (MW)")
ax.legend(ncol=5, fontsize=7, loc="upper right"); savefig(fig, "fig_cv_folds"); plt.close(fig)
# ---- figure: % change of PW-MAE and TD-MAE vs abs/none
fig, axs = plt.subplots(1, 2, figsize=(6.6, 3.6), sharey=True)
labels = [f"{t.replace('delta_origin','Δ-origin').replace('delta_seasonal','Δ-slot').replace('abs','abs')}\n{w.replace('none','L1').replace('bansal','PW mean').replace('topdecile','PW top10')}" for t in ["abs", "delta_origin", "delta_seasonal"] for w in ["none", "bansal", "topdecile"]][1:]
arches = [a for a in ["LightGBM", "Transformer"] if a in set(S.arch)]; wbar = .8 / len(arches); xx = np.arange(len(labels))
for j, c in enumerate(["PW_MAE_chg%", "TD_MAE_chg%"]):
    for k, arch in enumerate(arches):
        v = [S[(S.arch == arch) & (S.target == t) & (S.weights == w)][c].iloc[0] for t in ["abs", "delta_origin", "delta_seasonal"] for w in ["none", "bansal", "topdecile"]][1:]
        axs[j].barh(xx + k * wbar, v, height=wbar * .9, color=COLORS[k + 2] if False else COLORS[[0, 1, 2][k]], label=arch)
    axs[j].axvline(0, color=MUTED, lw=.8); axs[j].axvline(-10, color=MUTED, lw=.8, ls=":")
    axs[j].set_xlabel(("PW-MAE" if j == 0 else "Top-decile MAE") + " change vs abs/L1 (%)")
axs[0].set_yticks(xx + wbar * (len(arches) - 1) / 2, labels, fontsize=7); axs[0].invert_yaxis()
h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, ncol=3, loc="upper center", bbox_to_anchor=(0.55, 1.06), fontsize=8)
fig.tight_layout(); savefig(fig, "fig_ablation_target"); plt.close(fig)
