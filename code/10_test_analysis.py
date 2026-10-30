"""Step 10: statistical tests, error analysis, interpretability and costs on the already scored test predictions."""
import pickle, joblib, glob
import matplotlib.pyplot as plt, lightgbm as lgb, shap
from common import *
set_style()
sel = json.load(open(os.path.join(TAB, "selection.json")))
TP = pickle.load(open(os.path.join(TAB, "test_preds.pkl"), "rb")); P = TP["preds"]; y = TP["y"]; dates = TP["date"]; slots = TP["slot"]
res = pd.read_pickle(os.path.join(TAB, "test_results.pkl")); out = {}
FINAL = sel["final_model"]
# ---- bootstrap CIs (day blocks) for MAE and TD-MAE
mae = lambda a, b: np.mean(np.abs(a - b))
def tdmae(a, b): t = a >= np.quantile(a, .9); return np.mean(np.abs(a[t] - b[t]))
ci = {}
for m in P:
    ci[m] = dict(MAE=block_bootstrap_ci(dates, y, P[m], mae).tolist(), TD_MAE=block_bootstrap_ci(dates, y, P[m], tdmae, n=500).tolist())
out["ci"] = ci
# ---- Diebold-Mariano on daily MAE series
dm = {}
for a in P:
    for b in P:
        if a >= b: continue
        ea, eb = daily_mae(dates, y, P[a]).values, daily_mae(dates, y, P[b]).values
        s, p = dm_test(ea, eb); dm[f"{a} | {b}"] = dict(stat=s, p=p, mean_diff=float(np.mean(ea - eb)))
# DM on daily top-decile-weighted error (peak-focused)
def daily_pw(m):
    thr = np.quantile(y, .9); w = np.where(y >= thr, PW_LAMBDA, 1.0)
    return pd.DataFrame(dict(d=dates, e=w * np.abs(y - P[m]))).groupby("d").e.mean().values
dmpw = {}
for a, b in sel["dm_pairs_pw"]:
    s, p = dm_test(daily_pw(a), daily_pw(b)); dmpw[f"{a} | {b}"] = dict(stat=s, p=p, mean_diff=float(np.mean(daily_pw(a) - daily_pw(b))))
out["dm"] = dm; out["dm_pw"] = dmpw
# ---- error analysis of final model
df = pd.DataFrame(dict(date=dates, slot=slots, y=y, p=P[FINAL])); df["e"] = df.y - df.p; df["ae"] = df.e.abs()
g = load_grid(); temp = g.groupby("date").temp_popw.mean(); hol = g.groupby("date").holiday.first()
dd = df.groupby("date").agg(MAE=("ae", "mean"), bias=("e", "mean"), ymax=("y", "max")).join(temp.rename("temp")).join(hol.rename("holiday"))
dd["dow"] = pd.to_datetime(dd.index).day_name()
out["worst_days"] = dd.sort_values("MAE", ascending=False).head(8).round(1).reset_index().to_dict(orient="records")
out["corr_dailyMAE_temp"] = float(dd.MAE.corr(dd.temp)); out["corr_dailyMAE_ymax"] = float(dd.MAE.corr(dd.ymax))
out["mae_by_dow"] = df.assign(dow=pd.to_datetime(df.date).dt.day_name()).groupby("dow").ae.mean().round(1).to_dict()
out["mae_holidays"] = float(df[df.date.map(hol) == 1].ae.mean()) if (df.date.map(hol) == 1).any() else None
cold = df[(df.date >= "2026-08-04") & (df.date <= "2026-08-07")]
out["coldsnap"] = {m: dict(MAE=float(np.mean(np.abs(cold.y - pd.Series(P[m], index=df.index)[cold.index]))),
                          max_pred=float(pd.Series(P[m], index=df.index)[cold.index].max())) for m in P}
out["coldsnap_actual_max"] = float(cold.y.max())
out["train_max"] = float(load_grid().query("date < @TEST_START").mw.max())
for m in P: out.setdefault("test_max_pred", {})[m] = float(np.max(P[m]))
# ---- figures
names = sel["plot_models"]
fig, ax = plt.subplots(figsize=(6.6, 2.6))
wk = df[(df.date >= "2026-08-02") & (df.date <= "2026-08-09")]; x = pd.to_datetime(wk.date) + pd.to_timedelta(wk.slot * 30, unit="min")
ax.plot(x, wk.y, color=INK, lw=1.6, label="Actual")
for k, m in enumerate(names): ax.plot(x, pd.Series(P[m], index=df.index)[wk.index], color=COLORS[k], lw=1.1, label=pretty(m))
ax.axhline(out["train_max"], color=MUTED, ls="--", lw=.8); ax.text(x.iloc[-60], out["train_max"] + 40, "training maximum (6,705 MW)", fontsize=7, color=MUTED)
ax.set_ylabel("Demand (MW)"); ax.set_ylim(3300, 7900); ax.legend(ncol=4, loc="upper left", fontsize=7)
import matplotlib.dates as mdates; ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %d %b")); savefig(fig, "fig_test_coldsnap"); plt.close(fig)
fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.6))
for k, m in enumerate(names):
    e = pd.DataFrame(dict(s=slots, ae=np.abs(y - P[m]))).groupby("s").ae.mean(); axs[0].plot(np.arange(48) / 2, e.values, color=COLORS[k], label=pretty(m))
axs[0].set_xlabel("Local time (h)"); axs[0].set_ylabel("Test MAE (MW)"); axs[0].set_xticks(range(0, 25, 6)); axs[0].legend(fontsize=7)
a = df.pivot(index="date", columns="slot", values="y").max(1)
for k, m in enumerate(names):
    b = pd.DataFrame(dict(d=dates, s=slots, p=P[m])).pivot(index="d", columns="s", values="p").max(1)
    axs[1].scatter(a, b, s=9, color=COLORS[k], label=pretty(m), alpha=.8)
lo, hi = a.min() - 100, a.max() + 100; axs[1].plot([lo, hi], [lo, hi], color=MUTED, lw=.8, ls="--")
axs[1].set_xlabel("Actual daily peak (MW)"); axs[1].set_ylabel("Forecast daily peak (MW)")
fig.tight_layout(); savefig(fig, "fig_test_slot_peak"); plt.close(fig)
# ---- SHAP for final LightGBM
F = pd.read_pickle(os.path.join(PROC, "tab_observed_popw.pkl")); V = F[F.date >= TEST_START]
art = joblib.load(os.path.join(MOD, f"lgbm_{sel['shap_variant']}.joblib")); feats = art["features"]
ex = shap.TreeExplainer(art["model"]); sv = ex.shap_values(V[feats])
imp = pd.Series(np.abs(sv).mean(0), index=feats).sort_values(); out["shap_mean_abs"] = imp.sort_values(ascending=False).round(1).to_dict()
fig, ax = plt.subplots(figsize=(4.6, 3.6)); top = imp.tail(15)
ax.barh(top.index, top.values, color=COLORS[0], height=.6); ax.set_xlabel("Mean |SHAP value| on test (MW of target)"); savefig(fig, "fig_shap"); plt.close(fig)
# permutation importance by group on test (held-out), final LightGBM
from common import FEATURE_GROUPS
base = mae(V.y.values, art["model"].predict(V[feats]) + make_target(V, art["target"])[1]); rng = np.random.default_rng(SEED); perm = {}
for gname, cols in FEATURE_GROUPS.items():
    cols = [c for c in cols if c in feats]; deltas = []
    for r in range(5):
        Vp = V.copy(); idx = rng.permutation(len(V))
        for c in cols: Vp[c] = V[c].values[idx]
        deltas.append(mae(V.y.values, art["model"].predict(Vp[feats]) + make_target(Vp, art["target"])[1]) - base)
    perm[gname] = [float(np.mean(deltas)), float(np.std(deltas))]
out["perm_group_importance"] = perm
# ---- costs: inference latency for one day-ahead forecast (48 values) and model size
lat = {}
X1 = V[feats].iloc[:48]
t0 = time.perf_counter(); [art["model"].predict(X1) for _ in range(200)]; lat["LightGBM"] = (time.perf_counter() - t0) / 200 * 1000
rd = joblib.load(os.path.join(MOD, "ridge_pipeline.joblib")); t0 = time.perf_counter(); [rd["pipeline"].predict(X1) for _ in range(200)]; lat["Ridge"] = (time.perf_counter() - t0) / 200 * 1000
from deep import ARCH, build_daily, Scaler, day_masks
import torch
d = pickle.load(open(os.path.join(PROC, "daily_observed_popw.pkl"), "rb")); tr, te = day_masks(d, test=True)
sc = Scaler().fit(d, tr); H, Fu = sc.transform(d, te); H1, F1 = torch.tensor(H[:1]), torch.tensor(Fu[:1])
for arch in ["LSTM", "Transformer"]:
    tag = sel[f"{arch}_main_tag"]; mdl = ARCH[arch](H.shape[-1], Fu.shape[-1], **sel[f"{arch}_hp"]); mdl.load_state_dict(torch.load(os.path.join(MOD, f"{arch}_{tag}.pt"))); mdl.eval()
    with torch.no_grad():
        t0 = time.perf_counter(); [mdl(H1, F1) for _ in range(200)]; lat[arch] = (time.perf_counter() - t0) / 200 * 1000
out["latency_ms"] = lat
out["model_size_kb"] = {os.path.basename(f): os.path.getsize(f) / 1024 for f in glob.glob(os.path.join(MOD, "*"))}
# ---- RO3: seasonal breakdown from CV out-of-fold predictions and test
season = {}
for f in sorted(glob.glob(os.path.join(TAB, "oof_*.pkl"))):
    o = pd.read_pickle(f); name = os.path.basename(f)[4:-4]; m = pd.to_datetime(o.date).dt.month
    w = m.isin([6, 7, 8])
    season[name] = dict(winter_MAE=float(np.mean(np.abs(o.y - o.p)[w])), nonwinter_MAE=float(np.mean(np.abs(o.y - o.p)[~w])),
                        winter_TD=metrics(o.y[w], o.p[w])["TD_MAE"], nonwinter_TD=metrics(o.y[~w], o.p[~w])["TD_MAE"])
out["season_oof"] = season
save_json(out, "test_analysis.json"); print(json.dumps({k: v for k, v in out.items() if k not in ("dm", "season_oof", "model_size_kb")}, indent=1, default=float)[:6000])
