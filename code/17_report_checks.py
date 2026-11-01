"""Step 17: supporting numbers quoted in the report that no earlier step writes out.

Reads only saved outputs (no model is retrained) and writes out/tables/report_checks.json:
  A. target standard deviations on the training days (absolute, delta-to-origin, delta-to-same-slot),
  B. tuning cost per configuration for LightGBM, LSTM and Transformer, and the ratios between them,
  C. daily peak-timing comparison of the selected Transformer with the absolute-target Transformer on the test winter,
  D. Holm correction across the nine Diebold-Mariano tests reported in Table 4.5,
  E. the same DM tests with Newey-West (Bartlett) variances using 1 to 3 lags of daily autocovariance.
"""
from scipy import stats
import pickle
from common import *

sel = json.load(open(os.path.join(TAB, "selection.json")))
out = {}

# ---- A. target variance on training days (same targets as make_target)
F = pd.read_pickle(os.path.join(PROC, "tab_observed_popw.pkl"))
tr = F[(F.date >= FIRST_TARGET_DAY) & (F.date < TEST_START)]
sd = {k: float(np.std(make_target(tr, k)[0], ddof=1)) for k in ["abs", "delta_origin", "delta_seasonal"]}
out["target_sd_train"] = {k: round(v, 1) for k, v in sd.items()}
out["delta_seasonal_variance_share"] = round((sd["delta_seasonal"] / sd["abs"]) ** 2, 3)

# ---- B. tuning cost per configuration (LightGBM fit_s is per fold, so x6 folds)
tl = pd.read_pickle(os.path.join(TAB, "tune_lgbm.pkl"))
lg = float(tl.fit_s.mean() * len(FOLDS))
cost = {"LightGBM": lg}
for arch in ["LSTM", "Transformer"]:
    cost[arch] = float(pd.read_pickle(os.path.join(TAB, f"tune_{arch}.pkl")).sec.mean())
out["tuning_sec_per_config"] = {k: round(v, 1) for k, v in cost.items()}
out["tuning_cost_ratio_vs_lgbm"] = {k: round(cost[k] / lg, 1) for k in ["LSTM", "Transformer"]}

# ---- C. daily peak timing: selected Transformer vs absolute-target Transformer
P = pickle.load(open(os.path.join(TAB, "test_preds.pkl"), "rb"))
df = pd.DataFrame(dict(d=pd.to_datetime(P["date"]), s=np.asarray(P["slot"]), y=P["y"]))
A = df.pivot(index="d", columns="s", values="y").values
ta = A.argmax(1)
models = {"selected": sel["final_model"], "abs": "Transformer abs-L1"}
err = {}
pk = {}
for key, name in models.items():
    B = pd.DataFrame(dict(d=df.d, s=df.s, p=P["preds"][name])).pivot(index="d", columns="s", values="p").values
    tb = B.argmax(1)
    e = np.abs(ta - tb) / 2                      # hours (48 half-hour slots)
    swap = (ta < 24) != (tb < 24)               # peak placed in the wrong half of the day (morning vs evening)
    err[key] = e
    pk[key] = dict(model=name, mean_error_h=round(float(e.mean()), 2), exact_days=int((e == 0).sum()), days=int(len(e)),
                   am_pm_swap_days=int(swap.sum()), share_of_error_from_swaps=round(float(e[swap].sum() / e.sum()), 3))
s, p = dm_test(err["selected"], err["abs"])
pk["dm_selected_vs_abs"] = dict(DM=round(float(s), 2), p=round(float(p), 3))
out["peak_timing"] = pk

# ---- D. Holm correction over the Table 4.5 family (p-values from step 10)
TA = json.load(open(os.path.join(TAB, "test_analysis.json")))
def dm_p(table, a, b):
    for k in (f"{a} | {b}", f"{b} | {a}"):
        if k in TA[table]: return float(TA[table][k]["p"])
    raise KeyError((a, b))
S = sel["final_model"]
LG, LS, TA_ = "LightGBM delta_seasonal-none", "LSTM delta_seasonal-none", "Transformer abs-L1"
fam = [("dm", S, "Persistence D-1"), ("dm", S, "Persistence D-7"), ("dm", S, "Ridge"), ("dm", S, LG), ("dm", S, LS), ("dm", S, TA_),
       ("dm_pw", S, TA_), ("dm", LG, "LightGBM abs-L1"), ("dm", LS, "LSTM abs-L1")]
raw = [dm_p(t, a, b) for t, a, b in fam]
order = np.argsort(raw); m = len(raw); adj = [0.0] * m; run = 0.0
for rank, i in enumerate(order):
    run = max(run, min(1.0, raw[i] * (m - rank))); adj[i] = run
out["holm_table_4_5"] = [dict(test=t, a=a, b=b, p=round(r, 4), p_holm=round(q, 4), significant_holm=bool(q < 0.05))
                         for (t, a, b), r, q in zip(fam, raw, adj)]

# ---- E. DM with Newey-West variance (lags 1 to 3) on the daily MAE series
def daily(name):
    return pd.DataFrame(dict(d=df.d, e=np.abs(df.y.values - P["preds"][name]))).groupby("d").e.mean().values
def dm_nw(e1, e2, L):
    x = e1 - e2; n = len(x); c = x - x.mean()
    g = [np.sum(c[k:] * c[:n - k]) / n for k in range(L + 1)]
    v = g[0] + 2 * sum((1 - k / (L + 1)) * g[k] for k in range(1, L + 1))
    s = x.mean() / np.sqrt(v / n) * np.sqrt((n - 1) / n)          # same small-sample factor as dm_test with h = 1
    return float(2 * stats.t.sf(abs(s), n - 1))
nw = []
for t, a, b in fam:
    if t != "dm": continue
    ea, eb = daily(a), daily(b); c = (ea - eb) - (ea - eb).mean()
    ps = {f"lag{L}": round(dm_nw(ea, eb, L), 4) for L in (0, 1, 2, 3)}
    nw.append(dict(a=a, b=b, lag1_autocorr=round(float(np.corrcoef(c[1:], c[:-1])[0, 1]), 2), **ps,
                   same_decision_at_5pct=len({p < 0.05 for p in ps.values()}) == 1))
out["dm_newey_west"] = nw

save_json(out, "report_checks.json")
print(json.dumps(out, indent=1))
