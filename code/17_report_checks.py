"""Step 17: supporting numbers quoted in the report that no earlier step writes out.

Reads only saved outputs (no model is retrained) and writes out/tables/report_checks.json:
  A. target standard deviations on the training days (absolute, delta-to-origin, delta-to-same-slot),
  B. tuning cost per configuration for LightGBM, LSTM and Transformer, and the ratios between them,
  C. daily peak-timing comparison of the selected Transformer with the absolute-target Transformer on the test winter.
"""
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

save_json(out, "report_checks.json")
print(json.dumps(out, indent=1))
