"""Step 8: retrain the selected configurations on the full training period
(1 Sep 2024 to 31 May 2026) and score the held-out winter 2026 test set ONCE.
The variants were fixed from the CV results before this script was run (see out/tables/selection.json)."""
import pickle, joblib, lightgbm as lgb
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge
from deep import *
sel = json.load(open(os.path.join(TAB, "selection.json")))
F = pd.read_pickle(os.path.join(PROC, "tab_observed_popw.pkl")); FEATS = feature_cols(F)
tr, te = split_masks(F, test=True); T, V = F[tr], F[te]
d = pickle.load(open(os.path.join(PROC, "daily_observed_popw.pkl"), "rb")); dtr, dte = day_masks(d, test=True)
assert (np.repeat(d["dates"][dte], 48) == V.date.values).all()
SEEDS = [42, 43, 44, 45, 46]
preds, rows, costs = {}, [], {}
def add(name, p, seed_metrics=None):
    preds[name] = p; m = metrics(V.y, p, V.date.values, V.slot.values); m["model"] = name
    if seed_metrics is not None:
        sm = pd.DataFrame(seed_metrics)
        for c in ["MAE", "RMSE", "MAPE", "PW_MAE", "TD_MAE", "PeakMAE"]: m[c + "_seed_sd"] = sm[c].std(); m[c + "_seed_mean"] = sm[c].mean()
    rows.append(m); print(name, {k: round(v, 2) for k, v in m.items() if k in ["MAE", "RMSE", "MAPE", "PW_MAE", "TD_MAE", "PeakMAE"]}, flush=True)
add("Persistence D-1", V.lag_d1.values); add("Persistence D-7", V.lag_d7.values)
cat = ["slot", "dow"]; num = [c for c in FEATS if c not in cat]
t0 = time.time()
ridge = Pipeline([("pre", ColumnTransformer([("num", StandardScaler(), num), ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])), ("m", Ridge(alpha=sel["ridge_alpha"]))]).fit(T[FEATS], T.y)
costs["Ridge"] = dict(train_s=time.time() - t0); add("Ridge", ridge.predict(V[FEATS]))
joblib.dump(dict(pipeline=ridge, features=FEATS), os.path.join(MOD, "ridge_pipeline.joblib"))
for tag, (t, w) in sel["lgbm_variants"].items():
    ps, sm = [], []; t0 = time.time()
    for sd in SEEDS:
        ytr, _ = make_target(T, t); _, bva = make_target(V, t)
        m = lgb.LGBMRegressor(objective="l1", bagging_freq=1, random_state=sd, n_jobs=1, verbose=-1, **sel["lgbm_params"])
        m.fit(T[FEATS], ytr, sample_weight=make_weights(T.y.values, w)); p = m.predict(V[FEATS]) + bva; ps.append(p)
        sm.append(metrics(V.y, p, V.date.values, V.slot.values))
        if sd == 42: joblib.dump(dict(model=m, features=FEATS, target=t, weights=w), os.path.join(MOD, f"lgbm_{tag}.joblib"))
    costs[f"LightGBM {tag}"] = dict(train_s=(time.time() - t0) / len(SEEDS)); add(f"LightGBM {tag}", np.mean(ps, 0), sm)
for arch in ["LSTM", "Transformer"]:
    hp = sel[f"{arch}_hp"]
    for tag, (t, w) in sel[f"{arch}_variants"].items():
        ps, sm, eps = [], [], []; t0 = time.time()
        for sd in SEEDS:
            p, info, model, sc = fit_predict(d, dtr, dte, arch, hp, target=t, weights=w, seed=sd, return_model=True)
            ps.append(p.ravel()); sm.append(flat_metrics(d, dte, p)); eps.append(info["best_epoch"])
            # every seed's weights are saved, so the five-seed ensemble can be reloaded and averaged at serving time
            torch.save(model.state_dict(), os.path.join(MOD, f"{arch}_{tag}_seed{sd}.pt"))
            if sd == 42:
                torch.save(model.state_dict(), os.path.join(MOD, f"{arch}_{tag}.pt")); pickle.dump(info["history"], open(os.path.join(TAB, f"history_{arch}_{tag}.pkl"), "wb"))
                # the scaler is fitted on the same training days for every seed, so one copy serves the whole ensemble
                pickle.dump(vars(sc), open(os.path.join(MOD, f"{arch}_{tag}_scaler.pkl"), "wb"))
                costs[f"{arch} {tag}"] = dict(n_params=info["n_params"])
        costs[f"{arch} {tag}"].update(train_s=(time.time() - t0) / len(SEEDS), best_epoch_mean=float(np.mean(eps)))
        add(f"{arch} {tag}", np.mean(ps, 0), sm)
res = pd.DataFrame(rows); res.to_pickle(os.path.join(TAB, "test_results.pkl"))
pickle.dump(dict(preds=preds, date=V.date.values, slot=V.slot.values, y=V.y.values), open(os.path.join(TAB, "test_preds.pkl"), "wb"))
save_json(costs, "costs_train.json")
print(res[["model", "MAE", "RMSE", "MAPE", "R2", "PW_MAE", "TD_MAE", "TD_under", "PeakMAE", "PeakTimeErr_h", "PeakUnder", "bias"]].round(2).to_string())
