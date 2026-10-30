"""Step 4: tune the baselines, Ridge and LightGBM with rolling-origin CV on the training period only."""
import lightgbm as lgb
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge
from common import *

df = pd.read_pickle(os.path.join(PROC, "tab_observed_popw.pkl"))
FEATS = feature_cols(df)
rows = []

def cv_eval(fit_predict, name, extra=None, store_oof=False, frame=df):
    res, oof = [], []
    for k in range(len(FOLDS)):
        tr, va = split_masks(frame, k)
        t0 = time.time(); p = fit_predict(frame[tr], frame[va]); dt = time.time() - t0
        m = metrics(frame.y[va], p, frame.date[va].values, frame.slot[va].values); m.update(fold=k, fit_s=dt); res.append(m)
        if store_oof: oof.append(pd.DataFrame(dict(date=frame.date[va].values, slot=frame.slot[va].values, y=frame.y[va].values, p=p)))
    r = pd.DataFrame(res); out = dict(model=name, **(extra or {}))
    for c in ["MAE", "RMSE", "MAPE", "R2", "PW_MAE", "TD_MAE", "PeakMAE"]:
        out[c] = r[c].mean(); out[c + "_sd"] = r[c].std()
    out["per_fold_MAE"] = r.MAE.round(1).tolist(); out["fit_s"] = r.fit_s.mean()
    return out, (pd.concat(oof) if store_oof else None)

# ---- naive baselines
for name, col in [("Persistence D-1", "lag_d1"), ("Persistence D-7", "lag_d7")]:
    o, oof = cv_eval(lambda tr, va, c=col: va[c].values, name, store_oof=True); rows.append(o)
    oof.to_pickle(os.path.join(TAB, f"oof_{name.replace(' ', '_')}.pkl"))

# ---- Ridge (linear baseline, one-hot slot and day-of-week, standardised numerics)
cat = ["slot", "dow"]; num = [c for c in FEATS if c not in cat]
def ridge_fp(alpha):
    def fp(tr, va):
        pipe = Pipeline([("pre", ColumnTransformer([("num", StandardScaler(), num), ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])),
                         ("m", Ridge(alpha=alpha))])
        pipe.fit(tr[FEATS], tr.y); return pipe.predict(va[FEATS])
    return fp
ridge_grid = []
for a in np.logspace(-3, 3, 13):
    o, _ = cv_eval(ridge_fp(a), "Ridge", dict(alpha=a)); ridge_grid.append(o)
rg = pd.DataFrame(ridge_grid); rg.to_csv(os.path.join(TAB, "tune_ridge.csv"), index=False)
best_a = rg.loc[rg.MAE.idxmin(), "alpha"]
o, oof = cv_eval(ridge_fp(best_a), "Ridge", dict(alpha=best_a), store_oof=True); rows.append(o); oof.to_pickle(os.path.join(TAB, "oof_Ridge.pkl"))
print("ridge best alpha", best_a, o["MAE"])

# ---- LightGBM random search
rng = np.random.default_rng(SEED)
def sample():
    return dict(num_leaves=int(rng.choice([15, 31, 63, 127])), learning_rate=float(np.exp(rng.uniform(np.log(.01), np.log(.1)))),
                n_estimators=int(rng.choice([300, 600, 1000, 1500])), min_child_samples=int(rng.choice([10, 20, 50, 100])),
                feature_fraction=float(rng.uniform(.6, 1)), bagging_fraction=float(rng.uniform(.6, 1)),
                lambda_l2=float(np.exp(rng.uniform(np.log(1e-3), np.log(10)))), max_depth=int(rng.choice([-1, 6, 10])))
def lgb_fp(params, target="abs", weights="none", seed=SEED, feats=None):
    def fp(tr, va):
        f = feats or FEATS
        ytr, base_tr = make_target(tr, target); _, base_va = make_target(va, target)
        w = make_weights(tr.y.values, weights)
        m = lgb.LGBMRegressor(objective="l1", bagging_freq=1, random_state=seed, n_jobs=1, verbose=-1, **params)
        m.fit(tr[f], ytr, sample_weight=w); return m.predict(va[f]) + base_va
    return fp
search = []
for it in range(40):
    p = sample(); t0 = time.time(); o, _ = cv_eval(lgb_fp(p), "LightGBM", dict(params=p)); search.append(o)
    print(it, round(o["MAE"], 1), round(time.time() - t0, 1), flush=True)
sd = pd.DataFrame(search); sd.to_pickle(os.path.join(TAB, "tune_lgbm.pkl"))
best = sd.loc[sd.MAE.idxmin(), "params"]; json.dump(best, open(os.path.join(TAB, "best_lgbm.json"), "w"), indent=1)
o, oof = cv_eval(lgb_fp(best), "LightGBM", dict(params=best), store_oof=True); rows.append(o); oof.to_pickle(os.path.join(TAB, "oof_LightGBM.pkl"))
pd.DataFrame(rows).to_pickle(os.path.join(TAB, "cv_tabular_main.pkl"))
print(pd.DataFrame(rows)[["model", "MAE", "MAE_sd", "RMSE", "MAPE", "PW_MAE", "TD_MAE", "PeakMAE", "per_fold_MAE"]])
