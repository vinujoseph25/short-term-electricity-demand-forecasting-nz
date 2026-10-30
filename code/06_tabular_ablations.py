"""Step 6: LightGBM ablations on the rolling-origin folds (best hyperparameters held fixed)."""
import lightgbm as lgb
from common import *
best = json.load(open(os.path.join(TAB, "best_lgbm.json")))
frames = {k: pd.read_pickle(os.path.join(PROC, f"tab_{k}.pkl")) for k in ["observed_popw", "persist_popw", "none_popw", "observed_avg3", "observed_akl"]}
def run(frame, target="abs", weights="none", lam=None, drop=(), seeds=(42,), params=None, oof_name=None, train_days=None):
    params = params or best; feats = feature_cols(frame, drop); res, oofs = [], []
    for k in range(len(FOLDS)):
        tr, va = split_masks(frame, k)
        if train_days:   # learning curve: keep only the most recent N training days
            keep = sorted(frame.date[tr].unique())[-train_days:]; tr = tr & frame.date.isin(keep).values
        T, V = frame[tr], frame[va]; ytr, _ = make_target(T, target); _, bva = make_target(V, target)
        w = make_weights(T.y.values, weights, lam=lam); ps = []
        for sd in seeds:
            m = lgb.LGBMRegressor(objective="l1", bagging_freq=1, random_state=sd, n_jobs=1, verbose=-1, **params)
            m.fit(T[feats], ytr, sample_weight=w); p = m.predict(V[feats]) + bva; ps.append(p)
            mm = metrics(V.y, p, V.date.values, V.slot.values); mm.update(fold=k, seed=sd); res.append(mm)
        if oof_name: oofs.append(pd.DataFrame(dict(date=V.date.values, slot=V.slot.values, y=V.y.values, p=np.mean(ps, 0))))
    if oof_name: pd.concat(oofs).to_pickle(os.path.join(TAB, f"oof_{oof_name}.pkl"))
    return pd.DataFrame(res)
out = []
def rec(tag, **kw):
    t0 = time.time(); r = run(**kw); r["exp"] = tag; out.append(r)
    f = r.groupby("fold")[["MAE", "PW_MAE", "TD_MAE"]].mean()
    print(f"{tag:40s} MAE {f.MAE.mean():6.1f}±{f.MAE.std():5.1f} PW {f.PW_MAE.mean():6.1f} TD {f.TD_MAE.mean():6.1f}  {time.time()-t0:5.0f}s", flush=True)
    pd.concat(out).to_pickle(os.path.join(TAB, "ablation_lgbm.pkl"))
F = frames["observed_popw"]
# A: target x weighting (3 bagging seeds)
for t in ["abs", "delta_origin", "delta_seasonal"]:
    for w in ["none", "bansal", "topdecile"]:
        rec(f"A|{t}|{w}", frame=F, target=t, weights=w, seeds=(42, 43, 44), oof_name=f"LightGBM_{t}_{w}")
# B: lambda sensitivity
for lam in [2, 5]:
    for w in ["bansal", "topdecile"]: rec(f"B|lam{lam}|{w}", frame=F, weights=w, lam=lam)
# C: weather information and city proxy
rec("C|weather none", frame=frames["none_popw"]); rec("C|weather persist(D-1)", frame=frames["persist_popw"])
rec("C|city Auckland only", frame=frames["observed_akl"]); rec("C|city 3-city mean", frame=frames["observed_avg3"])
# D: feature groups
for grp in ["lags", "calendar", "weather"]: rec(f"D|drop {grp}", frame=F, drop=(grp,))
# E: learning curve (training-window length)
for n in [90, 180, 270]: rec(f"E|last {n} train days", frame=F, train_days=n)
