"""Step 7: LSTM and Transformer target x weighting ablation on the rolling-origin folds (2 seeds)."""
import sys, pickle
from deep import *
arch = sys.argv[1]
d = pickle.load(open(os.path.join(PROC, "daily_observed_popw.pkl"), "rb"))
hp = json.load(open(os.path.join(TAB, f"best_{arch}.json")))
out, oofs = [], {}
combos = [(t, w) for t in ["abs", "delta_origin", "delta_seasonal"] for w in ["none", "bansal", "topdecile"]]
part = ""
if len(sys.argv) > 3:            # optional slice of combos to run in parallel processes
    a, b = int(sys.argv[2]), int(sys.argv[3]); combos = combos[a:b]; part = f"_{a}"
for t, w in combos:
    t0 = time.time(); oo = []
    for k in range(len(FOLDS)):
        tr, va = day_masks(d, k); ps = []
        for sd in (42, 43):
            p, info = fit_predict(d, tr, va, arch, hp, target=t, weights=w, seed=sd); ps.append(p)
            m = flat_metrics(d, va, p); m.update(fold=k, seed=sd, exp=f"A|{t}|{w}", epochs=info["epochs"]); out.append(m)
        pm = np.mean(ps, 0)
        oo.append(pd.DataFrame(dict(date=np.repeat(d["dates"][va], 48), slot=np.tile(np.arange(48), va.sum()), y=d["Y"][va].ravel(), p=pm.ravel())))
    pd.concat(oo).to_pickle(os.path.join(TAB, f"oof_{arch}_{t}_{w}.pkl"))
    r = pd.DataFrame(out); r.to_pickle(os.path.join(TAB, f"ablation_{arch}{part}.pkl"))
    f = r[r.exp == f"A|{t}|{w}"].groupby("fold")[["MAE", "PW_MAE", "TD_MAE"]].mean()
    print(f"{t:15s} {w:10s} MAE {f.MAE.mean():6.1f}±{f.MAE.std():5.1f} PW {f.PW_MAE.mean():6.1f} TD {f.TD_MAE.mean():6.1f} {time.time()-t0:5.0f}s", flush=True)
