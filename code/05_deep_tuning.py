"""Step 5: random-search tuning of the LSTM and Transformer on the same rolling-origin folds."""
import sys, pickle
from deep import *
arch = sys.argv[1]; n_cfg = int(sys.argv[2]) if len(sys.argv) > 2 else 12
d = pickle.load(open(os.path.join(PROC, "daily_observed_popw.pkl"), "rb"))
rng = np.random.default_rng(SEED + (0 if arch == "LSTM" else 1))
def sample():
    base = dict(dropout=float(rng.choice([0, .1, .2, .3])), lr=float(np.exp(rng.uniform(np.log(3e-4), np.log(3e-3)))),
                head_hidden=int(rng.choice([32, 64])), batch=int(rng.choice([16, 32])))
    if arch == "LSTM": base.update(hidden=int(rng.choice([32, 64, 128])), layers=int(rng.choice([1, 2])))
    else: base.update(d_model=int(rng.choice([32, 64])), heads=int(rng.choice([2, 4])), layers=int(rng.choice([1, 2, 3])))
    return base
rows = []
for c in range(n_cfg):
    hp = sample(); res = []; t0 = time.time()
    for k in range(len(FOLDS)):
        tr, va = day_masks(d, k); p, info = fit_predict(d, tr, va, arch, hp)
        m = flat_metrics(d, va, p); m.update(fold=k, epochs=info["epochs"], best_epoch=info["best_epoch"], n_params=info["n_params"]); res.append(m)
    r = pd.DataFrame(res)
    rows.append(dict(hp=hp, MAE=r.MAE.mean(), MAE_sd=r.MAE.std(), PW_MAE=r.PW_MAE.mean(), per_fold=r.MAE.round(1).tolist(),
                     epochs=r.epochs.mean(), n_params=int(r.n_params.iloc[0]), sec=time.time() - t0))
    print(c, hp, round(r.MAE.mean(), 1), r.MAE.round(0).tolist(), round(time.time() - t0), flush=True)
    pd.DataFrame(rows).to_pickle(os.path.join(TAB, f"tune_{arch}.pkl"))
df = pd.DataFrame(rows); best = df.loc[df.MAE.idxmin(), "hp"]
json.dump(best, open(os.path.join(TAB, f"best_{arch}.json"), "w"), indent=1); print("BEST", best)
