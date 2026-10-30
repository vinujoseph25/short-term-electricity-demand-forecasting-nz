"""Step 3: build and cache the tabular (one row per target day x slot) feature sets."""
from common import *
g = load_grid()
for w, c in [("observed", "popw"), ("persist", "popw"), ("none", "popw"), ("observed", "avg3"), ("observed", "akl")]:
    t0 = time.time(); df = build_tabular(g, weather=w, city=c)
    df.to_pickle(os.path.join(PROC, f"tab_{w}_{c}.pkl")); print(w, c, df.shape, round(time.time() - t0, 1), "s")
df = pd.read_pickle(os.path.join(PROC, "tab_observed_popw.pkl"))
info = dict(rows=len(df), days=df.date.nunique(), first=df.date.min(), last=df.date.max(),
            n_features=len(feature_cols(df)), features=feature_cols(df),
            train_days=int((df.date < TEST_START).sum() / 48), test_days=int((df.date >= TEST_START).sum() / 48),
            folds=[dict(val=f, train_days=int((df.date < f[0]).sum() / 48), val_days=int(((df.date >= f[0]) & (df.date <= f[1])).sum() / 48)) for f in FOLDS],
            nan=int(df[feature_cols(df)].isna().sum().sum()))
save_json(info, "tabular_info.json"); print(info)
# daily arrays for the neural networks (one sample per target day)
import pickle
from deep import build_daily
pickle.dump(build_daily(g), open(os.path.join(PROC, "daily_observed_popw.pkl"), "wb"))
