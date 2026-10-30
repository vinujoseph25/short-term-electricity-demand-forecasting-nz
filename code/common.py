"""Shared utilities: paths, plotting style, features, folds, metrics, statistics."""
import os, json, time
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "processed")
FIG = os.path.join(ROOT, "out", "figs"); TAB = os.path.join(ROOT, "out", "tables"); MOD = os.path.join(ROOT, "out", "models")
for p in (FIG, TAB, MOD): os.makedirs(p, exist_ok=True)
SEED = 42
FIRST_TARGET_DAY = "2024-09-08"          # first day with a full 7-day history
TEST_START, TEST_END = "2026-06-01", "2026-08-31"
# expanding-window rolling-origin folds: validation blocks of two months
FOLDS = [("2025-06-01", "2025-07-31"), ("2025-08-01", "2025-09-30"), ("2025-10-01", "2025-11-30"),
         ("2025-12-01", "2026-01-31"), ("2026-02-01", "2026-03-31"), ("2026-04-01", "2026-05-31")]
HDD_BASE = 16.0
PW_LAMBDA = 3.0                            # Bansal (2026): weight 1 + 2 = 3 on peak samples
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, MUTED = "#0b0b0b", "#52514e"

def set_style():
    import matplotlib as mpl
    mpl.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "DejaVu Serif"],
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "lines.linewidth": 1.5,
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight", "legend.frameon": False})

def savefig(fig, name):
    fig.savefig(os.path.join(FIG, name + ".png")); fig.savefig(os.path.join(FIG, name + ".pdf"))

def load_grid():
    return pd.read_pickle(os.path.join(PROC, "national_halfhourly.pkl"))

# ------------------------------------------------------------------ features (tabular)
WEATHER_COLS = {"popw": "temp_popw", "avg3": "temp_avg3", "akl": "temp_auc"}

def day_matrix(g, col):
    """Return DataFrame indexed by date with 48 columns for a given column."""
    return g.pivot(index="date", columns="slot", values=col)

def build_tabular(g, weather="observed", city="popw"):
    """One row per (target day D, slot s). Origin = end of day D-1 (all of D-1 observed).
    weather: 'observed' (realised weather at target time, an oracle for a day-ahead forecast),
             'persist' (same slot of day D-1, available at forecast time), 'none'."""
    Y = day_matrix(g, "mw"); T = day_matrix(g, WEATHER_COLS[city])
    W = (day_matrix(g, "wind_auc") + day_matrix(g, "wind_wel") + day_matrix(g, "wind_chr")) / 3
    hol = g.groupby("date").holiday.first()
    dates = Y.index.tolist(); di = {d: i for i, d in enumerate(dates)}
    Yv, Tv, Wv = Y.values, T.values, W.values
    recs = []
    for d in dates:
        i = di[d]
        if d < FIRST_TARGET_DAY: continue
        dt = pd.Timestamp(d)
        y1, y2, y7 = Yv[i - 1], Yv[i - 2], Yv[i - 7]
        past7 = Yv[i - 7:i]
        if weather == "observed": tt, tw = Tv[i], Wv[i]
        elif weather == "persist": tt, tw = Tv[i - 1], Wv[i - 1]
        else: tt, tw = None, None
        for s in range(48):
            r = dict(date=d, slot=s, y=Yv[i, s], origin=y1[47],
                     slot_sin=np.sin(2 * np.pi * s / 48), slot_cos=np.cos(2 * np.pi * s / 48),
                     dow=dt.dayofweek, weekend=int(dt.dayofweek >= 5), holiday=int(hol[d]),
                     hol_prev=int(hol[dates[i - 1]]), hol_next=int(hol[dates[i + 1]]) if i + 1 < len(dates) else 0,
                     doy_sin=np.sin(2 * np.pi * dt.dayofyear / 365.25), doy_cos=np.cos(2 * np.pi * dt.dayofyear / 365.25),
                     lag_d1=y1[s], lag_d2=y2[s], lag_d7=y7[s], d1_mean=y1.mean(), d1_max=y1.max(),
                     w7_mean_slot=past7[:, s].mean(), w7_max=past7.max(), d1_min=y1.min())
            if tt is not None:
                r.update(temp=tt[s], hdd=max(0.0, HDD_BASE - tt[s]), temp_dmean=tt.mean(), temp_dmin=tt.min(),
                         wind=tw[s], temp_lag_d1=Tv[i - 1][s])
            recs.append(r)
    return pd.DataFrame(recs)

FEATURE_GROUPS = {
    "calendar": ["slot", "slot_sin", "slot_cos", "dow", "weekend", "holiday", "hol_prev", "hol_next", "doy_sin", "doy_cos"],
    "lags": ["origin", "lag_d1", "lag_d2", "lag_d7", "d1_mean", "d1_max", "d1_min", "w7_mean_slot", "w7_max"],
    "weather": ["temp", "hdd", "temp_dmean", "temp_dmin", "wind", "temp_lag_d1"],
}

def feature_cols(df, drop=()):
    cols = []
    for gname, gc in FEATURE_GROUPS.items():
        if gname in drop: continue
        cols += [c for c in gc if c in df.columns]
    return cols

# ------------------------------------------------------------------ targets & weights
def make_target(df, kind):
    if kind == "abs": return df["y"].values, np.zeros(len(df))
    if kind == "delta_origin": return (df["y"] - df["origin"]).values, df["origin"].values
    if kind == "delta_seasonal": return (df["y"] - df["lag_d1"]).values, df["lag_d1"].values
    raise ValueError(kind)

def make_weights(y_abs, scheme, ref=None, lam=None):
    """Peak weights computed on ABSOLUTE demand. ref = training-set demand used for thresholds."""
    ref = y_abs if ref is None else ref; lam = PW_LAMBDA if lam is None else lam
    if scheme == "none": return np.ones(len(y_abs))
    if scheme == "bansal": return 1 + (lam - 1) * (y_abs > np.mean(ref))
    if scheme == "topdecile": return 1 + (lam - 1) * (y_abs >= np.quantile(ref, 0.9))
    raise ValueError(scheme)

def split_masks(df, fold=None, test=False):
    d = df["date"]
    if test:
        return (d < TEST_START).values, ((d >= TEST_START) & (d <= TEST_END)).values
    vs, ve = FOLDS[fold]
    return (d < vs).values, ((d >= vs) & (d <= ve)).values

# ------------------------------------------------------------------ metrics
def metrics(y, p, dates=None, slots=None):
    y = np.asarray(y, float); p = np.asarray(p, float); e = y - p; ae = np.abs(e)
    thr = np.quantile(y, 0.9); top = y >= thr
    w = np.where(top, PW_LAMBDA, 1.0)
    out = dict(MAE=ae.mean(), RMSE=np.sqrt((e ** 2).mean()), MAPE=100 * np.mean(ae / y),
               R2=1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum(),
               PW_MAE=np.mean(w * ae), TD_MAE=ae[top].mean(), TD_under=100 * np.mean(p[top] < y[top]),
               bias=np.mean(p - y))
    if dates is not None:
        dfp = pd.DataFrame(dict(d=dates, s=slots, y=y, p=p))
        a = dfp.pivot(index="d", columns="s", values="y").values; b = dfp.pivot(index="d", columns="s", values="p").values
        out["PeakMAE"] = np.mean(np.abs(a.max(1) - b.max(1)))
        out["PeakTimeErr_h"] = np.mean(np.abs(a.argmax(1) - b.argmax(1))) / 2
        out["PeakUnder"] = 100 * np.mean(b.max(1) < a.max(1))
    return out

def dm_test(e1, e2, h=1, power=1):
    """Diebold-Mariano test with Harvey-Leybourne-Newbold small-sample correction.
    e1, e2: forecast errors of model 1 and 2 on the same targets (here: daily mean abs errors).
    Returns (DM statistic, two-sided p-value). Positive DM => model 1 has larger loss."""
    from scipy import stats
    d = np.abs(e1) ** power - np.abs(e2) ** power; n = len(d); dbar = d.mean()
    gam = [np.sum((d[k:] - dbar) * (d[:n - k] - dbar)) / n for k in range(h)]
    var = (gam[0] + 2 * sum(gam[1:])) / n
    dm = dbar / np.sqrt(var)
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    stat = dm * hln
    return stat, 2 * stats.t.sf(np.abs(stat), df=n - 1)

def daily_mae(dates, y, p):
    return pd.DataFrame(dict(d=dates, ae=np.abs(np.asarray(y) - np.asarray(p)))).groupby("d").ae.mean()

def block_bootstrap_ci(dates, y, p, fn, n=1000, seed=SEED):
    """Day-block bootstrap CI of a metric: resample whole days with replacement."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(dict(d=dates, y=y, p=p)); groups = [g for _, g in df.groupby("d")]
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(groups), len(groups))
        b = pd.concat([groups[i] for i in idx]); vals.append(fn(b.y.values, b.p.values))
    return np.percentile(vals, [2.5, 97.5])

def save_json(obj, name):
    json.dump(obj, open(os.path.join(TAB, name), "w"), indent=1, default=float)

def pretty(name):
    """'Transformer delta_seasonal-none' -> 'Transformer (Δ-slot, L1)'"""
    rep = {"abs-L1": "(abs, L1)", "delta_seasonal-none": "(Δ-slot, L1)", "delta_seasonal-bansal": "(Δ-slot, PW mean)",
           "delta_seasonal-topdecile": "(Δ-slot, PW top-10%)", "delta_origin-none": "(Δ-origin, L1)", "delta_origin-bansal": "(Δ-origin, PW mean)",
           "delta_origin-topdecile": "(Δ-origin, PW top-10%)"}
    for k, v in rep.items(): name = name.replace(k, v)
    return name
