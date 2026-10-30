"""Step 1: build the half-hourly national demand and weather dataset.

Reads the 24 monthly EMI Grid_export files and the three Open-Meteo city files,
and writes data/processed/national_halfhourly.parquet (local-clock 48-slot grid).
Every number about the raw data that the report quotes is printed here and saved
to out/tables/data_audit.json.
"""
import glob, json, os
import numpy as np, pandas as pd, holidays

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "data", "processed"); os.makedirs(OUT, exist_ok=True)
TAB = os.path.join(ROOT, "out", "tables")
TZ = "Pacific/Auckland"
audit = {}

# ---------------------------------------------------------------- demand
files = sorted(glob.glob(os.path.join(RAW, "20*_Grid_export.csv")))
raw = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
tps = [f"TP{i}" for i in range(1, 51)]
audit.update(files=len(files), raw_rows=int(len(raw)), raw_cols=int(raw.shape[1]),
             pocs=int(raw.POC.nunique()), days=int(raw.Trading_Date.nunique()),
             exact_duplicate_rows=int(raw.duplicated().sum()),
             duplicate_keys=int(raw.duplicated(["POC", "Nwk_Code", "Trader", "Trading_Date"]).sum()),
             negative_values=int((raw[tps] < 0).sum().sum()),
             unit=raw.Unit_Measure.unique().tolist(), flow=raw.Flow_Direction.unique().tolist(),
             status=raw.Status.unique().tolist())
pocs_per_month = raw.assign(m=raw.Trading_Date.str[:7]).groupby("m").POC.nunique()
audit["pocs_per_month_min"], audit["pocs_per_month_max"] = int(pocs_per_month.min()), int(pocs_per_month.max())

# national total per trading date and TP (kWh) -> MW (kWh / 500)
nat = raw.groupby("Trading_Date")[tps].sum(min_count=1)
tiwai = raw[raw.POC == "TWI2201"].groupby("Trading_Date")[tps].sum(min_count=1)
n_tp = nat.notna().sum(axis=1)
audit["days_by_tp_count"] = n_tp.value_counts().sort_index().to_dict()
audit["dst_days"] = {d: int(n) for d, n in n_tp[n_tp != 48].items()}

rows = []
for d, r in nat.iterrows():
    n = int(n_tp[d])
    midnight_utc = pd.Timestamp(d).tz_localize(TZ).tz_convert("UTC")
    for k in range(1, n + 1):          # TPs count real elapsed half-hours from local midnight
        rows.append((midnight_utc + pd.Timedelta(minutes=30 * (k - 1)), d, k,
                     r[f"TP{k}"] / 500.0, tiwai.loc[d, f"TP{k}"] / 500.0))
s = pd.DataFrame(rows, columns=["utc", "trading_date", "tp", "mw", "tiwai_mw"])
assert s.utc.is_unique and (s.utc.diff().dropna() == pd.Timedelta("30min")).all()
audit["halfhours_true_time"] = int(len(s))

# map true time onto a 48-slot local-clock grid
s["local"] = s.utc.dt.tz_convert(TZ)
s["date"] = s.local.dt.strftime("%Y-%m-%d")
s["slot"] = s.local.dt.hour * 2 + s.local.dt.minute // 30          # 0..47
g = s.groupby(["date", "slot"], as_index=False)[["mw", "tiwai_mw"]].mean()  # fall-back hour averaged
full = pd.MultiIndex.from_product([sorted(s.date.unique()), range(48)], names=["date", "slot"])
g = g.set_index(["date", "slot"]).reindex(full)
audit["slots_interpolated_spring_forward"] = int(g.mw.isna().sum())
g = g.interpolate(limit=2)                                          # 2 missing slots per spring-forward day
g = g.reset_index()
g["ts"] = pd.to_datetime(g.date) + pd.to_timedelta(g.slot * 30, unit="min")  # naive local clock
audit["grid_rows"] = int(len(g))

# ---------------------------------------------------------------- weather
# NOTE: the Auckland file is in fixed UTC+12, the Wellington and Christchurch files are in UTC.
offsets = {"Auckland": 12, "Wellington": 0, "Christchurch": 0}
wx = {}
for city, off in offsets.items():
    w = pd.read_csv(os.path.join(RAW, f"{city}_weather_2024-09_2026-08.csv"), skiprows=3)
    w.columns = ["time", "temp", "precip", "wind"]
    w["utc"] = (pd.to_datetime(w.time) - pd.Timedelta(hours=off)).dt.tz_localize("UTC")
    w = w.set_index("utc")[["temp", "precip", "wind"]]
    audit[f"wx_{city}_missing"] = int(w.isna().sum().sum())
    wx[city] = w
utc_index = pd.date_range(g.ts.min().tz_localize(TZ).tz_convert("UTC") - pd.Timedelta(hours=1),
                          s.utc.max() + pd.Timedelta(hours=1), freq="30min")
for city, w in wx.items():
    w = w.reindex(w.index.union(utc_index)).sort_index()
    w["temp"] = w["temp"].interpolate(method="time", limit_direction="both")
    w["wind"] = w["wind"].interpolate(method="time", limit_direction="both")
    w["precip"] = w["precip"].ffill().bfill() / 2.0          # hourly mm -> per half-hour
    wx[city] = w.reindex(utc_index)
# local-clock key for weather: average duplicates at fall-back, interpolate spring gap (as for demand)
def to_local_grid(df):
    df = df.copy(); loc = df.index.tz_convert(TZ)
    df["ts"] = loc.tz_localize(None).floor("30min")
    return df.groupby("ts").mean()
for city in offsets:
    lg = to_local_grid(wx[city]).reindex(g.ts).interpolate(limit_direction="both")
    for c in ["temp", "precip", "wind"]:
        g[f"{c}_{city[:3].lower()}"] = lg[c].values

# population weights: Stats NZ 2023 Census regional counts (Auckland, Wellington, Canterbury regions)
POP = {"auc": 1656486, "wel": 520971, "chr": 651027}
wsum = sum(POP.values())
g["temp_avg3"] = g[["temp_auc", "temp_wel", "temp_chr"]].mean(axis=1)
g["temp_popw"] = sum(g[f"temp_{k}"] * v / wsum for k, v in POP.items())

# ---------------------------------------------------------------- calendar
nz = holidays.country_holidays("NZ", years=[2024, 2025, 2026])
g["holiday"] = pd.to_datetime(g.date).dt.date.map(lambda d: d in nz).astype(int)
audit["holidays_in_window"] = sorted({str(d) for d in pd.to_datetime(g.date).dt.date if d in nz})
g.to_pickle(os.path.join(OUT, "national_halfhourly.pkl"))

# headline numbers (half-hourly, local grid)
audit["mw_mean"], audit["mw_sd"] = round(g.mw.mean(), 1), round(g.mw.std(), 1)
audit["mw_min"], audit["mw_max"] = round(g.mw.min(), 1), round(g.mw.max(), 1)
audit["tiwai_mean"] = round(g.tiwai_mw.mean(), 1)
json.dump(audit, open(os.path.join(TAB, "data_audit.json"), "w"), indent=1, default=str)
print(json.dumps(audit, indent=1, default=str))
