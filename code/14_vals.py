"""Collect the numbers quoted in the report into report/vals_auto.json, keyed by name."""
import glob, pickle, sklearn, lightgbm, torch, statsmodels, shap, matplotlib
from common import *
V = {}
a = json.load(open(os.path.join(TAB, "data_audit.json"))); e = json.load(open(os.path.join(TAB, "eda.json"))); ti = json.load(open(os.path.join(TAB, "tabular_info.json")))
V.update(v_pandas=pd.__version__, v_numpy=np.__version__, v_sklearn=sklearn.__version__, v_lgbm=lightgbm.__version__, v_torch=torch.__version__.split("+")[0],
         v_statsmodels=statsmodels.__version__, v_shap=shap.__version__, v_mpl=matplotlib.__version__)
V.update(raw_rows=f"{a['raw_rows']:,}", raw_cols=a["raw_cols"], pocs=a["pocs"], pocs_min=a["pocs_per_month_min"], pocs_max=a["pocs_per_month_max"], days=a["days"],
         hh_true=f"{a['halfhours_true_time']:,}", interp=a["slots_interpolated_spring_forward"], grid_rows=f"{a['grid_rows']:,}",
         r_hourly=f"{e['r_temp_popw']:.2f}".replace("-", "−"), r_daily=f"{e['r_daily_mean_temp_popw']:.2f}".replace("-", "−"), r_naive=f"{e['r_temp_naive_A1_alignment']:.2f}".replace("-", "−"),
         tiwai_mean=f"{e['tiwai_mean']:.0f}", tiwai_share=f"{100*e['tiwai_share']:.1f}", tiwai_low_days=e["tiwai_days_below_450"],
         n_days=ti["days"], n_rows=f"{ti['rows']:,}", n_feat=ti["n_features"], acf1=f"{e['acf']['1']:.2f}", acf48=f"{e['acf']['48']:.2f}", acf96=f"{e['acf']['96']:.2f}", acf336=f"{e['acf']['336']:.2f}",
         wkday=f"{e['weekday_mean']:,.0f}", wkend=f"{e['weekend_mean']:,.0f}", peak_slot=e["modal_peak_slot"], evening_share=f"{100*e['peak_slot_share_evening']:.0f}",
         peak_winter_share=f"{100*e['share_peak_in_winter_top_decile']:.0f}", train_days=ti["train_days"], test_days=ti["test_days"], test_rows=f"{ti['test_days']*48:,}",
         winter_mean=f"{e['winter_mean']:,.0f}", nonwinter_mean=f"{e['nonwinter_mean']:,.0f}", train_mean=f"{e['by_split']['train']['mean']:,.0f}", test_mean=f"{e['by_split']['test']['mean']:,.0f}",
         yoy1=f"{e['yoy']['Sep24-Aug25']['mean']:,.0f}", yoy2=f"{e['yoy']['Sep25-Aug26']['mean']:,.0f}", yoymax1=f"{e['yoy']['Sep24-Aug25']['max']:,.0f}", yoymax2=f"{e['yoy']['Sep25-Aug26']['max']:,.0f}")
# tuning times (sum of per-config seconds; machine shared by two jobs)
tl = pd.read_pickle(os.path.join(TAB, "tune_lgbm.pkl")); ls = pd.read_pickle(os.path.join(TAB, "tune_LSTM.pkl")); tf = pd.read_pickle(os.path.join(TAB, "tune_Transformer.pkl"))
lg_sec = tl.fit_s.sum() * 6
hm = lambda s: f"{s/3600:.1f} h" if s > 3600 else f"{s/60:.0f} min"
vt = dict(t_ridge="<1 min", t_lgbm=hm(lg_sec), t_lstm=hm(ls.sec.sum()), t_tf=hm(tf.sec.sum()))
save_json(vt, "vals_tuning.json"); V["dl_cost_ratio"] = f"{(ls.sec.mean()+tf.sec.mean())/2/(lg_sec/len(tl)):.0f}"
V["total_runtime"] = "about 3 hours on an Apple-silicon MacBook Pro, CPU only (about 8 hours on a 2-vCPU Linux machine); dominated by LSTM tuning and ablations"
os.makedirs(os.path.join(ROOT, "report"), exist_ok=True)
json.dump(V, open(os.path.join(ROOT, "report", "vals_auto.json"), "w"), indent=1, default=str); print(V); print(vt)
