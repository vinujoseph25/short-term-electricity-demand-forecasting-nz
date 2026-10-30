"""Step 16: post-hoc sensitivity analyses, run after model selection (they change no selection decision).

A. Weather sensitivity of the selected model. The selected Transformer (delta-to-same-slot target, L1 loss,
   tuned hyperparameters from selection.json) is retrained on the training period with five seeds, once with
   observed target-day weather (as in the main results) and once with persistence weather (the same slot of
   day D-1, which is known at the forecast origin). Both are scored on the winter 2026 test set.
B. Unsupervised day-type analysis. K-means clusters the standardised 48-slot demand profiles of the training
   days (scaler and clusters fitted on training days only, k chosen by silhouette score). Test days are assigned
   to the nearest centroid, and the forecast error of both weather variants is broken down by cluster and by
   distance to the nearest centroid (how unusual a day is compared with the training data).
"""
import pickle
from deep import *
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

sel = json.load(open(os.path.join(TAB, "selection.json")))
hp = sel["Transformer_hp"]
TARGET, WEIGHTS = "delta_seasonal", "none"
SEEDS = [42, 43, 44, 45, 46]
g = load_grid()
out = {}

# ---- A. weather sensitivity of the selected Transformer
preds = {}
for weather in ["observed", "persist"]:
    d = build_daily(g, weather=weather)
    dtr, dte = day_masks(d, test=True)
    ps, single = [], []
    t0 = time.time()
    for sd in SEEDS:
        p, info = fit_predict(d, dtr, dte, "Transformer", hp, target=TARGET, weights=WEIGHTS, seed=sd)
        ps.append(p); single.append(flat_metrics(d, dte, p)["MAE"])
        print(weather, sd, round(single[-1], 1), info["best_epoch"], flush=True)
    p = np.mean(ps, 0); preds[weather] = p
    m = flat_metrics(d, dte, p)
    out[weather] = dict(MAE=m["MAE"], RMSE=m["RMSE"], MAPE=m["MAPE"], PW_MAE=m["PW_MAE"], TD_MAE=m["TD_MAE"],
                        PeakMAE=m["PeakMAE"], single_seed_mean=float(np.mean(single)), single_seed_sd=float(np.std(single, ddof=1)),
                        minutes=(time.time() - t0) / 60)
    print(weather, {k: round(v, 2) for k, v in out[weather].items()}, flush=True)

Y = d["Y"][dte]; dates_te = d["dates"][dte]
dmae = {w: np.abs(Y - preds[w]).mean(1) for w in preds}
s, pval = dm_test(dmae["persist"], dmae["observed"])
out["dm_persist_vs_observed"] = dict(delta=float((dmae["persist"] - dmae["observed"]).mean()), DM=float(s), p=float(pval))
out["persist_gap_pct"] = 100 * (out["persist"]["MAE"] / out["observed"]["MAE"] - 1)

# ---- B. unsupervised day types (K-means on training-day demand profiles)
P_tr, P_te = d["Y"][dtr], Y
zs = StandardScaler().fit(P_tr)
Z_tr, Z_te = zs.transform(P_tr), zs.transform(P_te)
sil = {}
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit(Z_tr)
    sil[k] = float(silhouette_score(Z_tr, km.labels_))
k_best = max(sil, key=sil.get)
km = KMeans(n_clusters=k_best, n_init=20, random_state=SEED).fit(Z_tr)
lab_tr, lab_te = km.labels_, km.predict(Z_te)
dist_tr = np.min(km.transform(Z_tr), 1); dist_te = np.min(km.transform(Z_te), 1)
dates_tr = pd.to_datetime(d["dates"][dtr]); dte_ts = pd.to_datetime(dates_te)
rows = []
for c in range(k_best):
    mtr, mte = lab_tr == c, lab_te == c
    rows.append(dict(cluster=c, train_days=int(mtr.sum()), test_days=int(mte.sum()),
                     mean_mw=float(P_tr[mtr].mean()) if mtr.any() else np.nan,
                     peak_mw=float(P_tr[mtr].max(1).mean()) if mtr.any() else np.nan,
                     weekend_share=float((dates_tr[mtr].dayofweek >= 5).mean()) if mtr.any() else np.nan,
                     winter_share=float(dates_tr[mtr].month.isin([6, 7, 8]).mean()) if mtr.any() else np.nan,
                     test_mae_observed=float(dmae["observed"][mte].mean()) if mte.any() else np.nan,
                     test_mae_persist=float(dmae["persist"][mte].mean()) if mte.any() else np.nan))
out["kmeans"] = dict(k=k_best, silhouette=sil, clusters=rows,
                     novelty_threshold_p95=float(np.quantile(dist_tr, .95)),
                     test_days_beyond_p95=int((dist_te > np.quantile(dist_tr, .95)).sum()),
                     spearman_dist_vs_mae_observed=[float(x) for x in spearmanr(dist_te, dmae["observed"])],
                     spearman_dist_vs_mae_persist=[float(x) for x in spearmanr(dist_te, dmae["persist"])])
far = dist_te > np.quantile(dist_tr, .95)
out["kmeans"]["mae_far_vs_near"] = dict(observed=[float(dmae["observed"][far].mean()), float(dmae["observed"][~far].mean())],
                                         persist=[float(dmae["persist"][far].mean()), float(dmae["persist"][~far].mean())])
cold = (dte_ts >= "2026-08-04") & (dte_ts <= "2026-08-07")
out["kmeans"]["cold_spell"] = dict(clusters=[int(x) for x in lab_te[cold]], beyond_p95=[bool(x) for x in far[cold]],
                                    mae_observed=float(dmae["observed"][cold].mean()), mae_persist=float(dmae["persist"][cold].mean()))
pickle.dump(dict(preds=preds, dates=dates_te, y=Y, labels_test=lab_te, dist_test=dist_te), open(os.path.join(TAB, "weather_clusters_preds.pkl"), "wb"))
save_json(out, "weather_and_clusters.json")
print(json.dumps(out, indent=1, default=float))
