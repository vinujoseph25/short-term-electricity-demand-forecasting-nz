"""Learning curve (training-window length) and neural-network training curves."""
import pickle, matplotlib.pyplot as plt
from common import *
set_style()
r = pd.read_pickle(os.path.join(TAB, "ablation_lgbm.pkl")); ti = json.load(open(os.path.join(TAB, "tabular_info.json")))
pts = []
for e, n in [("E|last 90 train days", 90), ("E|last 180 train days", 180), ("E|last 270 train days", 270), ("A|abs|none", None)]:
    q = r[r.exp == e].groupby("fold")[["MAE", "TD_MAE"]].mean()
    pts.append((n or np.mean([f["train_days"] for f in ti["folds"]]), q.MAE.mean(), q.MAE.std(), q.TD_MAE.mean()))
pts = np.array(pts)
fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.5))
axs[0].errorbar(pts[:, 0], pts[:, 1], yerr=pts[:, 2], marker="o", ms=4, color=COLORS[0], capsize=3, label="MAE (± fold SD)")
axs[0].plot(pts[:, 0], pts[:, 3], marker="s", ms=4, color=COLORS[1], label="Top-decile MAE")
axs[0].set_xlabel("Training days used (most recent)"); axs[0].set_ylabel("CV error (MW)"); axs[0].legend(fontsize=7)
sel = json.load(open(os.path.join(TAB, "selection.json")))
for k, (arch, tag) in enumerate([("LSTM", sel["LSTM_main_tag"]), ("Transformer", sel["Transformer_main_tag"])]):
    h = np.array(pickle.load(open(os.path.join(TAB, f"history_{arch}_{tag}.pkl"), "rb")))
    axs[1].plot(np.arange(1, len(h) + 1), h[:, 0], color=COLORS[k], lw=1, ls="--", label=f"{arch} train")
    axs[1].plot(np.arange(1, len(h) + 1), h[:, 1], color=COLORS[k], lw=1.5, label=f"{arch} early-stop set")
axs[1].set_xlabel("Epoch"); axs[1].set_ylabel("Weighted L1 (scaled)"); axs[1].legend(fontsize=7)
fig.tight_layout(); savefig(fig, "fig_learning"); plt.close(fig)
print(pts.round(1))
