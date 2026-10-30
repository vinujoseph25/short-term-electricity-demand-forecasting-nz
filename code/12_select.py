"""Step 12: fix every selection decision from the CV results BEFORE the test set is scored (writes selection.json)."""
import subprocess
from common import *
subprocess.run(["python3", "11_cv_figures.py"], check=True, capture_output=True)
S = pd.read_pickle(os.path.join(TAB, "ablation_summary.pkl"))
rg = pd.read_csv(os.path.join(TAB, "tune_ridge.csv"))
sel = dict(ridge_alpha=float(rg.loc[rg.MAE.idxmin(), "alpha"]), lgbm_params=json.load(open(os.path.join(TAB, "best_lgbm.json"))),
           rule_variant="per architecture: the target x loss combination with the lowest mean CV PW-MAE (RO2 candidate), plus the abs/L1 reference",
           rule_final="lowest mean CV PW-MAE among all carried variants whose mean CV MAE is within 5% of the best mean CV MAE")
cands = []
for arch in ["LightGBM", "LSTM", "Transformer"]:
    s = S[S.arch == arch]; nb = s[~((s.target == "abs") & (s.weights == "none"))]; b = nb.loc[nb.PW_MAE.idxmin()]
    v = {"abs-L1": ["abs", "none"], f"{b.target}-{b.weights}": [b.target, b.weights]}
    sel[f"{arch}_variants" if arch != "LightGBM" else "lgbm_variants"] = v
    if arch != "LightGBM": sel[f"{arch}_hp"] = json.load(open(os.path.join(TAB, f"best_{arch}.json")))
    for tag, (t, w) in v.items():
        r = s[(s.target == t) & (s.weights == w)].iloc[0]; cands.append(dict(name=f"{arch} {tag}", MAE=r.MAE, PW=r.PW_MAE))
C = pd.DataFrame(cands); ok = C[C.MAE <= 1.05 * C.MAE.min()]; final = ok.loc[ok.PW.idxmin(), "name"]
sel["final_model"] = final; sel["candidates_cv"] = C.round(2).to_dict(orient="records")
arch_f = final.split()[0]
sel["LSTM_main_tag"] = [k for k in sel["LSTM_variants"] if k != "abs-L1"][0] if arch_f != "LSTM" else final.split()[1]
sel["Transformer_main_tag"] = [k for k in sel["Transformer_variants"] if k != "abs-L1"][0] if arch_f != "Transformer" else final.split()[1]
sel["shap_variant"] = [k for k in sel["lgbm_variants"] if k != "abs-L1"][0]
best_of = {a: C[C.name.str.startswith(a)].sort_values("PW").name.iloc[0] for a in ["LightGBM", "LSTM", "Transformer"]}
pairs = [[final, "Persistence D-1"], [final, "Persistence D-7"], [final, "Ridge"]] + [[final, best_of[a]] for a in best_of if best_of[a] != final]
for a in ["LightGBM", "LSTM", "Transformer"]:
    vv = sel["lgbm_variants" if a == "LightGBM" else f"{a}_variants"]; ro2 = [k for k in vv if k != "abs-L1"][0]; pairs.append([f"{a} {ro2}", f"{a} abs-L1"])
sel["dm_pairs"] = pairs; sel["dm_pairs_pw"] = [p for p in pairs[-3:]]
sel["plot_models"] = [final] + [best_of[a] for a in best_of if best_of[a] != final][:1] + ["Ridge"]
sel["decided_at"] = time.strftime("%Y-%m-%d %H:%M:%S %Z")
json.dump(sel, open(os.path.join(TAB, "selection.json"), "w"), indent=1); print(json.dumps(sel, indent=1))
