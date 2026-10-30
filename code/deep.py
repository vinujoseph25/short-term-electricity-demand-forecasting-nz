"""LSTM and patch-Transformer day-ahead forecasters (PyTorch), shared data builder and trainer."""
import warnings; warnings.filterwarnings("ignore", message="enable_nested_tensor")
import torch, torch.nn as nn
from common import *
torch.set_num_threads(int(os.environ.get("TORCH_THREADS", 2)))

def build_daily(g, weather="observed", city="popw"):
    """Arrays indexed by target day D. hist: (7*48, Fh) for days D-7..D-1; fut: (48, Ff) for day D."""
    Y = day_matrix(g, "mw").values; T = day_matrix(g, WEATHER_COLS[city]).values
    dates = sorted(g.date.unique()); hol = g.groupby("date").holiday.first().values
    idx = [i for i, d in enumerate(dates) if d >= FIRST_TARGET_DAY]
    s = np.arange(48); ssin, scos = np.sin(2 * np.pi * s / 48), np.cos(2 * np.pi * s / 48)
    H, Fu, Yt, O, L1, D = [], [], [], [], [], []
    for i in idx:
        dt = pd.Timestamp(dates[i]); hrows = []
        for j in range(i - 7, i):
            dj = pd.Timestamp(dates[j])
            hrows.append(np.stack([Y[j], T[j], np.full(48, hol[j]), np.full(48, dj.dayofweek >= 5), ssin, scos], 1))
        H.append(np.concatenate(hrows, 0))
        if weather == "observed": tt = T[i]
        elif weather == "persist": tt = T[i - 1]
        else: tt = np.zeros(48)
        dow = np.zeros((48, 7)); dow[:, dt.dayofweek] = 1
        nxt = hol[i + 1] if i + 1 < len(dates) else 0
        Fu.append(np.column_stack([tt, np.maximum(0, HDD_BASE - tt) if weather != "none" else np.zeros(48), ssin, scos, dow,
                                   np.full(48, hol[i]), np.full(48, hol[i - 1]), np.full(48, nxt),
                                   np.full(48, np.sin(2 * np.pi * dt.dayofyear / 365.25)), np.full(48, np.cos(2 * np.pi * dt.dayofyear / 365.25)),
                                   Y[i - 1], Y[i - 7]]))
        Yt.append(Y[i]); O.append(Y[i - 1, 47]); L1.append(Y[i - 1]); D.append(dates[i])
    return dict(H=np.array(H, np.float32), F=np.array(Fu, np.float32), Y=np.array(Yt, np.float32),
                O=np.array(O, np.float32), L1=np.array(L1, np.float32), dates=np.array(D))

class Scaler:
    """Fitted on training days only: demand and temperature standardisation."""
    def fit(self, d, m):
        self.ym, self.ys = d["Y"][m].mean(), d["Y"][m].std()
        self.tm, self.ts = d["F"][m][:, :, 0].mean(), d["F"][m][:, :, 0].std() + 1e-6
        self.hm, self.hs = d["F"][m][:, :, 1].mean(), d["F"][m][:, :, 1].std() + 1e-6
        return self
    def transform(self, d, m):
        H = d["H"][m].copy(); F = d["F"][m].copy()
        H[:, :, 0] = (H[:, :, 0] - self.ym) / self.ys; H[:, :, 1] = (H[:, :, 1] - self.tm) / self.ts
        F[:, :, 0] = (F[:, :, 0] - self.tm) / self.ts; F[:, :, 1] = (F[:, :, 1] - self.hm) / self.hs
        F[:, :, -2:] = (F[:, :, -2:] - self.ym) / self.ys
        return H, F

class Head(nn.Module):
    def __init__(self, zdim, ff, hh, drop):
        super().__init__(); self.net = nn.Sequential(nn.Linear(zdim + ff, hh), nn.GELU(), nn.Dropout(drop), nn.Linear(hh, 1))
    def forward(self, z, F):                      # z: (B, zdim), F: (B, 48, ff)
        return self.net(torch.cat([z.unsqueeze(1).expand(-1, F.shape[1], -1), F], -1)).squeeze(-1)

class LSTMNet(nn.Module):
    def __init__(self, fh, ff, hidden=64, layers=1, dropout=.1, head_hidden=64, **_):
        super().__init__()
        self.lstm = nn.LSTM(fh, hidden, layers, batch_first=True, dropout=dropout if layers > 1 else 0)
        self.head = Head(hidden, ff, head_hidden, dropout)
    def forward(self, H, F):
        # 336 half-hours -> 168 hourly steps (mean of each pair), matching the 168-h lookback of Bansal (2026)
        Hh = H.reshape(H.shape[0], 168, 2, H.shape[-1]).mean(2)
        _, (h, _) = self.lstm(Hh); return self.head(h[-1], F)

class PatchTransformer(nn.Module):
    """Each past day (48 half-hours x Fh) is one token; the target day's known covariates form an 8th token."""
    def __init__(self, fh, ff, d_model=64, heads=4, layers=2, dropout=.1, head_hidden=64, **_):
        super().__init__()
        self.hist_proj = nn.Linear(48 * fh, d_model); self.fut_proj = nn.Linear(48 * ff, d_model)
        self.pos = nn.Parameter(torch.randn(8, d_model) * .02)
        enc = nn.TransformerEncoderLayer(d_model, heads, 2 * d_model, dropout, batch_first=True, norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(enc, layers); self.norm = nn.LayerNorm(d_model)
        self.head = Head(d_model, ff, head_hidden, dropout)
    def forward(self, H, F):
        B = H.shape[0]; tok = torch.cat([self.hist_proj(H.reshape(B, 7, -1)), self.fut_proj(F.reshape(B, 1, -1))], 1) + self.pos
        z = self.norm(self.enc(tok))[:, -1]; return self.head(z, F)

ARCH = {"LSTM": LSTMNet, "Transformer": PatchTransformer}

def fit_predict(d, tr, te, arch, hp, target="abs", weights="none", seed=SEED, max_epochs=120, patience=12, return_model=False):
    torch.manual_seed(seed); np.random.seed(seed)
    tr_idx = np.where(tr)[0]; n_in = max(14, int(round(.1 * len(tr_idx))))
    fit_idx, es_idx = tr_idx[:-n_in], tr_idx[-n_in:]           # chronological inner split for early stopping
    fm = np.zeros(len(tr), bool); fm[fit_idx] = True
    sc = Scaler().fit(d, fm)
    def base(m):
        if target == "abs": return np.zeros_like(d["Y"][m])
        if target == "delta_origin": return np.repeat(d["O"][m][:, None], 48, 1)
        if target == "delta_seasonal": return d["L1"][m]
    def tens(m):
        H, F = sc.transform(d, m); yt = (d["Y"][m] - base(m) - (sc.ym if target == "abs" else 0)) / sc.ys
        w = make_weights(d["Y"][m].ravel(), weights, ref=d["Y"][fm].ravel()).reshape(-1, 48)
        return [torch.tensor(a, dtype=torch.float32) for a in (H, F, yt, w)]
    esm = np.zeros(len(tr), bool); esm[es_idx] = True
    Htr, Ftr, Ytr, Wtr = tens(fm); Hes, Fes, Yes, Wes = tens(esm)
    model = ARCH[arch](Htr.shape[-1], Ftr.shape[-1], **hp)
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=1e-4)
    bs = hp.get("batch", 32); best, best_state, bad, hist = np.inf, None, 0, []
    for ep in range(max_epochs):
        model.train(); perm = torch.randperm(len(Htr)); tl = 0
        for b in range(0, len(perm), bs):
            ii = perm[b:b + bs]; opt.zero_grad()
            loss = (Wtr[ii] * (model(Htr[ii], Ftr[ii]) - Ytr[ii]).abs()).mean(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); tl += loss.item() * len(ii)
        model.eval()
        with torch.no_grad(): vl = (Wes * (model(Hes, Fes) - Yes).abs()).mean().item()
        hist.append((tl / len(Htr), vl))
        if vl < best - 1e-5: best, best_state, bad = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience: break
    model.load_state_dict(best_state); model.eval()
    H, F = sc.transform(d, te)
    with torch.no_grad(): out = model(torch.tensor(H), torch.tensor(F)).numpy()
    pred = out * sc.ys + (sc.ym if target == "abs" else 0) + base(te)
    info = dict(epochs=len(hist), best_epoch=int(np.argmin([h[1] for h in hist]) + 1), history=hist,
                n_params=sum(p.numel() for p in model.parameters()))
    return (pred, info, model, sc) if return_model else (pred, info)

def day_masks(d, fold=None, test=False):
    dd = pd.Series(d["dates"])
    if test: return (dd < TEST_START).values, ((dd >= TEST_START) & (dd <= TEST_END)).values
    vs, ve = FOLDS[fold]; return (dd < vs).values, ((dd >= vs) & (dd <= ve)).values

def flat_metrics(d, m, pred):
    y = d["Y"][m]; dates = np.repeat(d["dates"][m], 48); slots = np.tile(np.arange(48), m.sum())
    return metrics(y.ravel(), pred.ravel(), dates, slots)
