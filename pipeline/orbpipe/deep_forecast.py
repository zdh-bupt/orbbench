"""Deep time-series forecasters on the Table-4 columns (Sentinel-1A/2A/2B/3A/3B, Starlink x3, All Sentinel/Starlink/cohort).

Why retrain: the six checkpoints shipped in 模型训练记录/ belong to an older protocol (global z-score, 512-step input,
90-step output, different Informer/FEDformer layers) and do not load into the current src/models.py, and the reported
table came from a run on another machine. We therefore retrain the six supervised models with the current code's
architectures, representation (origin-RTN-relative, src/representation.py) and hyper-parameters (protocol Sec. 7 of
实验配置.md) on Pruned LEO trajectories, then run inference on the same windows as the SGP4 baselines.

Training   120 on-disk Pruned LEO objects (seeded selection), 1-min TEME, history 1440 steps, endpoints {30,60,240,480,
           720,1440} steps; train origins 2024-01-01..2025-01-01 stride 12 h, validation 2025-01-01..2025-06-01 stride 7 d;
           AdamW 1e-3 / 1e-4, batch 32, <=30 epochs, patience 10, 8,192 windows per epoch, cosine LR, AMP, clip 1.0, seed 42.
Inference  history = nearest-TLE SGP4 at 1 min for the 24 h before the origin (TLEs with epoch <= origin), truth = precise
           orbit (Sentinel POD, TEME), operator ephemeris (Starlink MEME -> TEME) or the L2 reference (cohort).
           Baselines on the same windows: SGP4-prev (latest TLE) and SGP4-MTF (causal fusion). Optional zero-shot
           Chronos / Chronos-2 (HF mirror), horizon adaptation as in src/evaluate.py.
Outputs    results/deep_forecast_long.csv (set, column, model, h, n, p50/mean/p95 km), deep_forecast_table_24h.csv,
           deep_forecast_table_1h.csv, deep_forecast_train_history.json; data/deep/ (arrays, checkpoints, windows).
"""
from __future__ import annotations
import sys, os, io, json, math, time, subprocess, importlib.util
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
from pathlib import Path
import numpy as np, pandas as pd

from . import config as C
from .util import get_logger, set_status, Timer, save_df, read_tles, sgp4_states, to_utc

log = get_logger("deep_forecast")
SRC = Path(os.environ.get("ORBBENCH_DEEP_SRC", str(C.SUMMARY_ROOT / "t4_deep_models/src")))   # models.py / representation.py of the eight-model study
DEEP = C.DATA / "deep"; (DEEP / "train").mkdir(parents=True, exist_ok=True); (DEEP / "ckpt").mkdir(exist_ok=True); (DEEP / "windows").mkdir(exist_ok=True)

def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
MODELS = _load("p2_models", SRC / "models.py"); REP = _load("p2_rep", SRC / "representation.py")

INPUT_STEPS = 1440; TARGET_STEPS = [30, 60, 240, 480, 720, 1440]; HZ = [0.5, 1.0, 4.0, 8.0, 12.0, 24.0]
TRAIN_START, TRAIN_END, VAL_START, VAL_END = [pd.Timestamp(s, tz="UTC") for s in ("2024-01-01", "2025-01-01", "2025-01-01", "2025-06-01")]
N_TRAIN_OBJ = int(os.environ.get("DEEP_N_OBJ", "120")); TRAIN_STRIDE = 720; VAL_STRIDE = 10080
MAX_EPOCHS = int(os.environ.get("DEEP_EPOCHS", "30")); PATIENCE = 10; WINDOWS_PER_EPOCH = 8192; BATCH = 32; LR = 1e-3; WD = 1e-4; SEED = 42
MODEL_NAMES = ["lstm", "gru", "dlinear", "tcn", "informer", "fedformer"]; LABEL = {"lstm": "LSTM", "gru": "GRU", "dlinear": "DLinear", "tcn": "TCN", "informer": "Informer", "fedformer": "FEDformer"}
STARLINK = {44752: "Starlink-1047", 44723: "Starlink-1017", 44751: "Starlink-1046"}
COLS = ["Sentinel-1A", "Sentinel-2A", "Sentinel-2B", "Sentinel-3A", "Sentinel-3B", "Starlink-1047", "Starlink-1017", "Starlink-1046", "All Sentinel", "All Starlink", "All cohort"]

# ============================================================================ A. training data
def _tail_parse(path: str, nbytes=150_000_000) -> pd.DataFrame | None:
    out = subprocess.run(["tail", "-c", str(nbytes), path], capture_output=True).stdout
    if len(out) < 1000: return None
    txt = out[out.find(b"\n") + 1:]
    d = pd.read_csv(io.BytesIO(txt), header=None, usecols=[0, 1, 2, 3, 4, 5, 6]); d.columns = ["t", "x", "y", "z", "vx", "vy", "vz"]
    d["t"] = pd.to_datetime(d.t, utc=True, errors="coerce", format="mixed"); d = d.dropna(subset=["t"]).sort_values("t").drop_duplicates("t")
    d = d[(d.t >= TRAIN_START - pd.Timedelta(days=2)) & (d.t < VAL_END + pd.Timedelta(days=2))]
    return d if len(d) > 3000 else None

def prepare_training() -> list[dict]:
    man = DEEP / "train_manifest.json"
    if man.exists():
        m = json.loads(man.read_text())
        if len(m) >= N_TRAIN_OBJ: return m[:N_TRAIN_OBJ]
    obj = pd.read_csv(C.DATA / "objects.csv"); leo = obj[(obj.source == "pruned") & (obj.regime == "LEO")].copy()
    leo = leo[leo.path.map(os.path.exists)]; rng = np.random.default_rng(SEED); leo = leo.iloc[rng.permutation(len(leo))]
    entries = []
    for r in leo.itertuples():
        if len(entries) >= N_TRAIN_OBJ: break
        try:
            d = _tail_parse(r.path)
            if d is None: continue
            arr = (d[["x", "y", "z", "vx", "vy", "vz"]].to_numpy(np.float64) / 1000.0).astype(np.float32)          # km, km/s
            tmin = (d.t.values.astype("datetime64[ns]").astype("int64") // 60_000_000_000).astype(np.int64)         # minutes
            if not np.isfinite(arr).all(): continue
            p = DEEP / "train" / f"{int(r.norad)}.npy"; pt = DEEP / "train" / f"{int(r.norad)}_t.npy"; np.save(p, arr); np.save(pt, tmin)
            entries.append(dict(norad=int(r.norad), path=str(p), tpath=str(pt), rows=int(len(arr)), type=str(r.type)))
            if len(entries) % 20 == 0: log.info(f"training arrays {len(entries)}/{N_TRAIN_OBJ}")
        except Exception as ex:
            log.warning(f"{r.norad}: {ex}")
    man.write_text(json.dumps(entries, indent=1)); return entries

def build_origins(entries, t_lo, t_hi, stride):
    """(entry_index, origin) pairs whose 2880-step window is contiguous at 1 min and whose last history time lies in [t_lo, t_hi)."""
    lo = int(t_lo.value // 60_000_000_000); hi = int(t_hi.value // 60_000_000_000); out = []
    for k, e in enumerate(entries):
        t = np.load(e["tpath"]); n = len(t)
        for o in range(INPUT_STEPS, n - TARGET_STEPS[-1] + 1, stride):
            if t[o + TARGET_STEPS[-1] - 1] - t[o - INPUT_STEPS] != INPUT_STEPS + TARGET_STEPS[-1] - 1: continue
            if lo <= t[o - 1] < hi: out.append((k, o))
    return out

class WindowSet:
    def __init__(self, entries, pairs): self.entries = entries; self.pairs = pairs; self.cache = {}
    def __len__(self): return len(self.pairs)
    def arr(self, k):
        if k not in self.cache: self.cache[k] = np.load(self.entries[k]["path"], mmap_mode="r")
        return self.cache[k]
    def __getitem__(self, i):
        import torch
        k, o = self.pairs[i]; a = self.arr(k); ctx = np.asarray(a[o - INPUT_STEPS:o], np.float32); tgt = np.asarray(a[o + np.asarray(TARGET_STEPS) - 1], np.float32)
        ec, et = REP.encode_window(ctx, tgt); return torch.from_numpy(ec), torch.from_numpy(et), torch.from_numpy(ctx[-1].copy())

# ============================================================================ B. training
def train_model(name: str, train_ws: WindowSet, val_ws: WindowSet, device) -> dict:
    import torch
    from torch.utils.data import DataLoader, RandomSampler
    ck = DEEP / "ckpt" / f"{name}.pt"
    if ck.exists():
        pl = torch.load(ck, map_location="cpu", weights_only=False)
        if pl.get("complete"): log.info(f"{name}: checkpoint exists (best epoch {pl['best_epoch']})"); return pl["history"]
    torch.manual_seed(SEED); np.random.seed(SEED)
    cfg = MODELS.model_config(name, INPUT_STEPS, TARGET_STEPS); model = MODELS.build_model(name, cfg).to(device)
    gen = torch.Generator().manual_seed(SEED)
    sampler = RandomSampler(train_ws, replacement=False, num_samples=min(WINDOWS_PER_EPOCH, len(train_ws)), generator=gen)
    tl = DataLoader(train_ws, batch_size=BATCH, sampler=sampler, num_workers=4, pin_memory=device.type == "cuda", persistent_workers=True)
    vl = DataLoader(val_ws, batch_size=128, shuffle=False, num_workers=4, pin_memory=device.type == "cuda", persistent_workers=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS)
    use_amp = device.type == "cuda" and name != "fedformer"                      # cuFFT has no half-precision path for length 1440
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp); best = math.inf; best_ep = 0; bad = 0; hist = []
    for ep in range(1, MAX_EPOCHS + 1):
        model.train(); t0 = time.time(); ls = 0.0; nb = 0
        for x, y, _ in tl:
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True); opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                loss = torch.mean((model(x) - y) ** 2)
            scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update()
            ls += float(loss.item()); nb += 1
        sched.step()
        # validation in physical units
        model.eval(); errs = []
        with torch.inference_mode():
            for x, y, org in vl:
                p = model(x.to(device)).float().cpu().numpy(); o = org.numpy().astype(np.float64)
                pr = REP.decode_states(p, o); tr = REP.decode_states(y.numpy(), o); errs.append(np.linalg.norm(pr[:, :, :3] - tr[:, :, :3], axis=2))
        e = np.concatenate(errs); pos_mean = e.mean(0); score = float(pos_mean.mean())
        hist.append(dict(epoch=ep, train_mse=ls / max(nb, 1), val_pos_mean_km=[float(v) for v in pos_mean], val_pos_p50_km=[float(v) for v in np.median(e, 0)], score=score, seconds=round(time.time() - t0, 1)))
        log.info(f"{name} ep{ep:02d} train_mse {ls / max(nb, 1):.3e} val mean km {np.round(pos_mean, 2).tolist()} ({time.time() - t0:.0f}s)")
        if score < best: best = score; best_ep = ep; bad = 0; torch.save(dict(model=name, model_config=cfg, state_dict=model.state_dict(), best_epoch=ep, history=hist, complete=False), ck)
        else: bad += 1
        if bad >= PATIENCE: break
    pl = torch.load(ck, map_location="cpu", weights_only=False); pl["history"] = hist; pl["complete"] = True; pl["quality_gate_1h_le_10km"] = bool(hist[best_ep - 1]["val_pos_mean_km"][1] <= 10.0); torch.save(pl, ck)
    return hist

# ============================================================================ C. inference windows
def tle_history(tles: pd.DataFrame, t0: pd.Timestamp) -> np.ndarray | None:
    """Nearest-TLE SGP4 states (TEME km, km/s) at 1 min for [t0-1439 min, t0], using TLEs with epoch <= t0."""
    c = tles[(tles.epoch <= t0) & (tles.epoch >= t0 - pd.Timedelta(days=6))]
    if len(c) == 0: c = tles[tles.epoch <= t0].tail(1)
    if len(c) == 0: return None
    times = pd.date_range(t0 - pd.Timedelta(minutes=INPUT_STEPS - 1), t0, freq="1min", tz="UTC")
    ep = c.epoch.values.astype("datetime64[ns]").astype("int64"); tt = times.values.astype("datetime64[ns]").astype("int64")
    idx = np.abs(tt[:, None] - ep[None, :]).argmin(1); out = np.full((len(times), 6), np.nan)
    for j in np.unique(idx):
        m = idx == j; r = c.iloc[int(j)]; out[m] = sgp4_states(r.line1, r.line2, times[m])
    return out if np.isfinite(out).all() else None

def baselines(tles: pd.DataFrame, t0: pd.Timestamp, targets: pd.DatetimeIndex):
    from .t4_forecast import _causal_tles, _fuse
    prev = tles[tles.epoch <= t0].tail(1)
    sp = sgp4_states(prev.line1.iloc[0], prev.line2.iloc[0], targets) if len(prev) else np.full((len(targets), 6), np.nan)
    sm, _, _ = _fuse(_causal_tles(tles, t0), t0, targets)
    return sp, sm

def _hermite_km(df: pd.DataFrame, grid: pd.DatetimeIndex, max_gap_s=90.0) -> np.ndarray:
    from .stage0_staging import _hermite_km as H
    return H(df, grid, 1.0, max_gap_s=max_gap_s).astype(np.float64)

def windows_sentinel():
    p = DEEP / "windows" / "sentinel.npz"
    if p.exists(): return
    ctx, tru, sp, sm, meta = [], [], [], [], []
    for name, (norad, _, _) in C.SENTINELS.items():
        pod = pd.read_parquet(C.DATA / f"sentinel_pod_{name}.parquet"); pod["time"] = pd.to_datetime(pod.time, utc=True)
        pod = pod.set_index("time")[["x_teme", "y_teme", "z_teme", "vx_teme", "vy_teme", "vz_teme"]]
        origins = pd.date_range(to_utc(C.SENTINEL_TEST_START), to_utc(C.WINDOW_END) - pd.Timedelta(hours=24), freq="12h", tz="UTC")
        tles = read_tles(norad, origins[0] - pd.Timedelta(days=10), origins[-1] + pd.Timedelta(days=1)); n_ok = 0
        for t0 in origins:
            targets = pd.DatetimeIndex([t0 + pd.Timedelta(minutes=s) for s in TARGET_STEPS])
            tr = pod.reindex(targets).to_numpy(np.float64)
            if not np.isfinite(tr).all(): continue
            h = tle_history(tles, t0)
            if h is None: continue
            a, b = baselines(tles, t0, targets); ctx.append(h.astype(np.float32)); tru.append(tr); sp.append(a); sm.append(b); meta.append(dict(set="sentinel", column=f"Sentinel-{name[1:]}", norad=norad, t0=t0)); n_ok += 1
        log.info(f"sentinel {name}: {n_ok} windows")
    np.savez_compressed(p, ctx=np.stack(ctx), truth=np.stack(tru), sgp4_prev=np.stack(sp), sgp4_mtf=np.stack(sm)); pd.DataFrame(meta).to_parquet(DEEP / "windows" / "sentinel_meta.parquet", index=False)

def windows_starlink():
    p = DEEP / "windows" / "starlink.npz"
    if p.exists(): return
    from .starlink_chain import read_txt, W0, W1, TRAIN_END as SL_TRAIN_END
    from .stage_starlink import RAW_TLE_DIR
    from .stage0_staging import gcrs_to_teme
    ctx, tru, sp, sm, meta = [], [], [], [], []
    for norad, label in STARLINK.items():
        m = pd.read_parquet(C.DATA / f"starlink/meme_{norad}.parquet"); m["time"] = pd.to_datetime(m.time, utc=True); m = m.set_index("time").sort_index()[["x", "y", "z", "vx", "vy", "vz"]]
        tles = read_txt(RAW_TLE_DIR / f"{norad}.txt", W0 - pd.Timedelta(days=10), W1)
        origins = pd.date_range(SL_TRAIN_END, W1 - pd.Timedelta(hours=24), freq="3h", tz="UTC"); n_ok = 0
        for t0 in origins:
            targets = pd.DatetimeIndex([t0 + pd.Timedelta(minutes=s) for s in TARGET_STEPS])
            g = _hermite_km(m, targets)
            if not np.isfinite(g).all(): continue
            tr = gcrs_to_teme(g, targets)
            h = tle_history(tles, t0)
            if h is None: continue
            a, b = baselines(tles, t0, targets); ctx.append(h.astype(np.float32)); tru.append(tr); sp.append(a); sm.append(b); meta.append(dict(set="starlink", column=label, norad=norad, t0=t0)); n_ok += 1
        log.info(f"starlink {norad} ({label}): {n_ok} windows")
    np.savez_compressed(p, ctx=np.stack(ctx), truth=np.stack(tru), sgp4_prev=np.stack(sp), sgp4_mtf=np.stack(sm)); pd.DataFrame(meta).to_parquet(DEEP / "windows" / "starlink_meta.parquet", index=False)

def windows_cohort():
    p = DEEP / "windows" / "cohort.npz"
    if p.exists(): return
    z = np.load(C.DATA / "c4_leo_1500_3d.npz"); traj = z["traj"].astype(np.float64); ids = z["ids"].astype(int); grid = pd.DatetimeIndex(z["grid"]).tz_localize("UTC")
    tl = pd.read_parquet(C.DATA / "e2e_tles_1500.parquet"); tl["epoch"] = pd.to_datetime(tl.epoch, utc=True); tles = {int(n): g.sort_values("epoch").reset_index(drop=True) for n, g in tl.groupby("norad")}
    ctx, tru, sp, sm, meta = [], [], [], [], []
    for o in (1440, 2160, 2880):
        t0 = grid[o - 1]; targets = pd.DatetimeIndex([t0 + pd.Timedelta(minutes=s) for s in TARGET_STEPS]); ti = o - 1 + np.asarray(TARGET_STEPS)
        for k, n in enumerate(ids):
            h = traj[k, o - INPUT_STEPS:o]; tr = traj[k, ti]
            if not (np.isfinite(h).all() and np.isfinite(tr).all()) or int(n) not in tles: continue
            a, b = baselines(tles[int(n)], t0, targets); ctx.append(h.astype(np.float32)); tru.append(tr); sp.append(a); sm.append(b); meta.append(dict(set="cohort", column="All cohort", norad=int(n), t0=t0))
        log.info(f"cohort origin {t0}: {len(meta)} windows so far")
    np.savez_compressed(p, ctx=np.stack(ctx), truth=np.stack(tru), sgp4_prev=np.stack(sp), sgp4_mtf=np.stack(sm)); pd.DataFrame(meta).to_parquet(DEEP / "windows" / "cohort_meta.parquet", index=False)

# ============================================================================ D. predictions
def predict_trained(name: str, ctx: np.ndarray, device) -> np.ndarray:
    import torch
    pl = torch.load(DEEP / "ckpt" / f"{name}.pt", map_location="cpu", weights_only=False); model = MODELS.build_model(name, pl["model_config"]).to(device); model.load_state_dict(pl["state_dict"], strict=True); model.eval()
    out = []
    with torch.inference_mode():
        for i in range(0, len(ctx), 256):
            c = ctx[i:i + 256].astype(np.float64); org = c[:, -1]; enc = REP.encode_states(c, org)
            p = model(torch.as_tensor(enc, dtype=torch.float32, device=device)).float().cpu().numpy(); out.append(REP.decode_states(p, org))
    return np.concatenate(out)

def exact_cadence(step: int, maxlen=60) -> int:
    for c in range(max(1, math.ceil(step / maxlen)), step + 1):
        if step % c == 0: return c
    return step

CHRONOS_BATCH = {"chronos": int(os.environ.get("CHRONOS_T5_BATCH", "2")), "chronos2": 64}
CHRONOS_T5_MAX = int(os.environ.get("CHRONOS_T5_MAX_WINDOWS", "600"))   # T5-large is autoregressive with 20 samples: subsample windows per set

def predict_chronos(kind: str, ctx: np.ndarray, device) -> np.ndarray:
    """Zero-shot Chronos ('chronos', amazon/chronos-t5-large, sample median) or Chronos-2 (quantile 0.5), per-window per-channel
    z-score on the RTN-relative representation, exact-endpoint horizon adaptation as in src/evaluate.py."""
    import torch
    from chronos import Chronos2Pipeline, ChronosPipeline
    mid = "amazon/chronos-t5-large" if kind == "chronos" else "amazon/chronos-2"
    pipe = (ChronosPipeline if kind == "chronos" else Chronos2Pipeline).from_pretrained(mid, device_map="cuda" if device.type == "cuda" else "cpu", torch_dtype=torch.bfloat16)
    out = []; B = CHRONOS_BATCH[kind]; sel = np.arange(len(ctx))
    if kind == "chronos" and len(ctx) > CHRONOS_T5_MAX: sel = np.sort(np.random.default_rng(SEED).choice(len(ctx), CHRONOS_T5_MAX, replace=False))
    full = np.full((len(ctx), len(TARGET_STEPS), 6), np.nan); ctx_s = ctx[sel]
    for i in range(0, len(ctx_s), B):
        c = ctx_s[i:i + B].astype(np.float64); org = c[:, -1]; enc = REP.encode_states(c, org).transpose(0, 2, 1)     # [B,6,L]
        mean = enc.mean(2, keepdims=True); std = enc.std(2, keepdims=True); std = np.where(std < 1e-8, 1.0, std); z = (enc - mean) / std
        ends = []
        for step in TARGET_STEPS:
            cad = exact_cadence(step); hist = np.ascontiguousarray(z[:, :, ::-cad][:, :, ::-1]); L = step // cad
            if kind == "chronos":
                flat = torch.as_tensor(hist.reshape(-1, hist.shape[-1]), dtype=torch.float32)
                s = pipe.predict(flat, prediction_length=L, num_samples=20)
                if isinstance(s, list): s = torch.stack(s)
                pt = s.median(dim=1).values[:, -1].float().cpu().numpy().reshape(hist.shape[0], 6)
            else:
                preds = pipe.predict([torch.as_tensor(h, dtype=torch.float32) for h in hist], prediction_length=L)
                qs = [float(q) for q in pipe.quantiles]; qi = min(range(len(qs)), key=lambda k: abs(qs[k] - 0.5))
                pt = np.stack([p.float().cpu().numpy()[:, qi, -1] for p in preds])
            ends.append(pt)
        pred = np.stack(ends, 1) * std.transpose(0, 2, 1) + mean.transpose(0, 2, 1); out.append(REP.decode_states(pred, org))
        if (i // B) % 20 == 0: log.info(f"{kind}: {i + len(c)}/{len(ctx_s)}")
    full[sel] = np.concatenate(out); del pipe; torch.cuda.empty_cache(); return full

# ============================================================================ E. main
def main():
    import torch
    set_status("S8m_deep_forecast", "running")
    try:
        with Timer() as T:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); log.info(f"device {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'}), torch {torch.__version__}")
            entries = prepare_training(); log.info(f"training objects: {len(entries)}")
            tr_pairs = build_origins(entries, TRAIN_START, TRAIN_END, TRAIN_STRIDE); va_pairs = build_origins(entries, VAL_START, VAL_END, VAL_STRIDE)
            log.info(f"train windows {len(tr_pairs)}, val windows {len(va_pairs)}")
            train_ws, val_ws = WindowSet(entries, tr_pairs), WindowSet(entries, va_pairs); hist = {}
            for name in MODEL_NAMES:
                hist[name] = train_model(name, train_ws, val_ws, device); (C.RESULTS / "deep_forecast_train_history.json").write_text(json.dumps(hist, indent=1))
            windows_sentinel(); windows_starlink(); windows_cohort()
            rows = []; per = []
            for s in ("sentinel", "starlink", "cohort"):
                z = np.load(DEEP / "windows" / f"{s}.npz"); meta = pd.read_parquet(DEEP / "windows" / f"{s}_meta.parquet"); ctx = z["ctx"]; truth = z["truth"]
                preds = {"SGP4-prev": z["sgp4_prev"], "SGP4-MTF": z["sgp4_mtf"]}; PC = DEEP / "preds"; PC.mkdir(exist_ok=True)
                def cached(lab, fn):
                    f = PC / f"{s}_{lab}.npy"
                    if f.exists(): return np.load(f)
                    p = fn(); np.save(f, p); return p
                for name in MODEL_NAMES: preds[LABEL[name]] = cached(LABEL[name], lambda: predict_trained(name, ctx, device)); log.info(f"{s}: {name} predicted")
                if os.environ.get("DEEP_CHRONOS", "1") == "1":
                    for kind, lab in (("chronos2", "Chronos-2"), ("chronos", "Chronos")):
                        try: preds[lab] = cached(lab, lambda: predict_chronos(kind, ctx, device)); log.info(f"{s}: {lab} predicted")
                        except Exception as ex: log.warning(f"{s}: {lab} failed: {ex}"); torch.cuda.empty_cache()
                for lab, p in preds.items():
                    e = np.linalg.norm(p[:, :, :3] - truth[:, :, :3], axis=2)                                   # [N,6] km
                    for j, h in enumerate(HZ):
                        allc = {"sentinel": "All Sentinel", "starlink": "All Starlink", "cohort": "All cohort"}[s]
                        for col, m in [(c_, g_) for c_, g_ in meta.groupby("column").groups.items() if c_ != allc] + [(allc, meta.index)]:
                            v = e[np.asarray(m), j]; v = v[np.isfinite(v)]
                            if len(v): rows.append(dict(set=s, column=col, model=lab, h=h, n=int(len(v)), p50_km=float(np.median(v)), mean_km=float(v.mean()), p95_km=float(np.percentile(v, 95))))
                    d = meta.copy(); d["model"] = lab
                    for j, h in enumerate(HZ): d[f"err_{h:g}h"] = e[:, j]
                    per.append(d)
            long = pd.DataFrame(rows); save_df(long, "deep_forecast_long.csv"); pd.concat(per, ignore_index=True).to_parquet(C.RESULTS / "deep_forecast_cases.parquet", index=False)
            order = ["SGP4-prev", "SGP4-MTF"] + [LABEL[n] for n in MODEL_NAMES] + ["Chronos", "Chronos-2"]
            for h in (24.0, 1.0):
                t = long[long.h == h].pivot_table(index="model", columns="column", values="p50_km", aggfunc="first").reindex(index=[m for m in order if m in set(long.model)], columns=COLS)
                save_df(t.round(3).reset_index(), f"deep_forecast_table_{h:g}h.csv"); log.info(f"P50 km at {h:g} h:\n" + t.round(2).to_string())
        set_status("S8m_deep_forecast", "done", elapsed_s=round(T.elapsed))
    except Exception as ex:
        log.exception("deep_forecast failed"); set_status("S8m_deep_forecast", "failed", message=str(ex)); raise

if __name__ == "__main__":
    main()
