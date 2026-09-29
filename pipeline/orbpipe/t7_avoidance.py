"""T7 Autonomous collision avoidance on high-risk events from T6 (screening events with predicted miss < 2 km
plus ESA CDM events with miss_distance < 2 km), in the relative-motion setting used by the existing use case.

Environment: relative position/velocity at TCA (RTN or Cartesian), 1-s steps, max 300 s, active satellite
thrusts with acceleration ≤ 0.3 m/s²; passive object ballistic; linear relative motion (Clohessy–Wiltshire
terms are negligible over 5 min at the encounter scale). Safety distance 2 km.
Strategies
  No-maneuver
  Physics-max      fixed direction (initial line of sight), max thrust until separation ≥ safety distance
  Physics-Pc       same direction, thrust only while Pc(σ) ≥ 1e-4 (σ from T4 σ-age via T6 events)  [chained-UQ]
  Proportional     thrust ∝ (safety - d)/safety along fixed direction
  PPO              stable-baselines3 PPO on thrust scalar α∈[0,1] (fixed direction) if the package is available
Metrics: safety rate (miss ≥ 2 km), total Δv (m/s), maneuver time, post-maneuver Pc; per strategy and per event source.
Outputs: results/t7_avoidance_metrics.csv, results/t7_avoidance_episodes.csv
"""
from __future__ import annotations
import sys
import numpy as np, pandas as pd
from . import config as C
from .util import get_logger, set_status, stage_done, Timer, save_df
from .t6_warning import pc_2d

log = get_logger("t7_avoidance")
SAFE_M = 2000.0; AMAX = 0.3; DT = 1.0; TMAX = 150; HBR_M = 8.0   # episode = [TCA-150 s, TCA]

def load_events() -> pd.DataFrame:
    ev = []
    p = C.RESULTS / "t6_events_for_t7.csv"
    if p.exists():
        d = pd.read_csv(p); d = d[(d.lead_h == d.lead_h.min()) & (d.miss_km < 2.0)]
        for r in d.itertuples():
            ev.append(dict(source="screening", event_id=f"{r.i}-{r.j}", rel_r_m=np.array([r.rx, r.ry, r.rz]) * 1000, rel_v_m_s=np.array([r.vx, r.vy, r.vz]) * 1000,
                           sigma_m=float(max(getattr(r, "age_h", 24.0), 1.0)) * 0 + 0.0, miss_m=r.miss_km * 1000))
    c = pd.read_csv(C.CDM_CSV); c = c[c.miss_distance < 2000]
    for r in c.itertuples():
        rr = np.array([r.relative_position_r, r.relative_position_t, r.relative_position_n], float); vv = np.array([r.relative_velocity_r, r.relative_velocity_t, r.relative_velocity_n], float)
        ev.append(dict(source="cdm", event_id=str(r.event_id), rel_r_m=rr, rel_v_m_s=vv, sigma_m=float(np.sqrt(r.t_sigma_r ** 2 + r.c_sigma_r ** 2)) if np.isfinite(r.t_sigma_r) else 100.0, miss_m=float(r.miss_distance)))
    return pd.DataFrame(ev)

LEAD_S = 150.0   # start of the avoidance window before TCA (s); episode ends at TCA (TMAX = LEAD_S)

def encounter_geometry(rel_r, rel_v):
    """Given relative state at TCA, return start state LEAD_S before TCA and the unit miss-vector direction
    (component of rel_r perpendicular to rel_v), which is the direction that increases the miss distance fastest."""
    r = np.array(rel_r, float); v = np.array(rel_v, float); vn = np.linalg.norm(v)
    vhat = v / vn if vn > 0 else np.array([1.0, 0, 0])
    r_perp = r - np.dot(r, vhat) * vhat
    if np.linalg.norm(r_perp) < 1e-3:                        # head-on: pick any perpendicular
        r_perp = np.cross(vhat, [0, 0, 1.0]);
        if np.linalg.norm(r_perp) < 1e-6: r_perp = np.cross(vhat, [0, 1.0, 0])
    dir0 = r_perp / np.linalg.norm(r_perp)
    r_start = r - v * LEAD_S
    return r_start, v, dir0

def simulate(rel_r, rel_v, policy, sigma_m):
    """Relative motion of active w.r.t. passive over [TCA-LEAD_S, TCA-LEAD_S+TMAX]: r' = v, v' = a_thrust.
    policy(d_pred_miss, step, dir0, sigma) -> thrust vector (m/s²). Safety judged on the minimum distance over the pass."""
    r, v, dir0 = encounter_geometry(rel_r, rel_v)
    dv = 0.0; t_thrust = 0; min_d = np.inf
    for k in range(TMAX):
        # predicted miss if no further thrust (linear relative motion)
        vn2 = np.dot(v, v); tca = -np.dot(r, v) / vn2 if vn2 > 0 else 0.0
        d_pred = np.linalg.norm(r + v * max(tca, 0.0))
        a = policy(d_pred, k, dir0, sigma_m, np.linalg.norm(v))
        v = v + a * DT; r = r + v * DT
        min_d = min(min_d, np.linalg.norm(r))
        an = np.linalg.norm(a); dv += an * DT; t_thrust += int(an > 0)
    vn2 = np.dot(v, v); tca = -np.dot(r, v) / vn2 if vn2 > 0 else 0.0
    min_d = min(min_d, np.linalg.norm(r + v * max(tca, 0.0)))
    return dict(min_dist_m=min_d, final_dist_m=min_d, dv_m_s=dv, thrust_s=t_thrust, safe=bool(min_d >= SAFE_M),
                pc_post=pc_2d(min_d / 1000, HBR_M / 1000, max(sigma_m, 1.0) / 1000))

def pol_none(d, k, dir0, s, vn=None): return np.zeros(3)
def pol_max(d, k, dir0, s, vn=None): return dir0 * AMAX if d < SAFE_M else np.zeros(3)
def pol_pc(d, k, dir0, s, vn=None): return dir0 * AMAX if pc_2d(d / 1000, HBR_M / 1000, max(s, 1.0) / 1000) >= 1e-4 else np.zeros(3)
def pol_prop(d, k, dir0, s, vn=None): return dir0 * AMAX * float(np.clip((SAFE_M - d) / SAFE_M, 0, 1))

def train_ppo(events: pd.DataFrame):
    try:
        import gymnasium as gym
        from gymnasium import spaces
        from stable_baselines3 import PPO
    except Exception as ex:
        log.warning(f"PPO unavailable: {ex}"); return None
    class Env(gym.Env):
        def __init__(self, evs):
            self.evs = evs; self.observation_space = spaces.Box(-5, 5, (4,), np.float32); self.action_space = spaces.Box(-1, 1, (1,), np.float32)
        def reset(self, seed=None, options=None):
            super().reset(seed=seed); e = self.evs.iloc[self.np_random.integers(len(self.evs))]
            self.r, self.v, self.dir0 = encounter_geometry(e.rel_r_m, e.rel_v_m_s)
            self.k = 0; self.fuel = 0.0; return self._obs(), {}
        def _dpred(self):
            vn2 = np.dot(self.v, self.v); tca = -np.dot(self.r, self.v) / vn2 if vn2 > 0 else 0.0
            return np.linalg.norm(self.r + self.v * max(tca, 0.0))
        def _obs(self):
            return np.array([min(self._dpred() / SAFE_M, 5.0), min(np.linalg.norm(self.v) / 15000.0, 5.0), self.fuel / 50, self.k / TMAX], np.float32)
        def step(self, act):
            a = float(np.clip((act[0] + 1) / 2, 0, 1)) * AMAX; d0 = self._dpred()
            self.v += self.dir0 * a * DT; self.r += self.v * DT; self.fuel += a * DT; self.k += 1
            d = self._dpred(); rew = (min(d, 1.2 * SAFE_M) - min(d0, 1.2 * SAFE_M)) / SAFE_M - 0.004 * a / AMAX
            done = self.k >= TMAX
            if done: rew += 5.0 if d >= SAFE_M else -5.0 - 3.0 * (SAFE_M - d) / SAFE_M
            return self._obs(), rew, done, False, {"safe": d >= SAFE_M, "total_fuel": self.fuel}
    env = Env(events); model = PPO("MlpPolicy", env, verbose=0, seed=C.SEED, n_steps=1024, batch_size=256, learning_rate=3e-4, device="cpu")
    model.learn(total_timesteps=400_000)
    def pol(d, k, dir0, s, vn=None):
        obs = np.array([min(d / SAFE_M, 5.0), min((vn or 7500.0) / 15000.0, 5.0), 0.0, k / TMAX], np.float32)
        a, _ = model.predict(obs, deterministic=True); return dir0 * float(np.clip((a[0] + 1) / 2, 0, 1)) * AMAX
    return pol

def main(force=False):
    if stage_done("S7_t7_avoidance") and not force: log.info("done"); return
    set_status("S7_t7_avoidance", "running")
    try:
        with Timer() as T:
            ev = load_events(); log.info(f"events: {ev.source.value_counts().to_dict()}")
            # σ for screening events from T4 σ-age fit (fallback 500 m)
            try:
                uq = pd.read_parquet(C.RESULTS / "t4_uq_cases.parquet"); s_med = float(np.median(np.sqrt(uq.sigma_age_r ** 2 + uq.sigma_age_t ** 2 + uq.sigma_age_n ** 2)) * 1000)
            except Exception: s_med = 500.0
            ev.loc[ev.source == "screening", "sigma_m"] = s_med
            pols = {"No-maneuver": pol_none, "Physics-max": pol_max, "Physics-Pc": pol_pc, "Proportional": pol_prop}
            rng = np.random.default_rng(C.SEED); idx = rng.permutation(len(ev)); tr, te = ev.iloc[idx[: len(ev) // 2]], ev.iloc[idx[len(ev) // 2:]]
            ppo = train_ppo(tr) if len(tr) >= 5 else None
            if ppo is not None: pols["PPO"] = ppo
            rows = []
            for name, pol in pols.items():
                for e in te.itertuples():
                    res = simulate(e.rel_r_m, e.rel_v_m_s, pol, e.sigma_m); rows.append(dict(strategy=name, source=e.source, event_id=e.event_id, miss0_m=e.miss_m, **res))
            ep = pd.DataFrame(rows); ep.to_csv(C.RESULTS / "t7_avoidance_episodes.csv", index=False)
            met = ep.groupby(["strategy", "source"]).agg(n=("safe", "size"), safety_rate=("safe", "mean"), dv_mean_m_s=("dv_m_s", "mean"), dv_max_m_s=("dv_m_s", "max"),
                                                       thrust_s_mean=("thrust_s", "mean"), pc_post_median=("pc_post", "median"), final_dist_med_m=("final_dist_m", "median")).reset_index()
            allm = ep.groupby("strategy").agg(n=("safe", "size"), safety_rate=("safe", "mean"), dv_mean_m_s=("dv_m_s", "mean"), dv_max_m_s=("dv_m_s", "max"),
                                             thrust_s_mean=("thrust_s", "mean"), pc_post_median=("pc_post", "median"), final_dist_med_m=("final_dist_m", "median")).reset_index(); allm["source"] = "ALL"
            met = pd.concat([met, allm], ignore_index=True); met.insert(0, "task", "T7"); save_df(met, "t7_avoidance_metrics.csv")
            log.info("\n" + met.round(3).to_string(index=False))
        set_status("S7_t7_avoidance", "done", elapsed_s=round(T.elapsed), n_events=int(len(te)))
    except Exception as ex:
        log.exception("T7 failed"); set_status("S7_t7_avoidance", "failed", message=str(ex)); raise

if __name__ == "__main__":
    main("--force" in sys.argv)
