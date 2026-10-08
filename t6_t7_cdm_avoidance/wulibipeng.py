#!/usr/bin/env python3
"""
物理避碰 - 使用 event_id=1568 真实数据
推力加速度 0.3 m/s²，每10步评估一次
"""

import numpy as np
from scipy.special import erf


def compute_smooth_pc(pos1_m, vel1_m, pos2_m, vel2_m,
                       hard_body_radius_m=8.0,
                       sigma_cov=2000.0,
                       sigma_each=2000.0):
    """平滑碰撞概率计算"""
    r1 = np.asarray(pos1_m, dtype=float)
    v1 = np.asarray(vel1_m, dtype=float)
    r2 = np.asarray(pos2_m, dtype=float)
    v2 = np.asarray(vel2_m, dtype=float)

    r = r1 - r2
    v = v1 - v2
    v_norm = np.linalg.norm(v)
    if v_norm < 1e-12:
        return 0.0, -30.0, 0.0

    v_hat = v / v_norm
    u = np.array([1.0, 0.0, 0.0]) if abs(v_hat[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(v_hat, u)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(v_hat, e1)

    tca = -np.dot(r, v) / (v_norm ** 2)
    r_ca = r + v * tca
    miss_nom = np.linalg.norm(r_ca)

    mu_x = np.dot(r_ca, e1)
    mu_y = np.dot(r_ca, e2)

    sigma_prop = np.sqrt(2.0) * sigma_each
    R_conservative = hard_body_radius_m + sigma_prop
    sigma = sigma_cov
    R = R_conservative
    n = 200

    x_min = -R - 5.0 * sigma
    x_max =  R + 5.0 * sigma
    dx = (x_max - x_min) / n
    total = 0.0

    for i in range(n + 1):
        x = x_min + i * dx
        y_sq = R*R - x*x
        if y_sq <= 0:
            continue
        y_bound = np.sqrt(y_sq)
        term_high = erf(( y_bound - mu_y) / (np.sqrt(2) * sigma))
        term_low  = erf((-y_bound - mu_y) / (np.sqrt(2) * sigma))
        fy = term_high - term_low
        fx = np.exp(-((x - mu_x) ** 2) / (2 * sigma * sigma))
        if i == 0 or i == n:
            w = 1.0
        elif i % 2 == 1:
            w = 4.0
        else:
            w = 2.0
        total += w * fy * fx

    Pc = total * dx / 3.0 / (2 * np.pi * sigma * sigma)
    logPc = np.log10(max(Pc, 1e-30))
    return Pc, logPc, miss_nom


def physics_guided_avoidance(pos1_m, vel1_m, pos2_m, vel2_m,
                              max_thrust_accel=0.3,
                              dt=1.0,
                              safe_logPc=-6.0,
                              safe_miss_m=5000.0,
                              max_steps=10000,
                              hard_body_radius_m=8.0,
                              eval_interval=10):
    """物理引导避碰，每10步评估一次"""
    r1 = np.asarray(pos1_m, dtype=float).copy()
    v1 = np.asarray(vel1_m, dtype=float).copy()
    r2 = np.asarray(pos2_m, dtype=float).copy()
    v2 = np.asarray(vel2_m, dtype=float).copy()

    _, logPc, miss_nom = compute_smooth_pc(
        r1, v1, r2, v2, hard_body_radius_m)

    r_rel = r1 - r2
    v_rel = v1 - v2
    v2_mag = np.dot(v_rel, v_rel)
    if v2_mag > 1e-20:
        tca = -np.dot(r_rel, v_rel) / v2_mag
        r_ca = r_rel + v_rel * tca
        thrust_dir = r_ca / np.linalg.norm(r_ca)
    else:
        thrust_dir = np.array([1.0, 0.0, 0.0])

    total_fuel = 0.0
    step = 0
    log_history = [(0, logPc, miss_nom, 0.0)]

    while (logPc > safe_logPc or miss_nom < safe_miss_m) and step < max_steps:
        step += 1

        accel = thrust_dir * max_thrust_accel
        v1 += accel * dt
        r1 += v1 * dt
        total_fuel += max_thrust_accel * dt

        if step % eval_interval == 0:
            _, logPc, miss_nom = compute_smooth_pc(
                r1, v1, r2, v2, hard_body_radius_m)
            log_history.append((step, logPc, miss_nom, total_fuel))

    if step % eval_interval != 0:
        _, logPc, miss_nom = compute_smooth_pc(
            r1, v1, r2, v2, hard_body_radius_m)
        log_history.append((step, logPc, miss_nom, total_fuel))

    success = logPc <= safe_logPc and miss_nom >= safe_miss_m
    total_time = step * dt

    return success, total_fuel, total_time, log_history


if __name__ == "__main__":
    pos1 = np.array([0.0, 0.0, 0.0])
    vel1 = np.array([0.0, 0.0, 0.0])

    rel_r = np.array([-5.8, 21.2, 53.5])
    rel_v = np.array([39.6, -13028.7, 4815.7])
    pos2 = -rel_r
    vel2 = -rel_v

    print("=" * 60)
    print("物理避碰 - event_id=1568")
    print("=" * 60)
    print(f"初始相对距离: {np.linalg.norm(rel_r):.1f} m")
    print(f"相对速度: {np.linalg.norm(rel_v):.1f} m/s")

    success, fuel, time_s, log = physics_guided_avoidance(
        pos1, vel1, pos2, vel2,
        max_thrust_accel=0.3,
        dt=1.0,
        safe_logPc=-6.0,
        safe_miss_m=5000.0,
        max_steps=10000,
        eval_interval=10
    )

    print(f"\n结果: {'✅ 成功' if success else '❌ 失败'}")
    print(f"总燃料: {fuel:.2f} m/s")
    print(f"总时间: {time_s:.0f} s")

    print("\n概率变化过程:")
    for step, logPc, miss, fuel_cum in log:
        print(f"  步 {step:>6}: logPc={logPc:8.4f}, miss={miss:8.0f} m, "
              f"燃料={fuel_cum:8.2f} m/s")