#!/usr/bin/env python3
"""
保守碰撞概率计算（Patera 方法）
保守策略：
  - 保持基础定轨精度 σ_base = 20 m 不变
  - 将每颗卫星的 SGP4 传播误差 (2000 m) 纳入硬体半径：R_conservative = R + √2 × 2000 ≈ R + 2828 m
  - 同时将名义 miss distance 缩小 2828 m 作为最坏情况 miss
"""

import numpy as np
from scipy.special import erf

def compute_conservative_pc(
    pos1, vel1, pos2, vel2,
    hard_body_radius=8.0,
    sigma_base=20.0,
    sigma_each=2000.0          # 每颗卫星的 SGP4 传播误差 (1σ)
):
    r1 = np.asarray(pos1, dtype=float)
    v1 = np.asarray(vel1, dtype=float)
    r2 = np.asarray(pos2, dtype=float)
    v2 = np.asarray(vel2, dtype=float)

    r = r1 - r2
    v = v1 - v2
    v_norm = np.linalg.norm(v)
    if v_norm < 1e-12:
        raise ValueError("相对速度为零")

    # 相遇平面
    v_hat = v / v_norm
    u = np.array([1.0, 0.0, 0.0]) if abs(v_hat[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(v_hat, u)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(v_hat, e1)

    tca = -np.dot(r, v) / (v_norm ** 2)
    r_ca = r + v * tca
    miss_nom = np.linalg.norm(r_ca)

    # 联合传播误差
    sigma_prop = np.sqrt(2.0) * sigma_each   # ≈ 2828 m

    # 保守硬体半径：原始半径 + 传播误差
    R_conservative = hard_body_radius + sigma_prop

    # 保守 miss distance：名义 miss 减去传播误差（最坏情况）
    miss_cons = max(1.0, miss_nom - sigma_prop)

    # 保持方向不变，缩放 miss 向量到保守值
    scale = miss_cons / miss_nom
    r_ca_cons = r_ca * scale
    mu_x = np.dot(r_ca_cons, e1)
    mu_y = np.dot(r_ca_cons, e2)

    # 协方差：只用基础定轨精度，不膨胀
    sigma = np.sqrt(2.0) * sigma_base

    # Patera 积分（圆心在原点，半径为 R_conservative）
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
    return Pc, logPc, miss_nom, miss_cons, R_conservative


if __name__ == "__main__":
    # STARLINK-3890 vs STARLINK-3896 (2022-05-23T01:42:01.000224 UTC)
    pos1 = [5066018.46, -2429797.818, 3694201.796]
    vel1 = [5054.380787, 3696.820198, -4483.586269]
    pos2 = [5069938.758, -2427133.779, 3690959.337]
    vel2 = [5049.052313, 3699.029331, -4487.028199]

    R = 8.0              # 原始联合硬体半径 (米)
    sigma_base = 20.0    # 基础定轨精度 (米)
    sigma_each = 2000.0  # 每颗卫星 SGP4 误差 (米)

    Pc, logPc, miss_nom, miss_cons, R_cons = compute_conservative_pc(
        pos1, vel1, pos2, vel2, R, sigma_base, sigma_each
    )

    print("保守碰撞概率计算结果")
    print("-" * 50)
    print(f"原始联合硬体半径:       {R} m")
    print(f"每颗卫星传播误差 (1σ):  {sigma_each} m")
    print(f"联合传播误差:            {np.sqrt(2)*sigma_each:.0f} m")
    print(f"保守联合硬体半径:        {R_cons:.0f} m")
    print(f"基础定轨精度 σ_base:     {sigma_base} m")
    print(f"联合基础标准差:          {np.sqrt(2)*sigma_base:.1f} m")
    print(f"名义 miss distance:       {miss_nom:.0f} m")
    print(f"保守 miss distance:       {miss_cons:.0f} m")
    print(f"保守碰撞概率 Pc:         {Pc:.6e}")
    print(f"log10(Pc):                {logPc:.2f}")