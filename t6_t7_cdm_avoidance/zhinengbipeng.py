#!/usr/bin/env python3
"""
智能避碰 - 固定方向 + RL 控制大小
潜力系数 100000，燃料权重 0.000101
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


class TrainingCallback(BaseCallback):
    def __init__(self, verbose=1):
        super().__init__(verbose)
        self.episode_count = 0
        self.safe_list = []
        self.fuel_list = []

    def _on_step(self):
        if self.locals.get('dones') is not None:
            for i, done in enumerate(self.locals['dones']):
                if done:
                    info = self.locals['infos'][i]
                    self.episode_count += 1
                    self.safe_list.append(info.get('safe', False))
                    self.fuel_list.append(info.get('total_fuel', 0))
                    if self.episode_count % 50 == 0 and len(self.safe_list) > 0:
                        recent = min(20, len(self.safe_list))
                        avg_safe = np.mean(self.safe_list[-recent:])
                        avg_fuel = np.mean(self.fuel_list[-recent:])
                        print(f"\n[训练] Ep {self.episode_count} | "
                              f"近{recent}次安全率: {avg_safe:.2f} | "
                              f"平均燃料: {avg_fuel:.4f} m/s")
        return True


class FixedDirectionAvoidanceEnv(gym.Env):
    def __init__(self, df, max_events=None):
        super().__init__()

        self.max_thrust_per_step = 0.3
        self.dt = 1.0
        self.max_steps = 300

        self.events = []
        for _, row in df.iterrows():
            rel_r = np.array([row['relative_position_r'],
                              row['relative_position_t'],
                              row['relative_position_n']]) / 1000.0
            initial_dist_m = np.linalg.norm(rel_r) * 1000.0
            if initial_dist_m < 2000.0:
                self.events.append({
                    'initial_dist_m': initial_dist_m,
                    'event_id': row['event_id']
                })

        if max_events and len(self.events) > max_events:
            self.events = self.events[:max_events]

        print(f"✅ 训练事件: {len(self.events)} 个 (初始距离<2km)")

        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        event = np.random.choice(self.events)
        self.initial_dist = event['initial_dist_m']
        self.event_id = event['event_id']
        self.current_dist = self.initial_dist
        self.sat_speed = 0.0
        self.total_fuel = 0.0
        self.step_count = 0
        self.thrust_steps = 0
        self.done = False
        self.prev_alpha = 0.0
        return self._get_obs(), {}

    def _get_obs(self):
        dist_n = np.clip(self.current_dist / 2000.0, -1, 1)
        speed_n = np.clip(self.sat_speed / 10.0, -1, 1)
        fuel_n = np.clip(self.total_fuel / 5.0, 0, 1)
        time_n = np.clip(self.step_count / self.max_steps, 0, 1)
        return np.array([dist_n, speed_n, fuel_n, time_n], dtype=np.float32)

    def _potential(self, distance):
        return -1.0 / max(distance, 1.0)

    def step(self, action):
        self.step_count += 1

        alpha = float(np.clip(action[0], 0.0, 1.0))
        thrust_magnitude = alpha * self.max_thrust_per_step

        if thrust_magnitude > 1e-6:
            self.thrust_steps += 1

        old_potential = self._potential(self.current_dist)

        self.sat_speed += thrust_magnitude
        self.current_dist += self.sat_speed * self.dt
        self.total_fuel += thrust_magnitude

        new_potential = self._potential(self.current_dist)

        # ========== 奖励函数（潜力100000，燃料0.000101） ==========
        reward = (new_potential - old_potential) * 100000.0
        reward -= abs(alpha - self.prev_alpha) * 0.3
        reward -= 0.000101 * thrust_magnitude
        self.prev_alpha = alpha

        if self.step_count >= self.max_steps:
            self.done = True

        info = {
            'safe': self.current_dist >= 2000.0,
            'total_fuel': self.total_fuel,
            'distance': self.current_dist,
            'steps': self.thrust_steps,
            'total_steps': self.step_count,
        }
        return self._get_obs(), reward, self.done, False, info


def train_agent():
    print("=" * 60)
    print("智能避碰 - 潜力100000，燃料0.000101")
    print("=" * 60)

    df = pd.read_csv('cdm_data.csv')
    env = DummyVecEnv([lambda: FixedDirectionAvoidanceEnv(df, max_events=500)])
    callback = TrainingCallback()

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=1e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=[64, 64]),
    )

    print("\n🚀 训练 (50000 步)...")
    model.learn(total_timesteps=50000, callback=callback)
    model.save("ppo_pot100000_fuel0000101")
    print("✅ 模型已保存")
    return model, callback


def test_and_plot(model, df):
    print("\n" + "=" * 60)
    print("测试 + 绘图")
    print("=" * 60)

    env = FixedDirectionAvoidanceEnv(df, max_events=None)
    results = []
    n_test = min(10, len(env.events))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for i in range(n_test):
        obs, _ = env.reset()
        initial_dist = env.current_dist
        event_id = env.event_id

        dist_history = [initial_dist]
        fuel_history = [0.0]

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, info = env.step(action)
            dist_history.append(env.current_dist)
            fuel_history.append(env.total_fuel)
            if done:
                break

        axes[0, 0].plot(dist_history, linewidth=1.5, label=f'event {event_id}')
        axes[0, 1].plot(fuel_history, linewidth=1.5, label=f'event {event_id}')

        results.append({
            'event_id': event_id,
            'initial_dist_m': initial_dist,
            'final_dist_m': env.current_dist,
            'fuel_m_s': env.total_fuel,
            'thrust_steps': info['steps'],
            'safe': info['safe'],
        })

    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Distance (m)')
    axes[0, 0].set_title('Distance Change during Avoidance')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('Fuel (m/s)')
    axes[0, 1].set_title('Fuel Consumption during Avoidance')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend(fontsize=8)

    event_ids = [str(r['event_id']) for r in results]
    final_dists = [r['final_dist_m'] for r in results]
    axes[1, 0].bar(event_ids, final_dists, color='steelblue')
    axes[1, 0].set_xlabel('Event ID')
    axes[1, 0].set_ylabel('Final Distance (m)')
    axes[1, 0].set_title('Final Distance by Event')
    axes[1, 0].grid(axis='y', alpha=0.3)
    axes[1, 0].tick_params(axis='x', rotation=45)

    fuels = [r['fuel_m_s'] for r in results]
    axes[1, 1].bar(event_ids, fuels, color='coral')
    axes[1, 1].set_xlabel('Event ID')
    axes[1, 1].set_ylabel('Fuel (m/s)')
    axes[1, 1].set_title('Fuel Consumption by Event')
    axes[1, 1].grid(axis='y', alpha=0.3)
    axes[1, 1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig('avoidance_results.png', dpi=300)
    print("\n💾 图表已保存: avoidance_results.png")

    df_res = pd.DataFrame(results)
    n_safe = df_res['safe'].sum()

    print(f"\n📊 测试结果 ({len(df_res)} 个事件):")
    print(f"  安全率: {n_safe}/{len(df_res)} ({n_safe/len(df_res)*100:.1f}%)")
    print(f"  平均燃料: {df_res['fuel_m_s'].mean():.4f} m/s")
    print(f"  平均最终距离: {df_res['final_dist_m'].mean():.0f} m")
    print(f"  平均机动步数: {df_res['thrust_steps'].mean():.1f}")

    print("\n前10个事件详情:")
    for _, row in df_res.head(10).iterrows():
        status = "✅" if row['safe'] else "❌"
        print(f"  {status} event={row['event_id']}: "
              f"{row['initial_dist_m']:.0f}m → {row['final_dist_m']:.0f}m, "
              f"燃料={row['fuel_m_s']:.3f} m/s, 机动步数={row['thrust_steps']}")

    return df_res


if __name__ == "__main__":
    df = pd.read_csv('cdm_data.csv')
    model, callback = train_agent()
    test_and_plot(model, df)