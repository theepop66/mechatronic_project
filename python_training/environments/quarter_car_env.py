import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math

class QuarterCarEnv(gym.Env):
    def __init__(self):
        super(QuarterCarEnv, self).__init__()

        # พารามิเตอร์มาตรฐานของรถยนต์นั่งส่วนบุคคล (Quarter Car)
        self.ms  = 320.0      # มวลตัวถัง (Sprung mass) [kg]
        self.mus = 40.0       # มวลล้อ (Unsprung mass) [kg]
        self.ks  = 18000.0    # ความแข็งสปริง [N/m]
        self.cs  = 1000.0     # ความหน่วงทางกล [N.s/m]
        self.kt  = 200000.0   # ความแข็งยาง [N/m]

        self.Ke    = 50.0     # ค่าคงที่แรงดัน [V / (m/s)]
        self.Kt    = 50.0     # ค่าคงที่แรง [N/A]
        self.R_int = 2.0      # ความต้านทานภายในมอเตอร์ [Ohm]

        # Action Space (10 ระดับความต้านทาน)
        self.action_space = spaces.Discrete(10)
        self.resistance_levels = np.linspace(10000.0, 10.0, 10)

        # Observation Space: [Voltage, Relative Velocity, Sprung Acceleration]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
        )

        # ✅ น้ำหนัก Reward — เน้น comfort เป็นหลัก
        self.weight_comfort = 0.80
        self.weight_holding = 0.15
        self.weight_energy  = 0.05

        # ✅ ค่า normalization (ปรับให้แต่ละ component อยู่ในระดับ 0–1 ก่อนถ่วงน้ำหนัก)
        #    ค่าเหล่านี้คือค่าอ้างอิงที่ passive suspension ทั่วไปได้รับ
        #    — หาจากการรัน passive suspension 1 รอบแล้วดู RMS
        self._norm_comfort = 25.0     # (zs_ddot_rms)² ≈ 5² = 25  [m²/s⁴]
        self._norm_holding = 1e-4     # (zus−zr)_rms² ≈ 0.01² = 1e-4  [m²]
        self._norm_energy  = 50.0     # Power_rms ≈ 50 [W]

        self.dt   = 0.01
        self.time = 0.0
        self.state_phys = np.zeros(4)

        # ✅ เก็บค่าเร่งก่อนหน้าสำหรับ reward shaping
        self._prev_zs_ddot = 0.0

    def _get_road_profile(self, t):
        """
        ✅ ปรับ road profile ให้ครอบคลุมความถี่ ISO 8608 Road Class B
           และใส่ลูกระนาดแบบ half-sine ที่สมจริงขึ้น
        """
        # ถนนคลื่นหลายความถี่ (สมจริงมากขึ้น)
        wave = (0.020 * math.sin(2 * math.pi * 1.5 * t)
              + 0.010 * math.sin(2 * math.pi * 5.0 * t)
              + 0.005 * math.sin(2 * math.pi * 10.0 * t))

        # ลูกระนาด half-sine ทุก ~4 วินาที
        bump = 0.0
        t_mod = t % 4.0
        if t_mod > 3.8:
            bump = 0.05 * math.sin(math.pi * (t_mod - 3.8) / 0.2)

        return wave + bump

    def step(self, action):
        R_ext = self.resistance_levels[action]

        zs, zs_dot, zus, zus_dot = self.state_phys
        zr = self._get_road_profile(self.time)

        # ฟิสิกส์ Regenerative Suspension
        rel_vel  = zs_dot - zus_dot
        Voltage  = self.Ke * rel_vel
        Current  = Voltage / (self.R_int + R_ext)
        F_regen  = self.Kt * Current
        Power    = (Voltage ** 2) / (self.R_int + R_ext)  # [W]

        # สมการการเคลื่อนที่
        zs_ddot  = (-self.ks * (zs - zus) - self.cs * rel_vel - F_regen) / self.ms
        zus_ddot = ( self.ks * (zs - zus) + self.cs * rel_vel + F_regen
                    - self.kt * (zus - zr)) / self.mus

        # Euler Integration
        zs_dot_new  = zs_dot  + zs_ddot  * self.dt
        zs_new      = zs      + zs_dot_new * self.dt
        zus_dot_new = zus_dot + zus_ddot  * self.dt
        zus_new     = zus     + zus_dot_new * self.dt

        self.state_phys = np.array([zs_new, zs_dot_new, zus_new, zus_dot_new])
        self.time += self.dt

        # ──────────────────────────────────────────────────────────
        # ✅ Reward แบบ Normalized
        #    หาร norm_* ก่อนถ่วงน้ำหนัก → แต่ละ term อยู่ในสเกลเดียวกัน
        # ──────────────────────────────────────────────────────────
        penalty_comfort_raw = zs_ddot ** 2
        penalty_holding_raw = (zus_new - zr) ** 2
        reward_energy_raw   = Power

        penalty_comfort_norm = penalty_comfort_raw / self._norm_comfort
        penalty_holding_norm = penalty_holding_raw / self._norm_holding
        reward_energy_norm   = reward_energy_raw   / self._norm_energy

        reward = (- self.weight_comfort * penalty_comfort_norm
                  - self.weight_holding * penalty_holding_norm
                  + self.weight_energy  * reward_energy_norm)

        # ✅ Reward Shaping: ให้ bonus เมื่อค่าเร่งลดลงจากขั้นก่อน
        #    ช่วยให้ agent เรียนรู้ทิศทางที่ถูกต้องได้เร็วขึ้น
        delta_accel = abs(self._prev_zs_ddot) - abs(zs_ddot)
        reward += 0.05 * np.clip(delta_accel / 5.0, -1.0, 1.0)
        self._prev_zs_ddot = zs_ddot

        # Observation
        obs = np.array([Voltage, rel_vel, zs_ddot], dtype=np.float32)

        terminated = self.time >= 10.0
        truncated  = False

        return obs, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.time = 0.0
        self.state_phys = np.zeros(4)
        self._prev_zs_ddot = 0.0
        obs = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        return obs, {}