import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
import os
import csv

class QuarterCarEnv(gym.Env):
    # CSV road profiles — loaded once at class level (lazy)
    _ROAD_PROFILES = None        # shape (N_100Hz, 5) after downsampling
    _ROAD_T_100HZ  = None        # time array at 100 Hz
    _ROAD_CSV_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'road_profiles.csv')

    PROFILE_NAMES = [
        "two_halfsine_bumps",     # profile_1 — isolated obstacles
        "smooth_wavy_road",       # profile_2 — low-frequency undulations
        "rough_asphalt",          # profile_3 — high-frequency noise
        "speed_breaker_dip",      # profile_4 — non-symmetric disturbance
        "coffee_run",             # profile_5 — mixed sines + noise + pothole
    ]

    def __init__(self, road_source="csv"):
        super(QuarterCarEnv, self).__init__()

        if road_source not in ("csv", "synthetic"):
            raise ValueError(f"road_source must be 'csv' or 'synthetic', got '{road_source}'")
        self._road_source = road_source

        # Load CSV data on first instantiation if needed
        if self._road_source == "csv" and QuarterCarEnv._ROAD_PROFILES is None:
            QuarterCarEnv._load_csv_data()

        # พารามิเตอร์มาตรฐานของรถยนต์นั่งส่วนบุคคล (Quarter Car)
        self.ms  = 320.0      # มวลตัวถัง (Sprung mass) [kg]
        self.mus = 40.0       # มวลล้อ (Unsprung mass) [kg]
        self.ks  = 18000.0    # ความแข็งสปริง [N/m]
        self.cs  = 1000.0     # ความหน่วงทางกล [N.s/m]
        self.kt  = 200000.0   # ความแข็งยาง [N/m]

        # Increased from 50 to 110 based on baseline analysis (see scripts/baseline_measure.py):
        # At Ke=50, regen damping max was 208 N.s/m vs passive cs=1000 N.s/m (only 17% control).
        # At Ke=110, regen damping max = 1008 N.s/m, matching the passive damper (50% control).
        self.Ke    = 110.0    # ค่าคงที่แรงดัน [V / (m/s)]
        self.Kt    = 110.0    # ค่าคงที่แรง [N/A]
        self.R_int = 2.0      # ความต้านทานภายในมอเตอร์ [Ohm]

        # Action Space (10 ระดับความต้านทาน)
        self.action_space = spaces.Discrete(10)
        self.resistance_levels = np.linspace(10000.0, 10.0, 10)

        # Observation Space: [Voltage, Sprung Acceleration]
        # rel_vel removed because Voltage = Ke * rel_vel (perfectly collinear in old 3-D obs).
        # 2-D reduces network size, speeds inference, and eliminates redundant channels.
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32
        )

        # ✅ น้ำหนัก Reward — เน้น comfort เป็นหลัก
        self.weight_comfort = 0.80
        self.weight_holding = 0.15
        self.weight_energy  = 0.05

        # Per-profile normalization constants + offsets
        # Calibrated from scripts/baseline_measure.py (Ke=Kt=110, 10 ep each profile).
        self._set_profile_norms(profile_idx=0)  # will be updated in reset()

        self.dt   = 0.01
        self.time = 0.0
        self.state_phys = np.zeros(4)

        # ✅ เก็บค่าเร่งก่อนหน้าสำหรับ reward shaping
        self._prev_zs_ddot = 0.0

        # CSV-specific: current profile index and time offset
        self._current_profile = 0
        self._time_offset = 0.0

    # ─────────────────────────────────────────
    # Per-profile normalization constants (class-level, read-only)
    # ─────────────────────────────────────────
    _PROFILE_NORMS = {
        0: {  # two_halfsine_bumps
            "norm_comfort": 0.462433,
            "norm_holding": 0.0000012479,
            "norm_energy":  3.0104,
            "offset":       0.145800,
        },
        1: {  # smooth_wavy_road
            "norm_comfort": 3.263138,
            "norm_holding": 0.0000084610,
            "norm_energy":  46.0383,
            "offset":       1.017380,
        },
        2: {  # rough_asphalt
            "norm_comfort": 0.076717,
            "norm_holding": 0.0000010035,
            "norm_energy":  2.9002,
            "offset":       0.032071,
        },
        3: {  # speed_breaker_dip
            "norm_comfort": 0.911524,
            "norm_holding": 0.0000100840,
            "norm_energy":  32.9249,
            "offset":       0.363973,
        },
        4: {  # coffee_run
            "norm_comfort": 1.372144,
            "norm_holding": 0.0000100124,
            "norm_energy":  39.3079,
            "offset":       0.492796,
        },
    }

    def _set_profile_norms(self, profile_idx):
        """Load normalization constants for the given profile index."""
        p = self._PROFILE_NORMS[profile_idx % len(self._PROFILE_NORMS)]
        self._norm_comfort = p["norm_comfort"]
        self._norm_holding = p["norm_holding"]
        self._norm_energy  = p["norm_energy"]
        self._reward_offset = p["offset"]

    # ─────────────────────────────────────────
    # CSV loading (class-level, cached)
    # ─────────────────────────────────────────
    @classmethod
    def _load_csv_data(cls):
        """Load road_profiles.csv, downsample from 200 Hz → 100 Hz, store as class arrays."""
        t_csv = []
        profiles_raw = []   # list of 5 lists
        try:
            with open(cls._ROAD_CSV_PATH, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    t_csv.append(float(row['t']))
                    cols = []
                    for i in range(1, 6):
                        val = float(row[f'profile_{i}']) if row[f'profile_{i}'].strip() else 0.0
                        cols.append(val)
                    profiles_raw.append(cols)
        except Exception as e:
            raise RuntimeError(f"Failed to load road profiles from {cls._ROAD_CSV_PATH}: {e}")

        t_csv = np.array(t_csv)                     # (4001,)  at 200 Hz, 0–20s
        profiles_raw = np.array(profiles_raw)       # (4001, 5)

        # Downsample to 100 Hz matching env dt=0.01s
        t_100 = np.arange(0.0, t_csv[-1] + 0.001, 0.01)   # (2001,) 0–20s
        profiles_100 = np.column_stack([
            np.interp(t_100, t_csv, profiles_raw[:, i])
            for i in range(5)
        ])                                               # (2001, 5)

        cls._ROAD_T_100HZ = t_100
        cls._ROAD_PROFILES = profiles_100

        print(f"[QuarterCarEnv] Loaded {cls._ROAD_CSV_PATH}: "
              f"{len(t_csv)} rows @ 200 Hz -> {len(t_100)} rows @ 100 Hz, "
              f"5 profiles: {', '.join(cls.PROFILE_NAMES)}")

    @property
    def current_profile_name(self):
        if self._road_source != "csv":
            return "synthetic"
        return self.PROFILE_NAMES[self._current_profile]

    # ─────────────────────────────────────────
    # Profile selection
    # ─────────────────────────────────────────
    def _pick_road_profile(self, profile_idx=None):
        """Pick a single profile (random or specified) for the episode."""
        n_profiles = self._ROAD_PROFILES.shape[1] if self._road_source == "csv" else 0
        if profile_idx is not None and self._road_source == "csv":
            idx = profile_idx % n_profiles
        else:
            idx = np.random.randint(0, n_profiles) if n_profiles > 0 else 0
        self._current_profile = idx
        self._set_profile_norms(idx)
        # Random 0–10s shift within the 20s CSV data
        self._time_offset = np.random.uniform(0.0, 10.0)

    # ─────────────────────────────────────────
    # Road profile
    # ─────────────────────────────────────────
    def _get_road_profile(self, t):
        if self._road_source == "csv" and self._ROAD_PROFILES is not None:
            effective_t = (t + self._time_offset) % 20.0
            idx = int(round(effective_t / self.dt))
            idx = np.clip(idx, 0, len(self._ROAD_T_100HZ) - 1)
            return float(self._ROAD_PROFILES[idx, self._current_profile])

        # ── Synthetic fallback (matching original behaviour) ──
        wave = (0.020 * math.sin(2 * math.pi * 1.5 * t)
              + 0.010 * math.sin(2 * math.pi * 5.0 * t)
              + 0.005 * math.sin(2 * math.pi * 10.0 * t))
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
                  + self.weight_energy  * reward_energy_norm
                  + self._reward_offset)

        # ✅ Reward Shaping: ให้ bonus เมื่อค่าเร่งลดลงจากขั้นก่อน
        #    ช่วยให้ agent เรียนรู้ทิศทางที่ถูกต้องได้เร็วขึ้น
        delta_accel = abs(self._prev_zs_ddot) - abs(zs_ddot)
        reward += 0.05 * np.clip(delta_accel / 5.0, -1.0, 1.0)
        self._prev_zs_ddot = zs_ddot

        # Observation: [Voltage, Sprung Acceleration]
        #   rel_vel is removed (redundant — Voltage = Ke * rel_vel)
        obs = np.array([Voltage, zs_ddot], dtype=np.float32)

        terminated = self.time >= 10.0
        truncated  = False

        return obs, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.time = 0.0
        self.state_phys = np.zeros(4)
        self._prev_zs_ddot = 0.0

        # Accept optional profile_idx from caller (e.g. train.py cycling profiles)
        profile_idx = options.get("profile_idx", None) if options else None
        if self._road_source == "csv":
            self._pick_road_profile(profile_idx=profile_idx)

        obs = np.array([0.0, 0.0], dtype=np.float32)
        return obs, {}