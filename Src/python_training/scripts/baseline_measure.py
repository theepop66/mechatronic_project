import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from environments.quarter_car_env import QuarterCarEnv

env = QuarterCarEnv()
MAX_STEPS = 1000
N_PROFILES = 5
N_EPISODES = 10  # run multiple episodes per profile for stable estimates

prof_names = [
    "two_halfsine_bumps",
    "smooth_wavy_road",
    "rough_asphalt",
    "speed_breaker_dip",
    "coffee_run",
]

print("=" * 72)
print("  Per-profile baseline measurement - Ke=Kt=110")
print("  Low damping (action 0) -> comfort & holding norms")
print("  High damping (action 9) -> energy norm")
print("=" * 72)

profile_norms = {}

for profile_idx in range(N_PROFILES):
    comfort_vals = []
    holding_vals = []
    energy_vals  = []
    ep_rewards_low  = []
    ep_rewards_high = []

    for ep in range(N_EPISODES):
        for action, regime in [(0, "low"), (9, "high")]:
            state, _ = env.reset(options={"profile_idx": profile_idx})
            ep_reward = 0.0
            for t in range(MAX_STEPS):
                zs, zs_dot, zus, zus_dot = env.state_phys
                rel_vel = zs_dot - zus_dot
                zr = env._get_road_profile(env.time)
                Voltage = env.Ke * rel_vel
                Current = Voltage / (env.R_int + env.resistance_levels[action])
                F_regen = env.Kt * Current
                Power = Voltage ** 2 / (env.R_int + env.resistance_levels[action])
                zs_ddot = (-env.ks * (zs - zus) - env.cs * rel_vel - F_regen) / env.ms

                obs, reward, term, trunc, _ = env.step(action)
                ep_reward += reward

                if regime == "low":
                    comfort_vals.append(zs_ddot ** 2)
                    holding_vals.append((zus - zr) ** 2)
                else:
                    energy_vals.append(Power)

                if term or trunc:
                    break

            if regime == "low":
                ep_rewards_low.append(ep_reward)
            else:
                ep_rewards_high.append(ep_reward)

    norm_c = float(np.mean(comfort_vals))
    norm_h = float(np.mean(holding_vals))
    norm_e = float(np.mean(energy_vals))
    low_avg  = float(np.mean(ep_rewards_low))
    high_avg = float(np.mean(ep_rewards_high))

    profile_norms[profile_idx] = {
        "norm_comfort": norm_c,
        "norm_holding": norm_h,
        "norm_energy": norm_e,
        "low_reward": low_avg,
        "high_reward": high_avg,
    }

    print(f"\n  Profile {profile_idx}: {prof_names[profile_idx]}")
    print(f"    _norm_comfort = {norm_c:.6f}")
    print(f"    _norm_holding = {norm_h:.10f}")
    print(f"    _norm_energy  = {norm_e:.4f}")
    print(f"    LOW damping reward (raw)  = {low_avg:.2f}")
    print(f"    HIGH damping reward (raw) = {high_avg:.2f}")

# Compute offsets to center LOW-damping at ~0
print("\n" + "=" * 72)
print("  Offsets (centers LOW-damping baseline at ~0 per profile)")
print("=" * 72)

for pid in range(N_PROFILES):
    p = profile_norms[pid]
    # Reward formula: -0.80*(comfort/norm_c) - 0.15*(holding/norm_h) + 0.05*(energy/norm_e)
    # At baseline (low damping), each normalized term ~ 1.0
    # offset = 0.80 + 0.15 - 0.05 = 0.90 (theoretical)
    # Actual: offset = -(low_reward) to bring it to 0
    offset = -p["low_reward"] / MAX_STEPS
    print(f"  Profile {pid} ({prof_names[pid]:20s}): "
          f"offset = {offset:.6f}  (step offset to center low at ~0)")

# Compute per-profile step offsets using weight formula
print("\n" + "=" * 72)
print("  Final per-profile constants - copy into env")
print("=" * 72)
print()
print("_PROFILE_NORMS = {")
for pid in range(N_PROFILES):
    p = profile_norms[pid]
    offset = -p["low_reward"] / MAX_STEPS
    print(f"    {pid}: {{  # {prof_names[pid]}")
    print(f"        \"norm_comfort\": {p['norm_comfort']:.6f},")
    print(f"        \"norm_holding\": {p['norm_holding']:.10f},")
    print(f"        \"norm_energy\":  {p['norm_energy']:.4f},")
    print(f"        \"offset\":       {offset:.6f},")
    print(f"    }},")
print("}")
