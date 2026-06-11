import numpy as np
import matplotlib.pyplot as plt
import os
from collections import deque

from environments.quarter_car_env import QuarterCarEnv
from agent.dqn_agent import DQNAgent

# ─────────────────────────────────────────
# 1. เตรียม Environment และ Agent
# ─────────────────────────────────────────
env = QuarterCarEnv(road_source="csv")
state_size  = int(env.observation_space.shape[0])   # 2
action_size = int(env.action_space.n)               # 10
agent = DQNAgent(state_size, action_size)

# ─────────────────────────────────────────
# 2. ตั้งค่าการเทรน
# ─────────────────────────────────────────
EPISODES        = 5000       # 5000 ep (0.995^920 ~ 0.01, reaches min by ep ~920)
MAX_STEPS       = 1000       # 10 วินาที x 100 Hz (1 profile/episode)
MOVING_AVG_WIN  = 20         # [OK]  หน้าต่าง moving average สำหรับวัดพัฒนาการ
EVAL_FREQ       = 50         # [OK]  ประเมินผล (ไม่สุ่ม) ทุก N episode
WARMUP_STEPS    = 2000       # [OK]  รอสะสม memory ก่อนเริ่ม replay
REPLAY_FREQ     = 2          # [OK]  replay ทุก 2 step

# ─── Curriculum Learning ──────────────────
# Stage: (end_ep_exclusive, allowed_profile_indices)
# Profiles: 0=two_halfsine_bumps, 1=smooth_wavy_road, 2=rough_asphalt,
#           3=speed_breaker_dip, 4=coffee_run
# Each stage = one profile only, 1000 episodes each.
# ε resets to 0.50 at each stage boundary, then decays via epsilon_decay=0.995.
# 0.50 * (0.995^1000) ≈ 0.0033 by end of stage -> plenty of exploitation.
CURRICULUM = [
    (1000,  [1]),              # Stage 1: smooth_wavy_road
    (2000,  [2]),              # Stage 2: rough_asphalt
    (3000,  [4]),              # Stage 3: coffee_run (mixed content)
    (4000,  [0]),              # Stage 4: two_halfsine_bumps (reactive obstacles)
    (5000,  [3]),              # Stage 5: speed_breaker_dip
]

model_dir  = "models"
os.makedirs(model_dir, exist_ok=True)
best_model_path = os.path.join(model_dir, "best_suspension_dqn_csv.weights.h5")
final_model_path = os.path.join(model_dir, "final_suspension_dqn_csv.weights.h5")
final_full_model_path = os.path.join(model_dir, "final_suspension_dqn_csv.keras")

reward_history     = []
profile_rewards    = {name: [] for name in env.PROFILE_NAMES}
best_moving_avg    = -np.inf
total_steps        = 0           # [OK]  นับ step สะสมทั้งหมด (สำหรับ warmup)
replay_step_counter = 0          # [OK]  นับ step สำหรับ replay frequency

def _get_curriculum(ep):
    """Return (allowed_profile_indices, stage_number) for a given episode."""
    for stage, (end_ep, profiles) in enumerate(CURRICULUM):
        if ep < end_ep:
            return profiles, stage
    return CURRICULUM[-1][1], len(CURRICULUM) - 1

print(">>  Starting AI training...")
print(f"   Episodes={EPISODES} | Curriculum: {len(CURRICULUM)} stages | Warmup={WARMUP_STEPS} steps | Eval every {EVAL_FREQ} ep\n")

for stage_idx, (end_ep, profs) in enumerate(CURRICULUM):
    names = [env.PROFILE_NAMES[i] for i in profs]
    print(f"     [Stage]  Stage {stage_idx+1}: ep 0-{end_ep} -> {names}")
print()

current_stage = -1

for e in range(EPISODES):
    allowed, stage = _get_curriculum(e)
    if stage != current_stage:
        current_stage = stage
        names = [env.PROFILE_NAMES[i] for i in allowed]

        # Reset epsilon to 0.50 for exploration on the new profile
        agent.epsilon = max(agent.epsilon, 0.50)

        # Flush old memories: keep only most recent 10k for new profile
        if len(agent.memory) > 10000:
            agent.memory = deque(
                list(agent.memory)[-10000:],
                maxlen=agent.memory.maxlen
            )

        print(f"\n     [Stage]  -> Stage {stage+1}: {names}  (eps reset to {agent.epsilon:.3f},"
              f" memory kept={len(agent.memory)})\n")

    profile_idx = allowed[e % len(allowed)]
    state, _ = env.reset(options={"profile_idx": profile_idx})
    state       = np.reshape(state, [1, state_size])
    total_reward = 0.0
    current_profile_rewards = {name: 0.0 for name in env.PROFILE_NAMES}

    for t in range(MAX_STEPS):
        action = agent.act(state)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done       = terminated or truncated
        next_state = np.reshape(next_state, [1, state_size])

        agent.remember(state, action, reward, next_state, done)
        state        = next_state
        total_reward += reward
        total_steps  += 1
        current_profile_rewards[env.current_profile_name] += reward

        # [OK]  replay ทุก REPLAY_FREQ step (ลด correlated updates)
        #    เริ่มได้หลัง warmup เท่านั้น เพื่อให้ memory มีความหลากหลายพอ
        if total_steps >= WARMUP_STEPS:
            replay_step_counter += 1
            if replay_step_counter % REPLAY_FREQ == 0:
                agent.replay()

        if done:
            break

    reward_history.append(total_reward)
    for name in env.PROFILE_NAMES:
        profile_rewards[name].append(current_profile_rewards[name])

    # ─── Per-episode epsilon decay ──────────
    agent.decay_epsilon()

    # ─── Moving Average ─────────────────────
    moving_avg = np.mean(reward_history[-MOVING_AVG_WIN:])

    # [OK]  บันทึกโมเดลที่ดีที่สุดตาม moving average (ไม่ใช่ episode เดี่ยว)
    if len(reward_history) >= MOVING_AVG_WIN and moving_avg > best_moving_avg:
        best_moving_avg = moving_avg
        agent.model.save_weights(best_model_path)

    # ─── Log ────────────────────────────────
    warmup_tag = "[WARMUP]" if total_steps < WARMUP_STEPS else "       "
    print(f"| Ep {e+1:03d}/{EPISODES} {warmup_tag} | "
          f"S{stage+1} | "
          f"Score: {total_reward:8.2f} | "
          f"MovAvg({MOVING_AVG_WIN}): {moving_avg:8.2f} | "
          f"eps: {agent.epsilon:.3f}")

    # ─── Evaluation Episode (ε=0) ───────────
    # [OK]  ทดสอบโมเดลแบบ greedy เพื่อดูประสิทธิภาพจริง (ไม่สุ่ม)
    if (e + 1) % EVAL_FREQ == 0:
        eps_backup    = agent.epsilon
        agent.epsilon = 0.0          # ปิดการสุ่มชั่วคราว
        eval_state, _ = env.reset()
        eval_state    = np.reshape(eval_state, [1, state_size])
        eval_reward   = 0.0

        for _ in range(MAX_STEPS):
            a = agent.act(eval_state)
            ns, r, term, trunc, _ = env.step(a)
            eval_state  = np.reshape(ns, [1, state_size])
            eval_reward += r
            if term or trunc:
                break

        agent.epsilon = eps_backup
        print(f"\n     [EVAL]  [EVAL] Greedy Score @ ep {e+1}: {eval_reward:.2f}  "
              f"(Best MovAvg so far: {best_moving_avg:.2f})\n")

# ─────────────────────────────────────────
# 3. บันทึก Model
# ─────────────────────────────────────────
agent.model.save_weights(final_model_path)
agent.model.save(final_full_model_path)        # full model สำหรับ TFLite conversion
print(f"\n[OK]  Training complete!")
print(f"     [Save]  Final weights -> {final_model_path}")
print(f"     [Save]  Full model    -> {final_full_model_path}")
print(f"     [Best]  Best model    -> {best_model_path}  (MovAvg = {best_moving_avg:.2f})")

# ─────────────────────────────────────────
# 4. กราฟพัฒนาการ -- รวมและแยกตาม Profile
# ─────────────────────────────────────────
def _moving_average(data, window):
    return [np.mean(data[max(0, i - window + 1): i + 1]) for i in range(len(data))]

def _plot_with_stages(data, title, filename, ylabel="Reward"):
    avg = _moving_average(data, MOVING_AVG_WIN)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(data, color='#90CAF9', linewidth=1, alpha=0.7, label='per Episode')
    ax.plot(avg,  color='#1565C0', linewidth=2,             label=f'Moving Avg ({MOVING_AVG_WIN} ep)')
    ax.axhline(0, color='red', linestyle='--', linewidth=1, label='Target (0)')
    # Curriculum stage boundaries
    colors = ['#e0e0e0', '#d0d0d0', '#c0c0c0', '#b0b0b0', '#a0a0a0']
    prev = 0
    for i, (end_ep, _) in enumerate(CURRICULUM):
        ax.axvspan(prev, end_ep - 1, alpha=0.15, color=colors[i], label=f'Stage {i+1}' if i < 3 else None)
        if end_ep < EPISODES:
            ax.axvline(end_ep - 1, color='gray', linestyle=':', linewidth=0.8)
        prev = end_ep
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"  [EVAL]  {filename}")

_plot_with_stages(reward_history,
    "AI Learning Progress -- Ride Comfort (DQN)",
    "learning_progress_csv.png",
    "Total Reward  (ยิ่งใกล้ 0 ยิ่งดี)")

# ─── Per-profile plots ────────────────────
for name in env.PROFILE_NAMES:
    _plot_with_stages(profile_rewards[name],
        f"Learning Progress -- {name}",
        f"learning_progress_{name}.png")