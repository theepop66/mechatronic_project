import numpy as np
import matplotlib.pyplot as plt
import os

from environments.quarter_car_env import QuarterCarEnv
from agent.dqn_agent import DQNAgent

# ─────────────────────────────────────────
# 1. เตรียม Environment และ Agent
# ─────────────────────────────────────────
env = QuarterCarEnv()
state_size  = int(env.observation_space.shape[0])   # 3
action_size = int(env.action_space.n)               # 10
agent = DQNAgent(state_size, action_size)

# ─────────────────────────────────────────
# 2. ตั้งค่าการเทรน
# ─────────────────────────────────────────
EPISODES        = 500        # ✅ เพิ่มรอบ — DQN ต้องการประสบการณ์เพียงพอ
MAX_STEPS       = 1000       # 10 วินาที x 100 Hz
MOVING_AVG_WIN  = 20         # ✅ หน้าต่าง moving average สำหรับวัดพัฒนาการ
EVAL_FREQ       = 50         # ✅ ประเมินผล (ไม่สุ่ม) ทุก N episode
WARMUP_STEPS    = 500        # ✅ รอสะสม memory ก่อนเริ่ม replay

model_dir  = "models"
os.makedirs(model_dir, exist_ok=True)
best_model_path = os.path.join(model_dir, "best_suspension_dqn.weights.h5")
final_model_path = os.path.join(model_dir, "final_suspension_dqn.weights.h5")
final_full_model_path = os.path.join(model_dir, "final_suspension_dqn.keras")

reward_history   = []
best_moving_avg  = -np.inf
total_steps      = 0           # ✅ นับ step สะสมทั้งหมด (สำหรับ warmup)

print("🚀 เริ่มต้นกระบวนการฝึกฝน AI...")
print(f"   Episodes={EPISODES} | Warmup={WARMUP_STEPS} steps | Eval ทุก {EVAL_FREQ} ep\n")

for e in range(EPISODES):
    state, _ = env.reset()
    state       = np.reshape(state, [1, state_size])
    total_reward = 0.0

    for t in range(MAX_STEPS):
        action = agent.act(state)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done       = terminated or truncated
        next_state = np.reshape(next_state, [1, state_size])

        agent.remember(state, action, reward, next_state, done)
        state        = next_state
        total_reward += reward
        total_steps  += 1

        # ✅ replay ทุก step (ไม่ต้องรอทุก 5 steps)
        #    และเริ่มได้หลัง warmup เท่านั้น เพื่อให้ memory มีความหลากหลายพอ
        if total_steps >= WARMUP_STEPS:
            agent.replay()

        if done:
            break

    reward_history.append(total_reward)

    # ─── Per-episode epsilon decay ──────────
    agent.decay_epsilon()

    # ─── Moving Average ─────────────────────
    moving_avg = np.mean(reward_history[-MOVING_AVG_WIN:])

    # ✅ บันทึกโมเดลที่ดีที่สุดตาม moving average (ไม่ใช่ episode เดี่ยว)
    if len(reward_history) >= MOVING_AVG_WIN and moving_avg > best_moving_avg:
        best_moving_avg = moving_avg
        agent.model.save_weights(best_model_path)

    # ─── Log ────────────────────────────────
    warmup_tag = "⏳WARMUP" if total_steps < WARMUP_STEPS else "       "
    print(f"🏁 Ep {e+1:03d}/{EPISODES} {warmup_tag} | "
          f"Score: {total_reward:8.2f} | "
          f"MovAvg({MOVING_AVG_WIN}): {moving_avg:8.2f} | "
          f"ε: {agent.epsilon:.3f}")

    # ─── Evaluation Episode (ε=0) ───────────
    # ✅ ทดสอบโมเดลแบบ greedy เพื่อดูประสิทธิภาพจริง (ไม่สุ่ม)
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
        print(f"\n   📊 [EVAL] Greedy Score @ ep {e+1}: {eval_reward:.2f}  "
              f"(Best MovAvg so far: {best_moving_avg:.2f})\n")

# ─────────────────────────────────────────
# 3. บันทึก Model
# ─────────────────────────────────────────
agent.model.save_weights(final_model_path)
agent.model.save(final_full_model_path)        # full model สำหรับ TFLite conversion
print(f"\n✅ เทรนเสร็จ!")
print(f"   💾 Final weights → {final_model_path}")
print(f"   💾 Full model    → {final_full_model_path}")
print(f"   🏆 Best model    → {best_model_path}  (MovAvg = {best_moving_avg:.2f})")

# ─────────────────────────────────────────
# 4. กราฟพัฒนาการ
# ─────────────────────────────────────────
moving_avgs = [
    np.mean(reward_history[max(0, i - MOVING_AVG_WIN + 1): i + 1])
    for i in range(len(reward_history))
]

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(reward_history, color='#90CAF9', linewidth=1, alpha=0.7, label='Score per Episode')
ax.plot(moving_avgs,    color='#1565C0', linewidth=2,             label=f'Moving Avg ({MOVING_AVG_WIN} ep)')
ax.axhline(0, color='red', linestyle='--', linewidth=1, label='Target (0)')
ax.set_title("AI Learning Progress — Ride Comfort (DQN)", fontsize=14)
ax.set_xlabel("Episode")
ax.set_ylabel("Total Reward  (ยิ่งใกล้ 0 ยิ่งดี)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("learning_progress.png", dpi=150)
print("📊 บันทึกกราฟไว้ที่: learning_progress.png")