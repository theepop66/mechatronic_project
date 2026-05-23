import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import random
from collections import deque

class DQNAgent:
    # ค่าคงที่สำหรับ Normalize Input (ใช้ตรงกันกับ ESP32 firmware)
    _OBS_SCALE = np.array([150.0, 3.0, 50.0], dtype=np.float32)

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        # พารามิเตอร์การเรียนรู้
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99         # per-episode
        self.learning_rate = 0.0005
        self.batch_size = 64

        # Polyak (soft target update)
        self.tau = 0.005

        self.memory = deque(maxlen=10000)
        self._train_step_counter = 0

        # สร้าง Main Network และ Target Network
        self.model = self._build_model()
        self.target_model = self._build_model()
        self._hard_update_target()

    def _build_model(self):
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.learning_rate,
            decay_steps=100000,
            decay_rate=0.5,
            staircase=True
        )
        model = keras.Sequential([
            layers.Input(shape=(self.state_size,)),
            layers.Dense(64, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(
            loss=keras.losses.Huber(),
            optimizer=keras.optimizers.Adam(
                learning_rate=lr_schedule,
                clipnorm=1.0
            )
        )
        return model

    def _hard_update_target(self):
        """✅ คัดลอกน้ำหนักจาก Main → Target Network (ใช้ครั้งแรกเท่านั้น)"""
        self.target_model.set_weights(self.model.get_weights())

    def _polyak_update_target(self):
        """✅ Soft update: τ * main + (1-τ) * target"""
        main_w = self.model.get_weights()
        target_w = self.target_model.get_weights()
        new_w = [
            self.tau * m + (1.0 - self.tau) * t
            for m, t in zip(main_w, target_w)
        ]
        self.target_model.set_weights(new_w)

    def _preprocess(self, state):
        return state / self._OBS_SCALE

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model(self._preprocess(state), training=False).numpy()
        return np.argmax(act_values[0])

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)

        states      = self._preprocess(np.array([i[0][0] for i in minibatch]))
        actions     = np.array([i[1]    for i in minibatch])
        rewards     = np.array([i[2]    for i in minibatch])
        next_states = self._preprocess(np.array([i[3][0] for i in minibatch]))
        dones       = np.array([i[4]    for i in minibatch])

        # ✅ Double DQN:
        #    - Main network เลือก action ที่ดีที่สุด
        #    - Target network ประเมิน Q-value ของ action นั้น
        #    → ป้องกัน overestimation ของ Q-value
        next_actions     = np.argmax(self.model(next_states, training=False).numpy(), axis=1)
        next_q_target    = self.target_model(next_states, training=False).numpy()
        next_q_selected  = next_q_target[np.arange(self.batch_size), next_actions]

        targets = self.model(states, training=False).numpy()
        for i in range(self.batch_size):
            if dones[i]:
                targets[i][actions[i]] = rewards[i]
            else:
                targets[i][actions[i]] = rewards[i] + self.gamma * next_q_selected[i]

        self.model.train_on_batch(states, targets)
        self._train_step_counter += 1

        # ✅ Polyak soft update (ทุก step)
        self._polyak_update_target()

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay