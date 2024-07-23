import gymnasium as gym
import numpy as np

# 'HalfCheetah-v4' は MuJoCo 環境の一例です
env = gym.make("HalfCheetah-v4", render_mode="rgb_array")

# 環境のリセット
obs, info = env.reset()

frames = []  # フレームを保存するリスト

# 数ステップだけ環境を動かしてみる
for _ in range(100):
    action = env.action_space.sample()  # ランダムなアクションを選択
    obs, reward, done, truncated, info = env.step(action)
    frames.append(env.render())  # フレームを保存

    if done or truncated:
        obs, info = env.reset()

env.close()

# 動画として保存
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig = plt.figure()

# フレームの初期化
im = plt.imshow(frames[0])

def update(frame):
    im.set_array(frame)
    return [im]

ani = animation.FuncAnimation(fig, update, frames, interval=50)
ani.save("mujoco_simulation.mp4", writer="ffmpeg")