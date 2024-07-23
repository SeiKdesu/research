import torch
from torch import nn
from torchvision import transforms as T
from PIL import Image
import numpy as np
from pathlib import Path
from collections import deque
import random, datetime, os, copy

# Gymは、Open AIのRL用ツールキットです
import gym
from gym.spaces import Box
from gym.wrappers import FrameStack

# OpenAI Gym用に使うNES エミュレーター
from nes_py.wrappers import JoypadSpace

#OpenAI Gymのスーパー・マリオ・ブラザーズの環境
import gym_super_mario_bros
# スーパー・マリオの環境を初期化
env = gym_super_mario_bros.make("SuperMarioBros-1-1-v3")

env.reset()
next_state, reward, done, info = env.step(action=0)
print(f"{next_state.shape},\n {reward},\n {done},\n {info}")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig=plt.figure()
plt.imshow(next_state)
