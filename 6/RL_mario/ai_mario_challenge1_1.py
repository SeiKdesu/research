# -*- coding: utf-8 -*-
"""AI_Mario_Challenge1_1.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Slhd7NsVIWXKTkrWMqdc9l1RL90bvg4X

# 初心者をAIマリオへいざなうColab Notebook

### このColab Notebookでやること
- ファミコン版スーパーマリオブラザーズ World 1-1を、人工知能にプレイさせてクリアをめざします。
- この人工知能の使い方は**強化学習**と呼ばれるものです。
- 人工知能は何度もゲームをプレイしながら、だんだん上手なプレイのしかたを学習してゆきます。このNotebookでは、約7,000回、5時間くらいのプレイでいい感じになります（運により結果はわりと変わります）。
- ほとんど知識のない方でもNotebookのセルを順番に実行するだけでクリアできるようにしました。
- 学習中に待っているだけだと飽きてしまうので、「変化している」ことが感じられるよう、定期的にGoogle Driveにプレイ動画を保存しています。
- 理解のカギになる用語を太字にしてありますので、この辺りをヒントに知識を広げていくとだんだん<del>沼にはまる</del>楽しくなってくると思います。

### **強化学習**についてポイントだけ説明
- **強化学習**には**「状態・行動・報酬」**の３つの要素があります……　って教科書には書いてあるのですが、そんなことを言ったとたんに「**環境**」がでてくるので大人はずるいと思ったりするかもしれません。
- **環境**は、ゲーム機そのものです。ファミコンのエミュレータでスーパーマリオブラザーズが動いています。
- **状態**は、ゲーム画面にいま表示されている映像です。人工知能は、画面の映像を読み取ってプレイします。
- **行動**は、コントローラーの操作です。人工知能は**状態**をもとに「**複雑な計算**」を行い、どのようにコントローラーを操作するか、**行動**を決定します。
- **報酬**は、**行動**の結果得られる得点のようなものです。今回はマリオが右へ進むと報酬が得られるようになっています。
- **強化学習**は、「**状態**をもとに**行動**を決定し、その結果、**報酬**が得られたり得られなかったりする」ことを通して、人工知能が多くの報酬を得られるように**行動**を（正確には「**複雑な計算**」の中身を）少しずつ変化させてゆくものです。

### 出典などの情報
本当は、このような強化学習の実行にはとても複雑なプログラミングが必要ですが、インターネット上で提供されているほかの方の成果をお借りすることでとてもコンパクトにまとめることができています。
- [PyTorch](https://pytorch.org/)
- [OpenAI Gym](https://github.com/openai/gym)
- [Nes-py](https://github.com/Kautenja/nes-py)
- [gym-super-mario-bros](https://github.com/Kautenja/gym-super-mario-bros)
- [このNotebookのもとになったNotebook](https://colab.research.google.com/github/YutaroOgawa/pytorch_tutorials_jp/blob/main/notebook/4_RL/4_2_mario_rl_tutorial_jp.ipynb#scrollTo=mMiIWSeXPKHb)

また、スーパーマリオブラザーズ全部のステージクリアしようぜ！というプロジェクトも行われていて、ほかの方の成果も集まっています。
- [からあげさんのQiitaの記事](https://qiita.com/karaage0703/items/e237887894f0f1382d58)
- [mario-ai-challenge](https://karaage0703.github.io/mario-ai-challenge/)

Google Colabが初めてという方は、インターネット上にたくさんのチュートリアルが存在するので、探してみてください。

#### このNotebookに含まれている工夫
- **強化学習**にはさまざまな手法がありますが、ここでは**DDQN**（Double Deep Q Network）を使っています。
- **状態**を認識させるために、NNの一部としてResNet（Residual Network）を参考にしたブロックを使っています。ResNetの良い点のひとつは、入力のサイズと出力のサイズが同じなので、層を増やすことが簡単で、また層を増やすことで複雑な状態を認識できるようになることです。ただし、今回は、２ブロックしか使っていません。
- 学習を効率よく進めるために、画面（**状態**）全体を使わずにマリオの前方だけを使っています。
- 特に独自の工夫として、**状態**をNNに入力する前に、ランダムにブレさせる画像処理をしています。これにより**行動**の選択が少し安全側に倒されるようです。
- **行動**も使える選択肢を限定しています。右ボタンとAボタンしか使わず、Bボタンは押したままです。

### 用語
軽く書いた適当な説明なので、ちゃんとした知識として知りたいかたは書籍などをあたってください。

|用語|説明|備考|
| :--- | :--- | ---- |
|NN|人工知能の核となる、ニューラルネットワーク（Neural Network）のこと。ではニューラルネットワークとは何かというと……　Wikipediaで調べてください。| |
|モデル|NNと、それを機能させるのに必要な付随する情報をひとまとめにしてこう呼ぶようです。 | |
|CUDA|NVIDIA社が提供しているライブラリ。NNの処理には膨大な計算量が必要なのですが、ColabではGPUを使って高速に計算できる環境を用意しています。NVIDIAのGPUでNNの計算をするにはこの機能が必要です。 | |
|Google Drive|みんな知っているクラウド上にファイルを保存できる場所。意外とすぐに容量が足りなくなるので注意。 | |
|フレーム|ゲーム画面の変化をパラパラ漫画のように捉えたときの、1ページが1フレームです。 | |
|エピソード(episode)|マリオひとり分のプレイを1エピソードと呼びます。10,000エピソード学習させるとは、つまり10,000人のマリオが犠牲になるということです。マリオが労働組合を結成しないことを祈りましょう。| |
|ステップ(step)|状態をもとに行動をおこすことを1ステップと呼びます。つまりマリオを1回操作する機会のことです。フレームと同じじゃないの？と思われるかもしれませんが、このNotebookではいくつかのフレームをまとめて1ステップにしています。| |
|学習(Learning)|状態をもとに行動を起こして報酬を得た結果をしばらく記憶しておいて、それをもとにNNの計算を変化させることを学習といいます。| |
|学習率(Learning rate)|学習するとき、計算方法をちょっとだけ変化させるのですが、変化の大きさを決めるのが学習率です。数字が大きいと大きく変化させるのですが、どのくらい変化させるべきかはなかなか難しいです。| |
|探索(Exploration)|より良い学習のためには、これまでと違う新しい行動を試してみることも必要です。ある確率に従って、ランダムにコントローラーを操作します。これを探索といいます。| |
|探索率(Exploration rate)|探索を行う確率です。1.0にすると、いつもランダムに操作します。0.0になると、ランダムな操作は無く、いつも計算によって行動を決めます（探索の反対語として活用といいます）。探索率は教科書ではε（イプシロン）と書かれます。| |
|環境(environment)|このプログラムではenvと書かれています。||
|状態(state)|このプログラムではstateと書かれています。||
|行動(action)|このプログラムではactと書かれています。||
|報酬(reward)|このプログラムではrewardと書かれています。||

#### （１）パラメータ設定

プログラムは複雑なので、動作の微調整は、最初にここでまとめて行います。
"""

# 実行に関わるスイッチ
DEBUG                  = True      # DEBUGをTrueにするとすべてのセルを実行。Falseにすると必須のセルだけ実行。
LOAD_MODEL             = False     # SAVED_MODELをロードして学習を継続する場合にTrue
DO_RECORD              = True      # 学習後にプレイ動画を保存
USE_CUDA               = True      # CUDA使う？ Colabならもちろん使おう。

# マリオが挑戦するワールドを指定
WORLD                  = "SuperMarioBros-1-1-v0"
WORLD_COLOR            = "SuperMarioBros-1-1-v0" # 再現録画をとるとき用

# 学習途中のNNやログを保存する場所。実行者のGoogle Driveに保存される。Google Driveの空き容量に注意。
LOG_DIR                = './AI_Mario'

# 学習済みモデルから開始する場合は*.chkptファイルをフルパス指定。
SAVED_MODEL            = './AI_Mario/checkpoints/2024-07-21T19-14-00/*.chkpt'

# 環境のパラメータ
CROPSIZE               = 170       # もとのゲーム画面のサイズは 246x231 だが、右下の170x170の範囲を使う
SKIPFRAMES             = 4         # 全部のフレームは計算に使わない。
RESIZE                 = 52        # なるべく縮小して学習時間を短縮
NUM_FRAME_STACK        = 4

# 学習のパラメータ
NUM_EPISODES           = 7000      # 学習のためマリオを走らせる人数
BURNIN                 = 10000     # このstep数だけ記憶がたまるまでは学習を開始しない
LEARN_EVERY            = 3         # このstep数ごとに小さな学習が行われる
LEARNING_RATE          = 0.000300
LEARNING_RATE_DECAY    = 0.000005
LEARNING_RATE_MIN      = 0.000050
SYNC_EVERY             = 10000     # このstep数ごとにonlineとtargetの2つのNNの同期が行われる
EXPLORATION_RATE       = 1.0       # ε値の初期値(stepごとにランダムに操作する確率)
EXPLORATION_RATE_DECAY = 0.9999975 # ε値を小さくしていく割合
EXPLORATION_RATE_MIN   = 0.1       # ε値の最小値
SAVE_EVERY             = 100000    # このstep数ごとにNNを保存する
QUEUE_MAXLEN           = 58000     # GPUメモリ上にすべてのstepの(state, act)が記憶されるが、最大の保存数。
                                   # 多くするとGPUメモリもたくさん必要になる。RESIZEが大きいと少なくする必要がある。
                                   # GPU環境に合わせて増減させる。
BATCH_SIZE             = 32        # 学習するときは、記憶のQUEUEからランダムにこの数だけ取り出して使う。
GAMMA                  = 0.9       # 学習のパラメータのひとつ。まだ変更したことがない。
MOVEMENT = [                       # マリオを操作するボタンの押し方一覧。選択肢が少ないほうが学習は短時間で進む。
#     ['NOOP'],
    ['B'],
    ['right', 'B'],
    ['right', 'A', 'B'],
    ['A', 'B']
]

# 録画用のパラメータ
RECORD_SUCCESS_XPOS = 3200         # 録画時、このXを超えるか、ゴールできたら「成功」として録画する。World 1-1は、X=3200くらいでゴール
RECORD_NFILES       = 10           # この数だけ成功録画を録画する。
RECORD_EPSILON      = 0.1          # 録画の時に使用するε値

# Colab環境に、実行者のGoogle Driveをマウント(接続)する。学習中のNNやログの保存用に使う。


"""##### （２）実行環境の準備"""

# Colab環境にマリオをインストール


"""#### （３）モジュールのロード"""

import os
import datetime
import random
import copy
import time

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
from pathlib import Path
from collections import deque
import pickle
from IPython import display

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms as T
import torchvision.transforms.functional as TF
from torchsummary import summary

import gym
from gym.spaces import Box
from gym.wrappers import FrameStack
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace

"""#### （４）環境(env)"""

class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        """スキップした後のフレームのみを返す"""
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """行動を繰り返し、報酬を合計する"""
        total_reward = 0.0
        done = False
        for i in range(self._skip):
            # 報酬を蓄積し、同じ行動を繰り返す
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info

class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def permute_orientation(self, observation):
        # [H, W, C] のarrayを、[C, H, W] のtensorに変換
        observation = np.transpose(observation, (2, 0, 1))
        observation = torch.tensor(observation.copy(), dtype=torch.float)
        return observation

    def observation(self, observation):
        observation = self.permute_orientation(observation)
        transform = T.Grayscale()
        observation = transform(observation)
        return observation

class MyShiftTransform:
    """
    画面をランダムに少しズラすことで、視界を乱して、マリオが少し余裕をみて操作されるように学習する。
    ResizeObservationの中で使用される。
    """
    def __init__(self):
        pass

    def __call__(self, x):
        nshift = random.random() * 5
        return TF.affine(x, angle=0, translate=(nshift, 0), scale=1.0, shear=0)

class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        transforms = T.Compose([
            T.Resize(self.shape),
            MyShiftTransform(), # カスタマイズ
            T.Normalize(0, 255)
        ])
        obs = T.FiveCrop(size=(CROPSIZE, CROPSIZE))(observation)[3] # 画面の右下だけ切り抜き
        observation = transforms(obs).squeeze(0)
        return observation

"""#### （５）【参考】環境(env)の確認"""

if DEBUG:
    # 画面をNNの入力として使えるように加工する様子を確認できます。
    env5 = gym_super_mario_bros.make(WORLD)
    env4 = JoypadSpace(env5, MOVEMENT)
    env3 = SkipFrame(env4, skip=SKIPFRAMES)
    env2 = GrayScaleObservation(env3)
    env1 = ResizeObservation(env2, shape=RESIZE)
    env  = FrameStack(env1, num_stack=NUM_FRAME_STACK)

if DEBUG:
    env.reset()
    ret5 = env5.step(action=0)
    ret4 = env4.step(action=0)
    ret3 = env3.step(action=0)
    ret2 = env2.step(action=0)
    ret1 = env1.step(action=0)
    ret  = env.step(action=0)

if DEBUG:
    # もとのゲーム画面
    print(ret5[0].shape)
    plt.axis('off')
    plt.imshow(ret5[0])

if DEBUG:
    # Joypadで操作できるようにする（見た目は変化なし）
    print(ret4[0].shape)
    plt.axis('off')
    plt.imshow(ret4[0])

if DEBUG:
    # フレームをスキップするようにする（報酬は足し合わされている）
    print(ret3[0].shape)
    plt.axis('off')
    plt.imshow(ret3[0])

if DEBUG:
    # データサイズを減らすためグレースケールにする
    print(ret2[0].shape)
    plt.axis('off')
    plt.imshow(ret2[0][0,:,:])

if DEBUG:
    # サイズを落としている
    print(ret1[0].shape)
    plt.axis('off')
    plt.imshow(ret1[0])

def display_all_frame():
    plt.figure(figsize=(16,16))
    for idx in range(ret[0].shape[0]):
        plt.subplot(1, NUM_FRAME_STACK, idx+1)
        plt.axis('off')
        plt.imshow(ret[0][idx])
    plt.show()

if DEBUG:
    # 画面の変化がわかるよう、NUM_FRAME_STACK枚だけまとめて入力する
    print(ret[0].shape)
    display_all_frame()

"""#### （６）環境(env)の作成"""

# 環境の作成
env = gym_super_mario_bros.make(WORLD)
env = JoypadSpace(env, MOVEMENT)
env = SkipFrame(env, skip=SKIPFRAMES)
env = GrayScaleObservation(env)
env = ResizeObservation(env, shape=RESIZE)
env = FrameStack(env, num_stack=NUM_FRAME_STACK)

"""#### （７）ニューラルネットワークの定義"""

class MarioBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv2d1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, padding=1, stride=1)
        self.bn1     = nn.BatchNorm2d(16)
        self.conv2d2 = nn.Conv2d(in_channels=16, out_channels=4, kernel_size=3, padding=1, stride=1)
        self.bn2     = nn.BatchNorm2d(in_channels)
        self.relu    = nn.ReLU(inplace=True)

    def forward(self, x):
        output = self.conv2d1(x)
        output = self.bn1(output)
        output = self.relu(output)
        output = self.conv2d2(output)
        output = self.bn2(output)
        output += x
        output = self.relu(output)
        return output

class MarioNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        c, h, w = input_dim

        self.online = nn.Sequential(
            MarioBlock(c),             # 主に画面を認識する部分
            MarioBlock(c),             # 主に画面を認識する部分
            nn.Flatten(),
            nn.Linear(c*h*w, 96),     # 操作を選択する部分
            nn.ReLU(),
            nn.Linear(96, output_dim) # 操作を選択する部分
        )

        # DQNでは、targetネットワークは普段は変更せず、定期的にonlineと同期される。
        self.target = copy.deepcopy(self.online)

        # targetはパラメータが変更されないよう固定する。
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, input, model):
        if model == "online":
            return self.online(input)
        elif model == "target":
            return self.target(input)

"""#### （８）エージェントの定義"""

class Mario:
    def __init__(self, state_dim, action_dim, save_dir):
        self.state_dim              = state_dim
        self.action_dim             = action_dim
        self.save_dir               = save_dir
        self.use_cuda               = torch.cuda.is_available()
        if USE_CUDA == False:
            self.use_cuda = False
        self.net                    = MarioNet(self.state_dim, self.action_dim).float()
        if self.use_cuda:
            self.net                = self.net.to(device="cuda")
        self.exploration_rate       = EXPLORATION_RATE
        self.exploration_rate_decay = EXPLORATION_RATE_DECAY
        self.exploration_rate_min   = EXPLORATION_RATE_MIN
        self.total_episodes         = 0
        self.total_steps            = 0
        self.curr_step              = 0
        self.save_every             = SAVE_EVERY
        self.memory                 = deque(maxlen=QUEUE_MAXLEN)
        self.pool                   = []
        self.batch_size             = BATCH_SIZE
        self.gamma                  = GAMMA
        self.optimizer              = torch.optim.Adam(self.net.parameters(), lr=LEARNING_RATE)
        self.loss_fn                = torch.nn.SmoothL1Loss()
        self.burnin                 = BURNIN       # 経験を訓練させるために最低限必要なステップ数
        self.learn_every            = LEARN_EVERY  # Q_onlineを更新するタイミングを示すステップ数
        self.sync_every             = SYNC_EVERY   # Q_target & Q_onlineを同期させるタイミングを示すステップ数

    def act(self, state, epsilon=-1):
        """
        MarioNetに3個の情報を入力し、Marioの操作(action)を決定する。
        state   ... 画面の状態
        epsilon ... 通常はself.exploration_rateでε値が決まるが、引数で変更することも可能。
        """
        if ((epsilon == -1) and (np.random.rand() < self.exploration_rate)) or (np.random.rand() < epsilon):
            # 【探索】ランダム値がε値以下の場合、適当な操作をする
            action_idx = np.random.randint(self.action_dim)
        else:
            # 【活用】MarioNetを使用して、もっとも成功が見込める操作をする
            state = state.__array__()
            if self.use_cuda:
                state = torch.tensor(state).cuda()
            else:
                state = torch.tensor(state)
            state = state.unsqueeze(0)
            action_values = self.net(state, model="online")
            action_idx = torch.argmax(action_values, axis=1).item()

        # exploration_rateを減衰させる
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        # ステップを+1します
        self.curr_step += 1
        self.total_steps += 1
        return action_idx

    def cache(self, state, next_state, action, reward, done):
        """
        経験をself.memory (replay buffer)に保存します

        Inputs:
            state (LazyFrame),
            next_state (LazyFrame),
            action (int),
            reward (float),
            done(bool))
        """
        state = state.__array__()
        next_state = next_state.__array__()

        if self.use_cuda:
            state = torch.tensor(state).cuda()
            next_state = torch.tensor(next_state).cuda()
            action = torch.tensor([action]).cuda()
            reward = torch.tensor([reward]).cuda()
            done = torch.tensor([done]).cuda()
        else:
            state = torch.tensor(state)
            next_state = torch.tensor(next_state)
            action = torch.tensor([action])
            reward = torch.tensor([reward])
            done = torch.tensor([done])

        self.memory.append((state, next_state, action, reward, done,))

    def pool_cache(self, state, next_state, action, reward, done):
        """
        経験を学習するか一時的に保留しておきます。
        """
        self.pool.append((state, next_state, action, reward, done,))

    def delete_pooled_cache(self):
        """
        pool_cacheで保留されていた経験を破棄します。
        """
        self.pool.clear()

    def remember_pooled_cache(self, times=1):
        """
        pool_cacheで保留されていた経験を記憶します。
        """
        for i in range(times):
            for m in self.pool:
                self.cache(m[0], m[1], m[2], m[3], m[4])
        self.pool.clear()

    def recall(self):
        """
        学習用に、記憶からまとまった量をランダムに取り出します
        """
        batch = random.sample(self.memory, self.batch_size)
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

    def learn(self):
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        if self.curr_step % self.save_every == 0:
            self.save()

        if self.curr_step < self.burnin:
            return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None

        state, next_state, action, reward, done = self.recall()
        td_est = self.td_estimate(state, action)
        td_tgt = self.td_target(reward, next_state, done)
        loss = self.update_Q_online(td_est, td_tgt)

        return (td_est.mean().item(), loss)

    def td_estimate(self, state, action):
        current_Q = self.net(state, model="online")[
            np.arange(0, self.batch_size), action
        ]  # Q_online(s,a)
        return current_Q

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        next_state_Q = self.net(next_state, model="online")
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.net(next_state, model="target")[
            np.arange(0, self.batch_size), best_action
        ]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()

    def update_Q_online(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())

    def save(self):
        save_path = (
            self.save_dir / f"mario_net_{int(self.curr_step // self.save_every)}.chkpt"
        )
        torch.save(
            dict(model=self.net.state_dict(), exploration_rate=self.exploration_rate, total_episodes=self.total_episodes, total_steps=self.total_steps),
            save_path,
        )
        print(f"MarioNet saved to {save_path} at step {self.curr_step}")

"""#### （９）ログの保存方法を定義"""

class MetricLogger:
    def __init__(self, save_dir):
        self.save_log = save_dir / "log"
        self.save_logX = save_dir / "logX"

        with open(self.save_log, "w") as f:
            f.write(
                f"{'Episode':>8}{'Step':>8}{'Epsilon':>10}{'MeanReward':>15}"
                f"{'MeanLength':>15}{'MeanLoss':>15}{'MeanQValue':>15}"
                f"{'TimeDelta':>15}{'Time':>20}\n"
            )
        self.ep_rewards_plot    = save_dir / "reward_plot.jpg"
        self.ep_lengths_plot    = save_dir / "length_plot.jpg"
        self.ep_avg_losses_plot = save_dir / "loss_plot.jpg"
        self.ep_avg_qs_plot     = save_dir / "q_plot.jpg"

        # 指標の履歴
        self.ep_rewards = []
        self.ep_lengths = []
        self.ep_avg_losses = []
        self.ep_avg_qs = []
        self.ep_posXs = []

        # record()が呼び出されるたびに追加される移動平均
        self.moving_avg_ep_rewards = []
        self.moving_avg_ep_lengths = []
        self.moving_avg_ep_avg_losses = []
        self.moving_avg_ep_avg_qs = []

        # 現在のエピソードの指標
        self.init_episode()

        # 時間を記録
        self.record_time = time.time()

    def log_step(self, reward, loss, q, posX):
        self.curr_ep_reward += reward
        self.curr_ep_length += 1
        self.curr_ep_posX    = posX
        if loss:
            self.curr_ep_loss += loss
            self.curr_ep_q += q
            self.curr_ep_loss_length += 1

    def log_episode(self):
        "エピソード終了時の記録"
        self.ep_rewards.append(self.curr_ep_reward)
        self.ep_lengths.append(self.curr_ep_length)
        self.ep_posXs.append(str(self.curr_ep_posX))
        if self.curr_ep_loss_length == 0:
            ep_avg_loss = 0
            ep_avg_q = 0
        else:
            ep_avg_loss = np.round(self.curr_ep_loss / self.curr_ep_loss_length, 5)
            ep_avg_q = np.round(self.curr_ep_q / self.curr_ep_loss_length, 5)
        self.ep_avg_losses.append(ep_avg_loss)
        self.ep_avg_qs.append(ep_avg_q)

        self.init_episode()

    def init_episode(self):
        self.curr_ep_reward = 0.0
        self.curr_ep_length = 0
        self.curr_ep_loss = 0.0
        self.curr_ep_q = 0.0
        self.curr_ep_loss_length = 0
        self.curr_ep_posX = 0

    def record(self, episode, epsilon, step):
        mean_ep_reward = np.round(np.mean(self.ep_rewards[-100:]), 3)
        mean_ep_length = np.round(np.mean(self.ep_lengths[-100:]), 3)
        mean_ep_loss = np.round(np.mean(self.ep_avg_losses[-100:]), 3)
        mean_ep_q = np.round(np.mean(self.ep_avg_qs[-100:]), 3)
        self.moving_avg_ep_rewards.append(mean_ep_reward)
        self.moving_avg_ep_lengths.append(mean_ep_length)
        self.moving_avg_ep_avg_losses.append(mean_ep_loss)
        self.moving_avg_ep_avg_qs.append(mean_ep_q)

        last_record_time = self.record_time
        self.record_time = time.time()
        time_since_last_record = np.round(self.record_time - last_record_time, 3)

        print('Episode %6d - Step %8d - Epsilon %1.9f - Reward %3.3f - Length %7s - Loss %1.3f - Q Value %3.3f - Time Delta %8s - Time %s' %
              (episode, step, epsilon, mean_ep_reward, str('%3.3f' %(mean_ep_length)),
               mean_ep_loss, mean_ep_q, str('%3.3f' %(time_since_last_record)),
               datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).strftime('%Y-%m-%dT%H:%M:%S')))

        with open(self.save_log, "a") as f:
            f.write(
                f"{episode:8d}{step:9d}{epsilon:10.3f}"
                f"{mean_ep_reward:15.3f}{mean_ep_length:15.3f}{mean_ep_loss:15.3f}{mean_ep_q:15.3f}"
                f"{time_since_last_record:15.3f}"
                f"{datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).strftime('%Y-%m-%dT%H:%M:%S'):>20}\n"
            )

        with open(self.save_logX, "a") as f:
            oneline = ",".join(self.ep_posXs)
            f.write(
                f"{oneline}\n"
            )
            self.ep_posXs = []

        for metric in ["ep_rewards", "ep_lengths", "ep_avg_losses", "ep_avg_qs"]:
            plt.plot(getattr(self, f"moving_avg_{metric}"))
            plt.title(metric, loc='right')
            plt.savefig(getattr(self, f"{metric}_plot"))
            plt.clf()

"""#### （１０）実行の準備"""

# ログの保管場所を作成
dt_now_JST = datetime.timezone(datetime.timedelta(hours=9))
save_dir = Path(LOG_DIR + "/checkpoints") / datetime.datetime.now(dt_now_JST).strftime("%Y-%m-%dT%H-%M-%S")
save_dir.mkdir(parents=True)
save_dir

# Marioの作成
mario = Mario(state_dim=(NUM_FRAME_STACK, RESIZE, RESIZE), action_dim=env.action_space.n, save_dir=save_dir)
# その他のログ出力を準備
logger = MetricLogger(save_dir)

# マリオの中身をざっくり確認
summary(mario.net.online, (SKIPFRAMES,RESIZE,RESIZE))

"""#### （１１）【任意】学習結果の復元

過去の状態を復元したいときは、上のMarioの作成を実行したあと、オブジェクトのデータを上書きする。
"""

# checkpointからデータを読み込み
if LOAD_MODEL and SAVED_MODEL != '':
    print('Model Loading... %s' % SAVED_MODEL)
    loaded_data = torch.load(SAVED_MODEL)
    # marioオブジェクトに値をロード
    mario.net.load_state_dict(loaded_data['model'])
    mario.exploration_rate = loaded_data['exploration_rate']
    mario.total_episodes   = loaded_data['total_episodes']
    mario.total_steps      = loaded_data['total_steps']
    print('exploration_rate:%f' % (mario.exploration_rate))
    print('total_episodes  :%d' % (mario.total_episodes))
    print('total_steps     :%f' % (mario.total_steps))

"""#### （１２）動画保存の準備"""

def make_play_movie(img_reserve, number, title=''):
    # カラーでの動画を表示
    matplotlib.rcParams['animation.embed_limit'] = 2**128

    dpi = 200
    interval = 17 # ms

    plt.figure(figsize=(240/dpi, 256/dpi), dpi=dpi)
    # 軸は消す
    plt.axis('off')
    plt.title(title, loc='right')
    #patch = plt.imshow(img_reserve[0]) # patch: AxesImage class
    patch = plt.imshow(img_reserve[0], animated=True ) #, interpolation='none', ) # patch: AxesImage class
    def animate(i):
        patch.set_data(img_reserve[i])
        return [patch]
    #animate = lambda i: patch.set_data(img_reserve[i])
    ani = animation.FuncAnimation(plt.gcf(), animate, frames=len(img_reserve), interval=interval, blit=True)

    writer = animation.FFMpegWriter(fps=1000/interval, codec='libx264', extra_args=['-pix_fmt', 'yuv420p', '-crf', '0'])
    ani.save(os.path.join(save_dir, ('ai_mario%05d.mp4' % (number))), writer=writer)

    # ani.save(os.path.join(save_dir, ('ai_mario%05d.mp4' % (number))))
    # display.display(display.HTML(ani.to_jshtml()))
    plt.close()

def make_env_for_movie():
    env_color = gym_super_mario_bros.make(WORLD_COLOR)
    return JoypadSpace(env_color, MOVEMENT)

"""#### （１３）学習の実施"""

episodes = NUM_EPISODES       # 学習回数
maxX = 1                      # 最大到達点を更新
goals = 0                     # ゴールできた回数を記録
history = 0
reward_history = [0] * 100    # 報酬の履歴を記録
estart = mario.total_episodes # 学習を再開するときのための処理
env_color = make_env_for_movie()

for e in range(estart, episodes):

    # Learning Rateを定期的に減らす
    if e % 1000 == 0 and LEARNING_RATE > LEARNING_RATE_MIN:
        LEARNING_RATE -= LEARNING_RATE_DECAY
        # 下限を突き抜けないようにする
        if LEARNING_RATE < LEARNING_RATE_MIN:
            LEARNING_RATE = LEARNING_RATE_MIN
        mario.optimizer = torch.optim.Adam(mario.net.parameters(), lr=LEARNING_RATE)

    # 100回に1回は録画をする
    '''if e % 100 == 0:
        rec = True
        img_color = []
        env_color.reset()
    else:
        rec = False'''
    rec =True
    img_color=[]
    env_color.reset()

    # プレイ開始前に環境をリセット
    state = env.reset()
    total_reward = 0

    # 1回のプレイを開始！
    while True:

        # 【行動】ε値は最初大きすぎてランダムに操作しすぎるので、10エピソードを1単位として、変化させている。
        ep = (e % 10) * 0.1
        if ep > mario.exploration_rate:
            action = mario.act(state)             # NNに設定されたε値で行動
        else:
            action = mario.act(state, epsilon=ep) # (小さく)カスタマイズされたε値で行動

        # 【環境】操作に応じて環境は変化する。
        next_state, reward, done, info = env.step(action)
        total_reward += reward

        # 録画用（4 skipしているので4回同じ行動をします）
        if rec:
            for i in range(SKIPFRAMES):
                next_state_color, _, done2, _ = env_color.step(action)
                rgb_img_color = np.stack(next_state_color, axis=0)
                img_color.append(rgb_img_color)
                if done2:
                    break

        # 【記憶】すぐにはcacheせずに、いったんpoolする。1エピソードが終わってから記憶するか決める。
        mario.pool_cache(state, next_state, action, reward, done)

        # 【学習】learn内部では記憶に応じて学習している
        q, loss = mario.learn()
        logger.log_step(reward, loss, q, info["x_pos"])

        # 状態の更新
        state = next_state

        # 1エピソードが終了したか確認
        if done or info["flag_get"]:
            # 終了時、到達点を記録
            reward_history[history] = info["x_pos"]
            history = (history + 1) % 100
            # 最長不倒記録は更新した？
            if info["x_pos"] > maxX:
                maxX = info["x_pos"]
                print(f"<<< 最長不倒更新! %d >>>" % info["x_pos"])
            if info["flag_get"]:
                # ゴールできたら知りたいですよね
                goals += 1
                print(f"<<< ゴール! %d回目 >>>" % goals)
                title = 'Episode:%4d' % (e)
                make_play_movie(img_color, mario.total_episodes, title=title)

            # 今回のエピソードを記憶する
            mario.remember_pooled_cache()

            # 録画用の処理
            if rec:
                title = 'Episode:%4d' % (e)
                make_play_movie(img_color, mario.total_episodes, title=title)

            # 1回のプレイを終了
            break

    logger.log_episode()
    if e % 20 == 0 and e != 0:
        logger.record(episode=e, epsilon=mario.exploration_rate, step=mario.curr_step)
    mario.total_episodes += 1

# 予定のエピソードを消化したら終わり。最後に保存。お疲れさまでした。
mario.save()

"""#### （１４）AIプレイ動画の作成

学習が終わったら、最後に好プレイ動画を予定数保存します。
"""

if DO_RECORD:
    # 学習環境は、データが削減されておりプレイ動画の作成には向いていません。
    # 別の環境を用意し、同じactionを両方の環境に適用することで、通常の画面のほうで動画を撮ります。
    env_color = gym_super_mario_bros.make(WORLD_COLOR)
    env_color = JoypadSpace(env_color, MOVEMENT)

    max_reward=0
    img_reserve=[]
    make_n_moviefiles = 0

    while make_n_moviefiles < RECORD_NFILES:
        # 環境と状態を初期化します
        state_color = env_color.reset()
        state = env.reset()
        # action_replay = [] # actionの履歴を全部保存する

        # 動画作成用の画像を溜めるリスト
        img = []
        img_color=[]

        # うまくいった？
        total_reward = 0
        # step数
        num_step = 0

        # ゲーム開始！
        while True:

            # 現在の状態に対するエージェントの行動を決める
            action = mario.act(state, epsilon=RECORD_EPSILON)
            # action_replay.append(action)

            # エージェントが行動を実行
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            # 「本番」は学習せず、記録にも残さない。
            # grayscaleの画像をRGBの画像に変換
            rgb_img = np.stack((state[0],)*3,axis=0).transpose(1,2,0)
            img.append(rgb_img)
            # 録画用（4 skipしているので4回同じ行動をします）
            for i in range(SKIPFRAMES):
                next_state_color, _, done2, _ = env_color.step(action)
                rgb_img_color = np.stack(next_state_color, axis=0)
                img_color.append(rgb_img_color)
                if done2:
                    break

            # 状態の更新
            state = next_state
            num_step+=1
            for f in range(state.shape[0]):
                state[f][0][0] = info['x_pos'] / 10000

            # ゲームが終了したかどうかを確認
            if done or done2 or info["flag_get"]:
                if info["flag_get"]:
                    print(f"<<< ゴール! %d >>>" % info["x_pos"])
                break

        print("num_step: %4d, reward: %5d xpos: %4d max: %4d" % ( num_step, total_reward, info['x_pos'], max_reward))

        if max_reward < total_reward:
            max_reward = total_reward
            img_reserve = copy.copy(img_color)

        # 既定の回数成功するまで繰り返し
        if info['x_pos'] > RECORD_SUCCESS_XPOS or info["flag_get"]:
            print("GG! Recording...")
            make_play_movie(img_color, make_n_moviefiles, title='Learning Complete')
            make_n_moviefiles += 1