import gym

env = gym.make('CartPole-v0')
observation = env.reset()
count = 0
epoch_count = 0
total_count = 0

while epoch_count < 20:
    env.render()  # 現在の状況を画面表示する
    # ランダムに動かす
    action = env.action_space.sample()
    step_result = env.step(action)
    # 最後の要素（空の辞書）を削除する
    observation, reward, done, info = step_result[:-1]
    count += 1
    print("Step:", count, "done:", done, "Reward:", reward, "Obs:", observation)
    if done:
        epoch_count += 1
        print("Episode", epoch_count, "finished after", count, "timesteps")
        total_count += count
        count = 0
        observation = env.reset()

print("試行回数:{0}, 平均:{1}".format(epoch_count, total_count/epoch_count))
