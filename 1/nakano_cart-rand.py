import gym
env = gym.make('CartPole-v0')
observation = env.reset()
count = 0
epoch_count = 0
total_count = 0
while epoch_count < 20:
    env.render() # 現在の状況を画面表示する
    # ランダムに動かす
    observation, reward, done, info = env.step(env.action_space.sample())
    count += 1
    print("Step:",count,done,"Reward:",reward,"Obs:",observation)
    if done:
        epoch_count += 1
        print("Episode",epoch_count,"finished after",count,"timesteps")
        total_count += count
        count = 0
        observation = env.reset()

print("試行回数:{0}, 平均:{1}".format(epoch_count, total_count/epoch_count))
