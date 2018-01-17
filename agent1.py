import gym
env = gym.make('MsPacman-ram-v0')
for i_episode in range(20):
    observation = env.reset()
    for t in range(1000):
        env.render()
        action = env.action_space.sample()
        print(action)
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            break
