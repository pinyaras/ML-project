import gym

env = gym.make("Taxi-v2")
env.reset()
env.observation_space.n
env.render()
env.action_space.n
env.env.s = 114
env.render()
env.step(1)
env.render()
(14, -1, False, {'prob': 1.0})
state = env.reset()
counter = 0
reward = None
while reward != 20:
    state, reward, done, info = env.step(env.action_space.sample())
    counter += 1

print(counter)


