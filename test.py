import gym
from observation_preprocessing import preprocess_atari
import matplotlib.pyplot as plt
from atari_wrapper import AtariWrapper
import random
import time

env = AtariWrapper("PongDeterministic-v4")
env.reset()
done = False
while not done:
    action = random.randint(0, 3) # int(input())
    observation, reward, done, info = env.step(action)
    print(info)
    print("done: " + str(done) + " reward: " + str(reward))
    time.sleep(0.1)
    env.render()
