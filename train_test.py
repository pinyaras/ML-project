import gym
import numpy as np
from gym import spaces, envs
from envs.simulator import NetworkSimulatorEnv
from agents.q_agent import networkTabularQAgent
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


####
# This script currently makes the agent use the a random policy to explore the state,action space
# then tests the value of the learning by every 1000 iterations using the best choice and printing the reward
####




if __name__ == '__main__':
    env = NetworkSimulatorEnv()
    agent = networkTabularQAgent(env.nnodes, env.nedges, env.distance, env.nlinks)
    print("strat")
    agent.train()