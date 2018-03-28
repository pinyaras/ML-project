import gym
import numpy as np
from gym import spaces, envs
from envs.simulator import NetworkSimulatorEnv
from agents.sarsa_agent import networkTabularSARSAAgent
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from operator import truediv
####
# This script currently makes the agent use the a random policy to explor the state,action space
# then tests the value of the learning by every 1000 iterations using the best choice and printing the reward
####

def main():
    callmean = 1.0
    for i in range(10):
        callmean += 1.0
        env = NetworkSimulatorEnv()
        state_pair = env._reset()
        env.callmean = callmean
        agent = networkTabularSARSAAgent(env.nnodes, env.nedges, env.distance, env.nlinks)
        done = False
        r_sum_random = r_sum_best = 0
        config = agent.config
        avg_delay = []
        avg_route = []

        for t in range(10001):
            if not done:

                current_state = state_pair[1]
                n = current_state[0]
                dest = current_state[1]

                for action in xrange(env.nlinks[n]):
                    reward, next_state = env.pseudostep(action)
                    agent.learn(current_state, next_state, reward, action, done, env.nlinks)

                action  = agent.act(current_state, env.nlinks, True)
                state_pair, reward, done, _ = env.step(action)

                next_state = state_pair[0]
                agent.learn(current_state, next_state, reward, action, done, env.nlinks)
                r_sum_random += reward

                avg_delay.append(float(env.total_routing_time))
                avg_route.append(float(env.routed_packets))

                if t%10000 == 0:

                    if env.routed_packets != 0:
                        print "sarsa learning with callmean:{} time:{}, average delivery time:{}, length of average route:{}, r_sum_random:{}".format(i, t, float(env.total_routing_time)/float(env.routed_packets), float(env.total_hops)/float(env.routed_packets), r_sum_random)

                        #avg_delays.append(r_sum_best)
                        for i in range(len(avg_route)):
                            if avg_route[i] == 0.0:
                                avg_route[i] = 1.0

                        avg_t = map(truediv, avg_delay, avg_route)
                        x_rtrace = np.arange(0, len(avg_t), 1)
                        y_rtrace = np.array(avg_t)
                        plt.plot(x_rtrace, y_rtrace)
                        # plt.plot(rtrace)
                        plt.xlabel('Iterations')
                        plt.ylabel('avg delivery time in train')

                        plt.show()




                    current_state = state_pair[1]
                    n = current_state[0]
                    dest = current_state[1]

                    for action in xrange(env.nlinks[n]):
                        reward, next_state = env.pseudostep(action)
                        agent.learn(current_state, next_state, reward, action, done, env.nlinks)

                    action  = agent.act(current_state, env.nlinks, True)
                    state_pair, reward, done, _ = env.step(action)

                    next_state = state_pair[0]
                    agent.learn(current_state, next_state, reward, action, done, env.nlinks)
                    r_sum_best += reward

                    if env.routed_packets != 0:
                        print "sarsa learning with callmean:{} time:{}, average delivery time:{}, length of average route:{}, r_sum_best:{}".format(i, t, float(env.total_routing_time)/float(env.routed_packets), float(env.total_hops)/float(env.routed_packets), r_sum_best)



if __name__ == '__main__':
    main()
