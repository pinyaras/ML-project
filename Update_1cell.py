import gym
import numpy as np
from gym import spaces, envs
from envs.simulator import NetworkSimulatorEnv
from agents.q_agent import networkTabularQAgent
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from operator import truediv
import pandas as pd


####
# This script currently makes the agent use the a random policy to explore the state,action space
# then tests the value of the learning by every 1000 iterations using the best choice and printing the reward
####

def main():
    callmean = 1.0  # network load
    avg_delays = []
    for i in range(3):
        callmean += 1.0
        env = NetworkSimulatorEnv()
        state_pair = env._reset()
        env.callmean = callmean
        agent = networkTabularQAgent(env.nnodes, env.nedges, env.distance, env.nlinks)
        done = False
        r_sum_random = r_sum_best = 0
        rtrace = []
        steps = []
        avg_reward_random = []
        avg_reward_best = []
        avg_delay = []
        avg_route = []

        # Initialize Q-table from Q-agent
        config = agent.config

        for t in range(20001):
            if not done:

                '''

                state_pair = ((1, 19), (3, 19))
                current_state = (3, 19)
                n = 3
                dest = 19
                env.nlinks = {0: 2, 1: 3, 2: 2, 3: 2, 4: 3, 5: 2, 6: 3, 7: 4, 8: 3, 9: 3, 10: 4, 11: 3, 12: 3, 13: 4, 14: 3, 15: 3, 16: 4, 17: 3, 18: 3, 19: 4, 20: 4, 21: 4, 22: 4, 23: 3, 24: 2, 25: 2, 26: 2, 27: 2, 28: 2, 29: 2, 30: 2, 31: 2, 32: 2, 33: 2, 34: 2, 35: 2}
                env.nlinks[3] = 2

                '''

                # print("state1",state_pair)
                # ('state1', ((1, 32), (12, 32)))

                current_state = state_pair[1]
                # print(current_state)
                #     (92, 65)

                n = current_state[0]
                # print(n)
                #  92
                dest = current_state[1]
                # print(dest)
                # #  65
                # print(env.nlinks)
                # print(env.nlinks[n])
                rewards = []
                next_states = []
                min_dict = {}
                for action in xrange(env.nlinks[n]):

                    reward, next_state = env.pseudostep(action)
                    min_dict[next_state] = reward
                    #min(min_dict, key=min_dict.get)

                next_state = min(min_dict, key = min_dict.get)
                reward = min_dict.values()

                agent.learn(current_state, next_state, reward, action, done, env.nlinks)


                # greedy pick from q-table without observe next state
                action = agent.act(current_state, env.nlinks, True)
                # action = agent.act(current_state, env.nlinks, True)
                # reward, next_state = env.pseudostep(action)
                # agent.learn(current_state, next_state, reward, action, done, env.nlinks)
                # print(action)

                state_pair, reward, done, _ = env.step(action)

                # print("state2",state_pair)
                # ('state2', ((12, 32), (31, 25)))
                # S2
                # no a1?
                next_state = state_pair[0]
                agent.learn(current_state, next_state, reward, action, done, env.nlinks)
                # print("current_state: ", current_state, "action: ", "next_state:", next_state, "a1:", "reward: ", reward)
                # print(current_state)
                rewards.append(reward)
                r_sum_random += reward
                avg_reward_random.append(r_sum_random)

                avg_delay.append(float(env.total_routing_time))
                avg_route.append(float(env.routed_packets))

                # print "q learning with callmean:{} time:{}, average delivery time:{}, length of average route:{}, r_sum_best:{}, Packet routed".format(
                #      i, t, float(env.total_routing_time) ,
                #            float(env.total_hops), r_sum_best),float(env.routed_packets)

                if t % 20000 == 0:
                    # rtrace.append(np.sum(rewards))

                    if env.routed_packets != 0:
                        print "q learning with callmean:{} time:{}, average delivery time:{}, length of average route:{}, r_sum_random:{}".format(
                            i, t, float(env.total_routing_time) / float(env.routed_packets),
                                  float(env.total_hops) / float(env.routed_packets), r_sum_random)

                        #print("routing time", env.routing_time)

                        for i in range(len(avg_route)):
                            if avg_route[i] == 0.0:
                                avg_route[i] = 0.00000000000000000000000000000000000000001

                        avg_t = map(truediv, avg_delay, avg_route)

                        # my_dict = {'Time': steps, 'total_routing_time': avg_delay, 'routed_packets': avg_route, 'Avg delivery time':avg_t }
                        # trace = pd.DataFrame(my_dict)
                        # trace.to_csv('trace.csv')
                        #avg_t = np.diff(avg_t)
                        # print(avg_t)
                        rtrace.append(rewards)
                        avg_delays.append(r_sum_best)
                        x_rtrace = np.arange(0, len(avg_t), 1)
                        y_rtrace = np.array(avg_t)
                        plt.plot(x_rtrace, y_rtrace)
                        # plt.plot(rtrace)
                        plt.xlabel('Iterations')
                        plt.ylabel('avg delivery time in train')

                        plt.show()

                        # current_state = state_pair[1]
                        # n = current_state[0]
                        # dest = current_state[1]
                        #
                        # for action in xrange(env.nlinks[n]):
                        #     reward, next_state = env.pseudostep(action)
                        #     agent.learn(current_state, next_state, reward, action, done, env.nlinks)
                        # # Select best action
                        # action = agent.act(current_state, env.nlinks, True)
                        # state_pair, reward, done, _ = env.step(action)
                        #
                        # next_state = state_pair[0]
                        # agent.learn(current_state, next_state, reward, action, done, env.nlinks)
                        # r_sum_best += reward
                        # avg_delay.append(float(env.total_routing_time))
                        # avg_route.append(float(env.routed_packets))
                        #
                        #
                        # if env.routed_packets != 0:
                        #     print "Final q learning with callmean:{} time:{}, average delivery time:{}, length of average route:{}, r_sum_best:{}".format(
                        #         i, t, float(env.total_routing_time) / float(env.routed_packets),
                        #               float(env.total_hops) / float(env.routed_packets), r_sum_best)
                        #
                        #     for i in range(len(avg_route)):
                        #         if avg_route[i] == 0.0:
                        #             avg_route[i] = 1.0
                        #
                        #     avg_t = map(truediv, avg_delay, avg_route)
                        #     #avg_t = np.diff(avg_t)
                        #     # print(avg_t)
                        #     rtrace.append(rewards)
                        #     avg_delays.append(r_sum_best)
                        #     x_rtrace = np.arange(0, len(avg_t), 1)
                        #     y_rtrace = np.array(avg_t)
                        #     plt.plot(x_rtrace, y_rtrace)
                        #     # plt.plot(rtrace)
                        #     plt.xlabel('Iterations')
                        #     plt.ylabel('avg delivery time in test')
                        #
                        #     plt.show()


if __name__ == '__main__':
    main()