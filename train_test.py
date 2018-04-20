import gym
import numpy as np
from gym import spaces, envs
from envs.simulator import NetworkSimulatorEnv
from agents.q_agent import networkTabularQAgent

from operator import truediv

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# from ggplot import *


####
# This script currently makes the agent use the a random policy to explore the state,action space
# then tests the value of the learning by every 1000 iterations using the best choice and printing the reward
####


def main():
    callmean = 1.0 # network load
    avg_delays = []
    env = NetworkSimulatorEnv()
    state_pair = env._reset()
    agent = networkTabularQAgent(env.nnodes, env.nedges, env.distance, env.nlinks)

    for i in range(1):
        callmean += 1.0
        print(callmean)
        env.callmean = callmean
        done = False
        r_sum_random = r_sum_best = 0
        rtrace = []
        steps = []
        avg_reward_random = []
        avg_reward_best = []

        avg_delay = []
        avg_route = []



        #Initialize Q-table from Q-agent
        config = agent.config

        for t in range(50001):
            rewards = []
            if not done:
                #state
                current_state = state_pair[1]
                # print(current_state)
                n = current_state[0]
                # print(n)
                dest = current_state[1]
                # print(dest)

                #Select Random action from current node., update every neighbor node?
                #env.nlinks = num of neighbor nodes.
                #Observes and update the neighbor nodes reward
                #pseudostep searchs of X's neighbor, (node y)
                for action in xrange(env.nlinks[n]):
                    reward, next_state = env.pseudostep(action)
                    agent.learn(current_state, next_state, reward, action, done, env.nlinks)
                #Radnom action
                #action = agent.act(current_state, env.nlinks, True)
                action = agent.act_softmax(current_state, env.links)

                #action = agent.act_eps(current_state, env.nlinks, 0.1)

                state_pair, reward, done, _ = env.step(action)

                next_state = state_pair[0]
                agent.learn(current_state, next_state, reward, action, done, env.nlinks)
                # print("current_state: ", current_state, "action: ", "next_state:", next_state, "a1:", "reward: ", reward)
                # print(current_state)
                rewards.append(reward)
                r_sum_random += reward
                avg_reward_random.append(r_sum_random)

                avg_delay.append(float(env.total_routing_time))
                avg_route.append(float(env.routed_packets))

                # if t == 1:
                #     env.routed_packets = 1
                # steps.append(t + 1)
                # avg_delay.append(float(env.total_routing_time))
               # print(avg_delay)
                if t % 50000 == 0:
                    # rtrace.append(np.sum(rewards))
                    # avg_delay.append(float(env.total_routing_time) / float(env.routed_packets))

                    if env.routed_packets != 0:
                        print "q training with callmean:{} time:{}, average delivery time:{}, length of average route:{}, r_sum_random:{}".format(
                            i, t, float(env.total_routing_time) / float(env.routed_packets),
                            float(env.total_hops) / float(env.routed_packets), r_sum_random)

                        for x in range(len(avg_route)):
                            if avg_route[x] == 0.0:
                                avg_route[x] = 0.00000000000000000000000000000000000000001

                        avg_t = map(truediv, avg_delay, avg_route)
                        x_rtrace = np.arange(0, len(avg_t), 1)
                        y_rtrace = np.array(avg_t)

                        plt.plot(x_rtrace, y_rtrace)
                        plt.xticks(np.arange(min(x_rtrace), max(x_rtrace) + 1, 5000))
                        #    plt.plot(rtrace)
                        plt.xlabel('Iterations')
                        plt.ylabel('Avg time in Train')

                        plt.show()
                        break
                        # print("state_r_sum_random", current_state)
    #Test
    #env = NetworkSimulatorEnv()
    state_pair = env._reset()
    #agent = networkTabularQAgent(env.nnodes, env.nedges, env.distance, env.nlinks)
    done = False
    avg_delay_test = []
    avg_route_test = []
    r_sum_best = 0
    i = 1
    for t in range(100001):
        if not done:
            current_state = state_pair[1]
            n = current_state[0]
            dest = current_state[1]

            # for action in xrange(env.nlinks[n]):
            #     reward, next_state = env.pseudostep(action)
                #agent.learn(current_state, next_state, reward, action, done, env.nlinks)
            action = agent.act(current_state, env.nlinks, True)

            reward, next_state = env.pseudostep(action)


            action = agent.act(current_state, env.nlinks, True)
            state_pair, reward, done, _ = env.step(action)

            next_state = state_pair[0]
            #agent.learn(current_state, next_state, reward, action, done, env.nlinks)
            r_sum_best += reward
            # avg_delay.append(float(env.total_routing_time))
            avg_delay_test.append(float(env.total_routing_time))
            avg_route_test.append(float(env.routed_packets))
            #print("testing")

            if t % 100000 == 0:
                if env.routed_packets != 0:
                    print "q testing with callmean:{} time:{}, average delivery time:{}, length of average route:{}, r_sum_best:{}".format(
                        i, t, float(env.total_routing_time) / float(env.routed_packets),
                        float(env.total_hops) / float(env.routed_packets), r_sum_best)

                    for x in range(len(avg_route_test)):
                        if avg_route_test[x] == 0.0:
                            avg_route_test[x] = 0.00000000000000000000000000000000000000001

                    avg_t_test = map(truediv, avg_delay_test, avg_route_test)
                    #print(avg_t_test)
                    x_rtrace = np.arange(0, len(avg_t_test), 1)
                    y_rtrace = np.array(avg_t_test)
                    plt.plot(x_rtrace, y_rtrace)
                    #    plt.plot(rtrace)
                    plt.xlabel('Iterations')
                    plt.ylabel('Avg time in Test')

                    plt.show()
#    return avg_delay, avg_route, avg_route_tes, avg_route_test, avg_t, avg_t_test


def ploting(avg_delay, avg_route, avg_route_tes, avg_route_test, avg_t, avg_t_test):
    x_rtrace = np.arange(0, len(avg_t_test), 1)
    y_rtrace = np.array(avg_t_test)
    plt.plot(x_rtrace, y_rtrace)
    #    plt.plot(rtrace)
    ply.title("plot from functiun")
    plt.xlabel('Iterations')
    plt.ylabel('Avg time in Test')

    plt.show()



if __name__ == '__main__':
    main()
    #ploting(avg_delay, avg_route, avg_route_tes, avg_route_test, avg_t, avg_t_test)