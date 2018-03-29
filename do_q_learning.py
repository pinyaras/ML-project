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

def main():
    callmean = 1.0 # network load
    avg_delays = []
    avg_delay = []
    for i in range(10):
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


        #Initialize Q-table from Q-agent
        config = agent.config

        for t in range(10001):
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
                action = agent.act(current_state, env.nlinks, True)

                state_pair, reward, done, _ = env.step(action)

                next_state = state_pair[0]
                agent.learn(current_state, next_state, reward, action, done, env.nlinks)
                # print("current_state: ", current_state, "action: ", "next_state:", next_state, "a1:", "reward: ", reward)
                # print(current_state)
                rewards.append(reward)
                r_sum_random += reward
                avg_reward_random.append(r_sum_random)

                # if t == 1:
                #     env.routed_packets = 1
                # steps.append(t + 1)
                # avg_delay.append(float(env.total_routing_time))
               # print(avg_delay)
                if t % 10000 == 0:
                    # rtrace.append(np.sum(rewards))
                    # avg_delay.append(float(env.total_routing_time) / float(env.routed_packets))

                    if env.routed_packets != 0:
                        print "q learning with callmean:{} time:{}, average delivery time:{}, length of average route:{}, r_sum_random:{}".format(
                            i, t, float(env.total_routing_time) / float(env.routed_packets),
                            float(env.total_hops) / float(env.routed_packets), r_sum_random)
                        # x_rtrace = np.arange(0, len(avg_delay), 1)
                        # y_rtrace = np.array(avg_delay)
                        # plt.plot(x_rtrace, y_rtrace)
                        # #    plt.plot(rtrace)
                        # plt.xlabel('Iterations')
                        # plt.ylabel('rewards')
                        #
                        # plt.show()
                        # print("state_r_sum_random", current_state)

                    current_state = state_pair[1]
                    n = current_state[0]
                    dest = current_state[1]

                    for action in xrange(env.nlinks[n]):
                        reward, next_state = env.pseudostep(action)
                        agent.learn(current_state, next_state, reward, action, done, env.nlinks)

                    action = agent.act(current_state, env.nlinks, True)
                    state_pair, reward, done, _ = env.step(action)

                    next_state = state_pair[0]
                    agent.learn(current_state, next_state, reward, action, done, env.nlinks)
                    r_sum_best += reward
                    # avg_delay.append(float(env.total_routing_time))

                    if env.routed_packets != 0:
                        print "q learning with callmean:{} time:{}, average delivery time:{}, length of average route:{}, r_sum_best:{}".format(
                            i, t, float(env.total_routing_time) / float(env.routed_packets),
                            float(env.total_hops) / float(env.routed_packets), r_sum_best)
                        avg_delay.append(float(env.total_routing_time) / float(env.routed_packets))

    #rtrace.append(rewards)
    #avg_delays.append(np.sum(avg_delay))
 #    print(avg_delay)
 #    x_rtrace = np.arange(0, len(avg_delay), 1)
 #    y_rtrace = np.array(avg_delay)
 #    plt.plot(x_rtrace, y_rtrace)
 # #    plt.plot(rtrace)
 #    plt.xlabel('Loads')
 #    plt.ylabel('Avg_time')
 # #
 #    plt.show()



if __name__ == '__main__':
    main()
