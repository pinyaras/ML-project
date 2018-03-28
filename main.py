import gym
import numpy as np
np.set_printoptions(threshold='nan')

from gym import spaces, envs
from envs.simulator import NetworkSimulatorEnv
from agents.q_agent import networkTabularQAgent
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import style

style.use('ggplot')


####
# This script currently makes the agent use the a random policy to explore the state,action space
# then tests the value of the learning by every 1000 iterations using the best choice and printing the reward
####

def main():
    callmean = 1.0 # network load
    for i in range(1):
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
        config = agent.config

        for t in range(10001):
            rewards = []
            if not done:

                current_state = state_pair[1]
                # print(current_state)
                n = current_state[0]
                # print(n)
                dest = current_state[1]
                # print(dest)

                for action in xrange(env.nlinks[n]):
                    reward, next_state = env.pseudostep(action)
                    agent.learn(current_state, next_state, reward, action, done, env.nlinks)
                action = agent.act(current_state, env.nlinks)

                # print "Qx --- time:{}, state_pair:{}, current_state:{}, n:{}, dest:{}, env.nlinks[n]:{}, action:{}, q+s:{}, packet_count:{}".format(
                #     t, state_pair, current_state, n, dest, env.nlinks[n], action, reward, env.count_packet)


                state_pair, reward, done, _ = env.step(action)

                next_state = state_pair[0]
                n = next_state[0]
                # print(n)
                dest = next_state[1]
                agent.learn(current_state, next_state, reward, action, done, env.nlinks)
                env.routing_time.append(float(env.temp))

                #env.temp = 0.0

                # print("current_state: ", current_state, "action: ", "next_state:", next_state, "a1:", "reward: ", reward)
                # print(current_state)

                # print "Qy --- time:{}, state_pair:{}, next_state:{}, n:{}, dest:{}, T's:{}, packet_count:{} ".format(
                #     t, state_pair, next_state, n, dest, reward, env.count_packet)
                # print("----------------------------------------------------------------------------------------------------------------------------------------")
                r_sum_best += reward
                avg_reward_random.append(reward)





                if t % 10000 == 0:
                    # rtrace.append(np.sum(rewards))

                    if env.routed_packets != 0:
                        print "q learning with callmean:{} time:{}, average delivery time:{}, length of average route:{}, r_sum_best:{}".format(
                            i, t, float(env.total_routing_time) / float(env.routed_packets),
                            float(env.total_hops) / float(env.routed_packets), r_sum_best)
                        #print(env.routing_time)
                        #print(agent.q.shape)
                        # print(agent.q)

                        # print("state_r_sum_random", current_state)
                    #
                    # current_state = state_pair[1]
                    # n = current_state[0]
                    # dest = current_state[1]


                    #
                        # plt.plot(env.routing_time)
                        # plt.xlabel('Iterations')
                        # plt.ylabel('routing_time')
                        # plt.show()
                        x_rtrace = np.arange(0, len(env.avg_lists), 1)
                        y_rtrace = np.array(env.avg_lists)
                        plt.plot(x_rtrace, y_rtrace)
                        # plt.plot(rtrace)
                        plt.xlabel('Iterations')
                        plt.ylabel('avg delivery time in train')

                        plt.show()
                    # rewards.append(reward)
                    # print(rewards)
                    # print("Final: \n", agent.q)
                    # print("Policy: \n", np.argmax(agent.q, axis=2))
        '''x = np.arange(0, len(rewards), 1)
        y = np.array(rewards)
        plt.plot(x, y)

        plt.xlabel('Episodes')
        plt.ylabel('Sum of rewards during episode')
        plt.title('The Q-routing reward')
      #  plt.grid(True)
        plt.show()

#	print(agent.q.shape)
#	print(agent.q)
'''
# def main():
#     callmean = 1.0 # network load
#     for i in range(10):
#         callmean += 1.0
#         env = NetworkSimulatorEnv()
#         state_pair = env._reset()
#         env.callmean = callmean
#         agent = networkTabularQAgent(env.nnodes, env.nedges, env.distance, env.nlinks)
#         done = False
#         r_sum_random = r_sum_best = 0
#         rtrace = []
#         steps = []
#         avg_reward_random = []
#         config = agent.config
#
#         for t in range(10001):
#             rewards = []
#             if not done:
#
#                 current_state = state_pair[1]
#                 # print(current_state)
#                 n = current_state[0]
#                 # print(n)
#                 dest = current_state[1]
#                 # print(dest)
#
#                 for action in xrange(env.nlinks[n]):
#                     reward, next_state = env.pseudostep(action)
#                 agent.learn(current_state, next_state, reward, action, done, env.nlinks)
#
#                 action = agent.act(current_state, env.nlinks, True)
#                 state_pair, reward, done, _ = env.step(action)
#
#                 next_state = state_pair[0]
#                 agent.learn(current_state, next_state, reward, action, done, env.nlinks)
#                 # print("current_state: ", current_state, "action: ", "next_state:", next_state, "a1:", "reward: ", reward)
#                 # print(current_state)
#                 r_sum_best += reward
#                 avg_reward_random.append(reward)
#                 steps.append(t + 1)
#                 if t % 10000 == 0:
#                     # rtrace.append(np.sum(rewards))
#
#                     if env.routed_packets != 0:
#                         print "q learning with callmean:{} time:{}, average delivery time:{}, length of average route:{}, r_sum_random:{}".format(
#                             i, t, float(env.total_routing_time) / float(env.routed_packets),
#                             float(env.total_hops) / float(env.routed_packets), r_sum_best)
#                         print(agent.q.shape)



if __name__ == '__main__':
    main()
