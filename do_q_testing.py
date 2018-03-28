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
    # for i in range(10):
    #     callmean += 1.0
    i=1
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
            r_sum_random += reward
            rewards.append(reward)
            avg_reward_random.append(r_sum_random)
            rtrace.append(np.sum(rewards))
            steps.append(t + 1)
            if t % 10000 == 0:

                if env.routed_packets != 0:
                    print "q learning with callmean:{} time:{}, average delivery time:{}, length of average route:{}, r_sum_best:{}".format(
                        i, t, float(env.total_routing_time) / float(env.routed_packets),
                        float(env.total_hops) / float(env.routed_packets), r_sum_best)
                        # print("state_r_sum_best", current_state)
                    fig = plt.figure(figsize=(15, 15))
                    ax = fig.add_subplot(211)
                    ax.set_title('r_sum_random')
                    plt.xlabel("Iteration")
                    plt.ylabel("Random_rewards")
                    x_rtrace = np.arange(0, len(avg_reward_random), 1)
                    y_rtrace = np.array(avg_reward_random)
                    plt.plot(x_rtrace, y_rtrace)

                    ax1 = fig.add_subplot(212)
                    ax1.set_title('Sum of rewards')
                    plt.xlabel("Iteration")
                    plt.ylabel("Sum of rewards")
                    x_rtrace = np.arange(0, len(rtrace), 1)
                    y_rtrace = np.array(rtrace)
                    plt.plot(x_rtrace, y_rtrace)
                    plt.show()
        # plt.plot(avg_reward_random)
        # plt.xlabel('Iterations')
        # plt.ylabel('rewards')
        # plt.show()
        # rewards.append(reward)
        # print(rewards)
        # print("Final: \n", agent.q)
        #print("Policy: \n", np.argmax(agent.q, axis=2))
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


if __name__ == '__main__':
    main()
