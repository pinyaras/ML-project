import collections
import numpy as np
from random import random
from gym import spaces, envs
from envs.simulator import NetworkSimulatorEnv



class networkTabularQAgent(object):
    """
    Agent implementing tabular Q-learning for the NetworkSimulatorEnv.
    """

    def __init__(self, num_nodes, num_actions, distance, nlinks):
        self.config = {
            "init_mean" : 0.0,      # Initialize Q values with this mean
            "init_std" : 0.0,       # Initialize Q values with this standard deviation
            "learning_rate" : 0.7,
            "eps": 0.1,            # Epsilon in epsilon greedy policies
            "discount": 1,
            "n_iter": 10000000}        # Number of iterations
        self.q = np.zeros((num_nodes,num_nodes,num_actions))


        for src in range(num_nodes):
            for dest in range(num_nodes):
                for action in range(nlinks[src]):
                    self.q[src][dest][action] = distance[src][dest]

    # def epsilon_greed(self, epsilon, s):
    #     #
    #     if np.random.rand() < epsilon:
    #         return np.random.randint(self.n_a)
    #     else:
    #         print("greedy! selected")
    #         return np.argmax(self.Q[s[0], s[1], :])

#choose action
    def act(self, state, nlinks,  best=False):
        n = state[0]
        dest = state[1]
# choose the smallest tx among neighbors
        if best is True:
            best = self.q[n][dest][0]
            best_action = 0
#Find the minimum action value, greedy
            for action in range(nlinks[n]):
                if self.q[n][dest][action] < best:  #+ eps:
                    best = self.q[n][dest][action]
                    best_action = action
        else: #select action from random neighbor
            best_action = int(np.random.choice((0.0, nlinks[n])))

        return best_action


    def learn(self, current_event, next_event, reward, action, done, nlinks):


        n = current_event[0]
        dest = current_event[1]

        n_next = next_event[0]
        dest_next = next_event[1]

        #future = np.max(self.q[s1[0], s1[1], :]
        # future = np.argmax(self.q[n_next][dest][:])
        future = self.q[n_next][dest][0]


#	find the minimum value among the next node's neighbor
        for link in range(nlinks[n_next]):
            if self.q[n_next][dest][link] < future:
                future = self.q[n_next][dest][link]

        #Q learning
        self.q[n][dest][action] += (reward + self.config["discount"]*future - self.q[n][dest][action])* self.config["learning_rate"]
	#print("Q_Matrix_state: ", self.q[n,dest,:])


#     def train(self, current_event, next_event, reward, action, done, nlinks):
#
#         print(self.q)
#         print(self.q.shape)
#
# agent = networkTabularQAgent(env)
# start = [0, 0]
# # increase epsilon to explore more
# rtrace, steps, trace = agent.train(start,
#                                    gamma=0.99,
#                                    alpha=0.9,
#                                    epsilon=0.1,
#                                    maxiter=100,
#                                    maxstep=1000)

        # callmean = 1.0  # network load
        # i = 1
        #
        # env = NetworkSimulatorEnv()
        # #get current state
        # state_pair = env._reset()
        # env.callmean = callmean
        #
        # done = False
        #
        # for t in range(10001):
        #
        #     if not done:
        #         rewards = []
        #         current_state = state_pair[1]
        #         # print(current_state)
        #         n = current_state[0]
        #         # print(n)
        #         dest = current_state[1]
        #         # print(dest)
        #
        #         for action in xrange(env.nlinks[n]):
        #             reward, next_state = env.pseudostep(action)
        #             agent.learn(current_state, next_state, reward, action, done, env.nlinks)
        #
        #         action = agent.act(current_state, env.nlinks)
        #         state_pair, reward, done, _ = env.step(action)
        #
        #         next_state = state_pair[0]
        #         agent.learn(current_state, next_state, reward, action, done, env.nlinks)
        #         # print("current_state: ", current_state, "action: ", "next_state:", next_state, "a1:", "reward: ", reward)
        #         # print(current_state)
        #         r_sum_random += reward
        #         rewards.append(reward)
        #
        #         if t % 10000 == 0:
        #             # rtrace.append(np.sum(rewards))
        #
        #             if env.routed_packets != 0:
        #                 print "q learning with callmean:{} time:{}, average delivery time:{}, length of average route:{}, r_sum_random:{}".format(
        #                     i, t, float(env.total_routing_time) / float(env.routed_packets),
        #                           float(env.total_hops) / float(env.routed_packets), r_sum_random)
        #
        #
        #
        #             current_state = state_pair[1]
        #             n = current_state[0]
        #             dest = current_state[1]