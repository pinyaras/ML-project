import collections
import numpy as np
from random import random
from gym import spaces, envs
from envs.simulator import NetworkSimulatorEnv
import matplotlib.pyplot as plt
from operator import truediv


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
                    #Initialize the reward with distance of shortestpath.
                    #self.q[src][dest][action] = distance[src][dest]
                    self.q[src][dest][action] = 0.01


    # def epsilon_greed(self, epsilon, s):
    #     #
    #     if np.random.rand() < epsilon:
    #         return np.random.randint(self.n_a)
    #     else:
    #         print("greedy! selected")
    #         return np.argmax(self.Q[s[0], s[1], :])

    # choose action
    def act_eps(self, state, nlinks, epsilon):
        n = state[0]
        dest = state[1]
        # choose the smallest tx among neighbors
        if np.random.rand() > epsilon:
            best = self.q[n][dest][0]
            best_action = 0
            # Find the minimum action value, greedy
            for action in range(nlinks[n]):
                if self.q[n][dest][action] < best:  # + eps:
                    best = self.q[n][dest][action]
                    best_action = action
        else:  # select random action from random neighbor
            best_action = int(np.random.choice((0.0, nlinks[n])))

        return best_action
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
        #print(best_action)

        return best_action

    def act_softmax(self, state, nlinks):
        n = state[0]
        dest = state[1]
        #print("state",state)
        # choose the smallest tx among neighbors
        #print("nlinks",nlinks[n])
        all_actions = np.array(self.q[n][dest][:len(nlinks[n])])
        all_actions = -1*np.array(all_actions)
        #print("all_action",all_actions)
        prob = self.softmax(all_actions)
        #print("prob",prob)
        #select = np.random.choice(SM)
        #pvals = [0.10, 0.25, 0.60, 0.05]
        #select = np.random.multinomial(1, prob)[0][0]
        best_action = np.where(np.random.multinomial(1, prob))[0][0]
        #print("select",select)
        #best_action = SM.index(select)
        #best_action = np.where(prob == select)[0][0]
        #print(best_action)
        return best_action

    # def softmax(self, z):
    #     return np.exp(-z) / np.sum(np.exp(-z), axis=1, keepdims=True)

    def softmax(self, x):

        '''Compute softmax values of array x.

        @param x the input array, vector
        @return the softmax array
        '''
        return np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x)))
        #return np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x)))

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


    def train(self ):
        env = NetworkSimulatorEnv()
        callmean = 1.0
        i=1
        max_t = 10000
        env.callmean = callmean
        done = False
        state_pair = env._reset()
        config = self.config

        r_sum_random =0
        avg_delay = []
        avg_route = []
        for t in range(max_t + 1):
            if not done:
                current_state = state_pair[1]
                n = current_state[0]
                dest = current_state[1]

                # for action in xrange(env.nlinks[n]):
                #     reward, next_state = env.pseudostep(action)
                #     self.learn(current_state, next_state, reward, action, done, env.nlinks)

                action = self.act(current_state, env.nlinks)
                state_pair, reward, done, _ = env.step(action)

                next_state = state_pair[0]
                self.learn(current_state, next_state, reward, action, done, env.nlinks)
                r_sum_random += reward

                avg_delay.append(float(env.total_routing_time))
                avg_route.append(float(env.routed_packets))


                if t % max_t == 0:

                    if env.routed_packets != 0:
                        print "q learning with callmean:{} time:{}, average delivery time:{}, length of average route:{}, r_sum_random:{}".format(
                            i, t, float(env.total_routing_time) / float(env.routed_packets),
                                  float(env.total_hops) / float(env.routed_packets), r_sum_random)
                        for i in range(len(avg_route)):
                            if avg_route[i] == 0.0:
                                avg_route[i] = 0.00000000000000000000000000000000000000001

                        avg_t = map(truediv, avg_delay, avg_route)

                        # my_dict = {'Time': steps, 'total_routing_time': avg_delay, 'routed_packets': avg_route, 'Avg delivery time':avg_t }
                        # trace = pd.DataFrame(my_dict)
                        # trace.to_csv('trace.csv')
                        # avg_t = np.diff(avg_t)
                        # print(avg_t)
                        # rtrace.append(rewards)
                        # avg_delays.append(r_sum_best)
                        x_rtrace = np.arange(0, len(avg_t), 1)
                        y_rtrace = np.array(avg_t)
                        plt.plot(x_rtrace, y_rtrace)
                        # plt.plot(rtrace)
                        plt.xlabel('Iterations')
                        plt.ylabel('avg delivery time in train')

                        plt.show()

