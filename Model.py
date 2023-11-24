import numpy as np
import networkx as nx

import random
import networkx as nx
import matplotlib.pyplot as plt


class Agent:
    def __init__(self, id):
        self.id = id
        self.utility = 0
        # Generate a random history of past 100 posts
        init_share = random.randint(0, 5)
        init_check = random.randint(0, 5 - init_share)  # Ensure the sum of num1 and num2 is not more than 100
        init_ignore = 5 - init_share - init_check
        self.history = [init_share, init_check, init_ignore]  # Store the history of actions

    def decide_action(self, neighbors, check_cost, credibility_gain, credibility_loss, similarity_bonus, nonsimilar_cost, prob_true, info_type):
        """
        Decide action based on utility calculation and learning from the entire history of neighbors.
        """
        # Check neighbors' entire history to estimate message truth
        # Given the history of neighbors, the user will find the maximum expected utility

        # Find the strategy for each of the neighbors
        strategy_neighbor = []
        for n in neighbors:
            strategy_n = np.array(n.history) / sum(n.history)
            strategy_neighbor.append(strategy_n)
        strategy_neighbor = np.array(strategy_neighbor)
        # The player only knows that prob_true = 0.7, prob_false = 0.3
        expected_share = 0
        expected_check = 0
        expected_ignore = 0
        if len(neighbors) > 0:
            expected_utility_true_share = np.dot(strategy_neighbor,
                                                 np.array([credibility_gain + similarity_bonus,
                                                           credibility_gain + similarity_bonus,
                                                           credibility_gain - nonsimilar_cost])
                                                 ).sum()
            expected_utility_true_check = np.dot(strategy_neighbor
                                                 , np.array([credibility_gain + similarity_bonus - check_cost,
                                                             credibility_gain + similarity_bonus - check_cost,
                                                             credibility_gain - check_cost - nonsimilar_cost])
                                                 ).sum()
            expected_utility_true_ignore = np.dot(strategy_neighbor,
                                                 np.array([-nonsimilar_cost,
                                                           -nonsimilar_cost,
                                                           0])
                                                 ).sum()
            expected_utility_false_share = np.dot(strategy_neighbor
                                                  , np.array([-credibility_loss + similarity_bonus,
                                                              -credibility_loss - nonsimilar_cost,
                                                              -credibility_loss - nonsimilar_cost])
                                                  ).sum()
            expected_utility_false_check = np.dot(strategy_neighbor
                                                  , np.array([-check_cost - nonsimilar_cost,
                                                              -check_cost,
                                                              -check_cost])
                                                  ).sum()
            expected_utility_false_ignore = np.dot(strategy_neighbor,
                                                 np.array([-nonsimilar_cost,
                                                           0,
                                                           0])
                                                 ).sum()
            expected_share = prob_true * expected_utility_true_share + (1-prob_true) * expected_utility_false_share
            expected_check = prob_true * expected_utility_true_check + (1-prob_true) * expected_utility_false_check
            expected_ignore = prob_true * expected_utility_true_ignore + (1-prob_true) * expected_utility_false_ignore
        else:
            # No neighbors, the user choose strategy themselves
            expected_share = prob_true * credibility_gain + (1-prob_true) * credibility_loss
            expected_check = prob_true * (credibility_gain - check_cost) - (1-prob_true) * check_cost
            expected_ignore = 0
        if self.id == 1:
            print(expected_check, expected_share, expected_ignore)
        # Choose the action with the highest utility
        best_action = \
            max(('check', expected_check), ('share', expected_share), ('ignore', expected_ignore), key=lambda x: x[1])[
                0]
        #print(best_action)

        if best_action == "share":
            self.history[0] += 1
        elif best_action == "check":
            self.history[1] += 1
        else:
            self.history[2] += 1

        # Find the actual payoff of agents
        # if info_type == 'true':

        # # Adjust expected utility based on neighbor's history and probability of truth
        # expected_utility_true = neighbor_truth_estimate * credibility_gain
        # expected_utility_false = (1 - neighbor_truth_estimate) * credibility_loss
        #
        # # Calculate similarity bonus
        # similarity_utility = sum(similarity_bonus for n in neighbors if n.shared_last_round)
        #
        # # Combine utilities to make a decision
        # share_utility = expected_utility_true + expected_utility_false + similarity_utility
        # check_utility = -check_cost
        # ignore_utility = 0
        #
        # # Choose the action with the highest utility
        # best_action = \
        # max(('check', check_utility), ('share', share_utility), ('ignore', ignore_utility), key=lambda x: x[1])[0]
        #
        return best_action

    # def update_history_and_credibility(self, action, message_truth, credibility_gain, credibility_loss):
    #     # Update history
    #     self.history.append(action == 'share')
    #
    #     # Update credibility
    #     if action == 'share':
    #         if message_truth:
    #             self.credibility += credibility_gain
    #         else:
    #             self.credibility += credibility_loss  # Credibility should decrease if the message is false
    #
    #     self.shared_last_round = action == 'share'


# Update the simulation parameters
num_agents = 1000
num_rounds = 50
check_cost = 0.2
credibility_gain = 0.4
credibility_loss = 0.75
similarity_bonus = 0.2
nonsimilar_cost = 0.1
prob_true = 0.6 # Probability that the message is true

# Create a network of agents
agents = [Agent(i) for i in range(num_agents)]
G = nx.gnp_random_graph(num_agents, 0.2)

# Add agents to graph nodes with the updated agents
for i in range(num_agents):
    G.nodes[i]['agent'] = agents[i]

# Re-run the simulation with the updated Agent class and decision-making strategy
for round in range(num_rounds):
    print(f"Round {round + 1}")
    rand = random.random()   # Determine the truth of the message based on probability
    info_type = "true" if rand < prob_true else "false"
    for i, agent in enumerate(agents):
        neighbors = [G.nodes[n]['agent'] for n in G.neighbors(i)]
        action = agent.decide_action(neighbors, check_cost, credibility_gain, credibility_loss, similarity_bonus,
                                     nonsimilar_cost, prob_true, info_type)
        if i == 1:
            print(f"Agent {i}: Action = {action}, Utility = {agent.utility}")

# Visualize the network
# nx.draw(G, with_labels=True)
# plt.show()
