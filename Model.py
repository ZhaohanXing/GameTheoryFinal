import numpy as np
import networkx as nx

import random
import networkx as nx
import matplotlib.pyplot as plt


class infomation_and_platform:
    def __init__(self, satisfaction, disutility, info_uncertainty):
        # info_uncertainty is the theta, which is used for information value
        # v is the satisfaction from sharing
        self.satisfaction = satisfaction
        # b is the disutility from misinfomation
        self.disutility = disutility
        # epsilon is the misinformation level (Inaccuracy level)
        # self.inaccuracy = random.uniform(0, 1)
        self.inaccuracy = info_uncertainty
        # This is the value of information. Which is theta
        # self.info_value = np.random.normal(0, np.sqrt(info_uncertainty))


class Agent:
    def __init__(self, id, G):
        self.network_structure = G
        self.id = id
        self.utility = 0
        self.utility_history = []
        # Generate a random history of past 100 posts
        # init_share = random.randint(0, 25)
        # init_ignore = 25 - init_share
        self.choice_history = []  # Store the history of actions
        self.choice_last_time = np.random.choice([-1, 1])
        self.choice = np.random.choice([-1, 1])
        # miu is the belief of agents
        self.belief = np.random.normal(0, 1)
        self.belief_history = []

    def clear_history(self):
        self.utility_history = []
        self.belief_history = []
        self.choice_history = []

    def decide_action_intervention_exact(self, info, neighbors):
        # The platform does not give recommendation. Instead, give the inaccuracy value
        network_externality = sum([j.choice_last_time for j in neighbors])
        if len(neighbors) > 0:
            deviation = 1 / len(neighbors) * sum([(j.belief - self.belief) ** 2 for j in neighbors])
        else:
            deviation = 0
        share_utility = info.satisfaction - info.disutility * info.inaccuracy + network_externality - deviation
        not_share_utility = -deviation
        # not_share_utility = 0
        if share_utility > not_share_utility:
            self.choice_last_time = self.choice
            self.choice = 1
            self.choice_history.append(1)
            self.utility = share_utility
            self.utility_history.append(share_utility)
        else:
            self.choice_last_time = self.choice
            self.choice = -1
            self.choice_history.append(-1)
            self.utility = not_share_utility
            self.utility_history.append(not_share_utility)

    def decide_action_intervention(self, info, neighbors, threshold):
        network_externality = sum([j.choice_last_time for j in neighbors])
        if info.inaccuracy <= threshold:
            if len(neighbors) > 0:
                deviation = 1 / len(neighbors) * sum([(j.belief - self.belief) ** 2 for j in neighbors])
            else:
                deviation = 0
            # 0.5 is the expected inaccuracy (mean), which is used here since people do not know the exact value
            share_utility = info.satisfaction - info.disutility * 0.5 + network_externality - deviation
            not_share_utility = -deviation
            # not_share_utility = 0
            if share_utility > not_share_utility:
                self.choice_last_time = self.choice
                self.choice = 1
                self.choice_history.append(1)
                self.utility = share_utility
                self.utility_history.append(share_utility)
            else:
                self.choice_last_time = self.choice
                self.choice = -1
                self.choice_history.append(-1)
                self.utility = not_share_utility
                self.utility_history.append(not_share_utility)
        if info.inaccuracy > threshold:
            if len(neighbors) > 0:
                deviation = 1 / len(neighbors) * sum([(j.belief - self.belief) ** 2 for j in neighbors])
            else:
                deviation = 0
            # 0.5 is the expected inaccuracy (mean), which is used here since people do not know the exact value
            share_utility = info.satisfaction - info.disutility * (
                        (1 + threshold) / 2) + network_externality - deviation
            not_share_utility = -deviation
            # not_share_utility = 0
            if share_utility > not_share_utility:
                self.choice_last_time = self.choice
                self.choice = 1
                self.choice_history.append(1)
                self.utility = share_utility
                self.utility_history.append(share_utility)
            else:
                self.choice_last_time = self.choice
                self.choice = -1
                self.choice_history.append(-1)
                self.utility = not_share_utility
                self.utility_history.append(not_share_utility)

    def decide_action_no_intervention(self, info, neighbors):
        # The platform does not give anything. People know only the mean of inaccuracy
        network_externality = sum([j.choice_last_time for j in neighbors])
        if len(neighbors) > 0:
            deviation = 1 / len(neighbors) * sum([(j.belief - self.belief) ** 2 for j in neighbors])
        else:
            deviation = 0
        share_utility = info.satisfaction - info.disutility * 0.5 + network_externality - deviation
        not_share_utility = -deviation
        # not_share_utility = 0
        if share_utility > not_share_utility:
            self.choice_last_time = self.choice
            self.choice = 1
            self.choice_history.append(1)
            self.utility = share_utility
            self.utility_history.append(share_utility)
        else:
            self.choice_last_time = self.choice
            self.choice = -1
            self.choice_history.append(-1)
            self.utility = not_share_utility
            self.utility_history.append(not_share_utility)

    def update_belief_simple(self, keep_belief_beta, neighbors):
        # update idea of the agent
        # We try a simpler model here first
        # keep_belief_beta = 0.8 # the lower, the more influential the message update
        neighbors_beliefs = [j.belief for j in neighbors if j.choice == 1]
        self.belief_history.append(self.belief)
        if len(neighbors_beliefs) > 0:
            self.belief = (keep_belief_beta * self.belief +
                           (1 - keep_belief_beta) * np.mean([j.belief for j in neighbors if j.choice == 1]))
        else:
            self.belief = self.belief

# # Create a network of agents
# agents = [Agent(i) for i in range(num_agents)]
# G = nx.gnp_random_graph(num_agents, 0.2)
#
# # Add agents to graph nodes with the updated agents
# for i in range(num_agents):
#     G.nodes[i]['agent'] = agents[i]
#
# # Re-run the simulation with the updated Agent class and decision-making strategy
# for round in range(num_rounds):
#     print(f"Round {round + 1}")
#     rand = random.random()   # Determine the truth of the message based on probability
#     info_type = "true" if rand < prob_true else "false"
#     for i, agent in enumerate(agents):
#         neighbors = [G.nodes[n]['agent'] for n in G.neighbors(i)]
#         action = agent.decide_action(neighbors, check_cost, credibility_gain, credibility_loss, similarity_bonus,
#                                      nonsimilar_cost, prob_true, info_type)
#         if i == 1:
#             print(f"Agent {i}: Action = {action}, Utility = {agent.utility}")
#
# # Visualize the network
# nx.draw(G, with_labels=True)
# plt.show()
