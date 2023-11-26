import matplotlib.pyplot as plt
import networkx as nx

from Model import *

num_agents = 1000
# v
satisfaction = 1
# b
disutility = 2
# epsilon
info_uncertainty = 0.7
G = nx.erdos_renyi_graph(num_agents, 0.4)
# agents = [Agent(i, G) for i in range(num_agents)]

mapping = {i: Agent(i, G) for i in range(num_agents)}
G = nx.relabel_nodes(G, mapping)
round = 10
# Track engagement rate
engagement = []

for round in range(round):
    # print(f"Round {round + 1}")
    info = infomation_and_platform(satisfaction, disutility, info_uncertainty)
    engage_user = 0
    for agent in G.nodes:
        neighbors = G[agent]
        agent.decide_action_no_intervention(info, neighbors)
        agent.update_belief_simple(0.8, neighbors)
        if agent.choice == 1:
            engage_user += 1
    engagement.append(engage_user/num_agents)


plt.plot(list(range(1, round+2)), engagement)
plt.xlabel('Iteration')
plt.ylabel('Engagement Rate')
plt.title('Engagement rate as time increases')
plt.show()

for agent in G.nodes:
    print(f"Agent {agent.id}: Choice History = {agent.choice_history}, \n "
          f"Utility History = {agent.utility_history}, \n"
          f"Belief History = {agent.belief_history}, \n")
