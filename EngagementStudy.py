import matplotlib.pyplot as plt

from Model import *

num_agents = 1000
# v
satisfaction = 0.5
# b
disutility = 1
iteration = 10
info_uncertainty = random.uniform(0, 1)
G = nx.erdos_renyi_graph(num_agents, 0.01)
# agents = [Agent(i, G) for i in range(num_agents)]

mapping = {i: Agent(i, G) for i in range(num_agents)}
G_no_intervention = nx.relabel_nodes(G, mapping)
G_intervention_threshold = nx.relabel_nodes(G, mapping)
G_intervention_givenvalue = nx.relabel_nodes(G, mapping)

# Track engagement rate
engagement_no_intervention = []
engagement_intervention_threshold = []
engagement_intervention_givenvalue = []

utility_no_intervention = []
utility_intervention_threshold = []
utility_intervention_givenvalue = []

simu_time = 30
for _ in range(simu_time):
    engagement_intervention_threshold_onesimu = []
    engagement_intervention_givenvalue_onesimu = []
    engagement_no_intervention_onesimu = []
    utility_no_intervention_onesimu = []
    utility_intervention_threshold_onesimu = []
    utility_intervention_givenvalue_onesimu = []
    for i in range(iteration):
        # print(f"Round {round + 1}")
        info = infomation_and_platform(satisfaction, disutility, random.uniform(0, 1))
        engage_user_no_intervention = 0
        engage_user_intervention_threshold = 0
        engage_user_intervention_givenvalue = 0
        utility_track_no_intervention = []
        utility_track_intervention_threshold = []
        utility_track_intervention_givenvalue = []
        for agent in G_no_intervention.nodes:
            neighbors = G_no_intervention[agent]
            agent.decide_action_no_intervention(info, neighbors)
            utility_track_no_intervention.append(agent.utility)
            agent.update_belief_simple(0.8, neighbors)
            if agent.choice == 1:
                engage_user_no_intervention += 1
        engagement_no_intervention_onesimu.append(engage_user_no_intervention/num_agents)
        for agent in G_intervention_threshold.nodes:
            neighbors = G_intervention_threshold[agent]
            agent.decide_action_intervention(info, neighbors, 0.8)
            utility_track_intervention_threshold.append(agent.utility)
            agent.update_belief_simple(0.8, neighbors)
            if agent.choice == 1:
                engage_user_intervention_threshold += 1
        engagement_intervention_threshold_onesimu.append(engage_user_intervention_threshold/num_agents)
        for agent in G_intervention_givenvalue.nodes:
            neighbors = G_no_intervention[agent]
            agent.decide_action_intervention_exact(info, neighbors)
            utility_track_intervention_givenvalue.append(agent.utility)
            agent.update_belief_simple(0.8, neighbors)
            if agent.choice == 1:
                engage_user_intervention_givenvalue += 1
        engagement_intervention_givenvalue_onesimu.append(engage_user_intervention_givenvalue/num_agents)
        utility_no_intervention_onesimu.append(utility_track_no_intervention)
        utility_intervention_threshold_onesimu.append(utility_track_intervention_threshold)
        utility_intervention_givenvalue_onesimu.append(utility_track_intervention_givenvalue)

        average_utility_no_intervention_onesimu = np.mean(utility_no_intervention_onesimu, axis=1)
        average_utility_intervention_threshold_onesimu = np.mean(utility_intervention_threshold_onesimu, axis=1)
        average_utility_intervention_givenvalue_onesimu = np.mean(utility_intervention_givenvalue_onesimu, axis=1)

    engagement_no_intervention.append(engagement_no_intervention_onesimu)
    engagement_intervention_threshold.append((engagement_intervention_threshold_onesimu))
    engagement_intervention_givenvalue.append(engagement_intervention_givenvalue_onesimu)

    utility_no_intervention.append(utility_no_intervention_onesimu)
    utility_intervention_threshold.append(utility_intervention_threshold_onesimu)
    utility_intervention_givenvalue.append(utility_intervention_givenvalue_onesimu)
    print(f"simutime {_} finish")

engagement_no_intervention = np.array(engagement_no_intervention)
engagement_intervention_threshold= np.array(engagement_intervention_threshold)
engagement_intervention_givenvalue= np.array(engagement_intervention_givenvalue)

average_engagement_no_intervention = np.mean(engagement_no_intervention, axis=0)
average_engagement_intervention_threshold = np.mean(engagement_intervention_threshold, axis=0)
average_engagement_intervention_givenvalue = np.mean(engagement_intervention_givenvalue, axis=0)


utility_no_intervention = np.array(utility_no_intervention)
utility_intervention_threshold = np.array(utility_intervention_threshold)
utility_intervention_givenvalue = np.array(utility_intervention_givenvalue)

average_utility_no_intervention = np.mean(utility_no_intervention, axis=(0, 2))
average_utility_intervention_threshold = np.mean(utility_intervention_threshold, axis=(0, 2))
average_utility_intervention_givenvalue = np.mean(utility_intervention_givenvalue, axis=(0, 2))



plt.plot(list(range(1, iteration+1)), average_engagement_no_intervention, label='No Intervention')
plt.plot(list(range(1, iteration+1)), average_engagement_intervention_threshold, label=f'$\sigma$=0.8')
plt.plot(list(range(1, iteration+1)), average_engagement_intervention_givenvalue, label=f'Given Exact Value $\epsilon$')
plt.xlabel('Iteration')
plt.ylabel('Engagement Rate')
plt.legend()
plt.title('Engagement rate as time increases')
plt.show()
#
#
# average_utility_no_intervention = np.mean(utility_no_intervention, axis=1)
# average_utility_intervention_threshold = np.mean(utility_intervention_threshold, axis=1)
# average_utility_intervention_givenvalue = np.mean(utility_intervention_givenvalue, axis=1)
#
plt.plot(list(range(1, iteration+1)), average_utility_no_intervention, label='No Intervention')
plt.plot(list(range(1, iteration+1)), average_utility_intervention_threshold, label=f'$\sigma$=0.8')
plt.plot(list(range(1, iteration+1)), average_utility_intervention_givenvalue, label=f'Given Exact Value $\epsilon$')
plt.xlabel('Iteration')
plt.ylabel('Average Utility')
plt.legend()
plt.title('Social Network Average Utility')
plt.show()
# for agent in G.nodes:
#     print(f"Agent {agent.id}: Choice History = {agent.choice_history}, \n "
#           f"Utility History = {agent.utility_history}, \n"
#           f"Belief History = {agent.belief_history}, \n")
