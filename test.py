import numpy as np
num_agents = 10
num_rounds = 5
check_cost = 1
credibility_gain = 2
credibility_loss = 3
similarity_bonus = 1
prob_true = 0.7  # Probability that the message is true
aaa = np.array([credibility_gain + similarity_bonus,
                        credibility_gain + similarity_bonus,
                        credibility_gain]).reshape(1,3)
print(aaa.shape)