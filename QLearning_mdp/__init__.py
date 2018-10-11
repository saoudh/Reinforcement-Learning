'''
Created on 09.07.2018

@author: Hussam Saoud
'''

from QLearning_mdp.Game import ChainDomain
from QLearning_mdp.MDP import MDP

game = ChainDomain()

# get all MDP paramters after init. the chain domain 
states, actions,rewards = game.s, game.a,game.r

# init. MDP with parameters
mdp = MDP(states,actions,rewards)

# start value iteration and extracting the policy
mdp.train()

# printing values and policy
print("Q-Values:",mdp.Q_table)
