'''
Created on 09.07.2018

@author: Hussam Saoud
'''
from MDP_Algorithm.Game import ChainDomain
from MDP_Algorithm.MDP import MDP

# init. dhain domain class 
game=ChainDomain()

# get all MDP paramters after init. the chain domain 
states, transitions, actions,rewards = game.getValues()

# init. MDP with parameters
mdp = MDP(states=states,transitions=transitions,actions=actions,rewards=rewards)

# start value iteration and extracting the policy
mdp.valueIteration()
mdp.extractPolicy()

# printing values and policy
print("Values:",mdp.values)
print("Policy:",mdp.policy)