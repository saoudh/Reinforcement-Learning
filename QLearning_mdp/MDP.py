'''
Created on 09.07.2018

@author: admin
'''
import numpy as np
import random
from QLearning_mdp.Game import ChainDomain

class MDP(object):
    """ 
        Q-Learning Algorithm
        
        params:
        States s 
        Actions a
        Rewards r(s,a)

    """

    def __init__(self, states, actions, rewards):
        # Arrays and Q-Table matrix
        self.s = states
        self.a = actions
        self.r = rewards
        self.Q_table = np.zeros(np.shape(rewards))
        
        # parameters for q-learning
        self.explore_rate=0.7
        self.discount = 0.99
        self.tau = 0.01
        self.epsilon = 0.01
        self.lr=0.1
        self.episodes=100
        self.probab_of_actions=[0.8,0.2]

        # init. chain domain
        self.game=ChainDomain()
      
    # updating Q-table
    def update_q(self, state, action, reward, new_state):
            # Q-Learning formula
            self.Q_table[state][action] = (1-self.lr) * self.Q_table[state][action]+self.lr * \
                    (reward + self.discount * np.max(self.Q_table[new_state]))

    # choosing next action
    def choose_action(self, state):
        #for the rate explore_rate choose new action according to their probability 
        if (np.random.random() < self.explore_rate):
            # randomly sample explore_rate percent of the time
            #return action with a  certain probability 
            return int(np.random.choice(self.a,1,self.probab_of_actions))
        else:
            # ... otherwise choose according to the Q-table
            return np.argmax(self.Q_table[state])

    def train(self):
        for e in range(self.episodes):
            print("")
            print("episode:",e)
            # randomly select beginning state of new episode
            current_state=np.random.randint(0,4)
            done = False

            while not done:
                action = self.choose_action(current_state)
                new_state,reward,done=self.game.step(action,current_state)
                self.update_q(current_state, action, reward, new_state)
                current_state = new_state
            print("done")
                
    def reset(self):
        return np.random.randint(0,4)

        
  
    