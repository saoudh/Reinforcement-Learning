'''
Created on 09.07.2018

@author: Hussam Saoud
'''

import numpy as np
import random

class ChainDomain():

    """ 
          Chain Domain:

        Problem: This domain consists of six states and two actions
        a (=0) and b (=1). With probability 0.2, the agent “slips”and actually
        performs the opposite action. The optimal policy for this
        domain (assuming a discount factor of 0.99) is to do action
        "a" everywhere. However, learning algorithms can get
        trapped at the initial state, preferring to follow the
        "b"-loop to obtain a series of smaller rewards.

    """

    def __init__(self):
        """ 
            Initializing chain domain arrays for Q-Learning algorithm
        """
        self.end_state=4
        # init. states 0 to 4
        self.s = np.arange(5)

        # init actions a (=0) and b (=1)
        self.a = np.arange(2)

        # Initialize rewards for states 0 to 4 and actions a and b
        self.r=np.array([ [0,2],
       [0,2],
       [0,2],
       [0,2],
       [0,10]])
    
    # doing a step in the chain domain with given action and current state as parameters
    def step(self, action, current_state):
        done=False
        new_state=-1
        reward=-1
       
        # if action=a and not final state then reward=0 
        if action == 0 and current_state<4:
            reward=0
            new_state=current_state+1
            if current_state==4:
                done=True
        # if action=a and final state then reward=10
        elif action == 0 and current_state==4:
            new_state=4
            reward=10
            done=True
        # if action=b, then always reward=0 
        if action == 1:
            reward=2
            new_state=0
            done=False
        return new_state, reward, done
   

     

       

