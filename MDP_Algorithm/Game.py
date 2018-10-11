'''
Created on 09.07.2018

@author: Hussam Saoud
'''
import numpy as np
import random
from MDP_Algorithm.MDP import MDP

class ChainDomain:

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
            Initializing MDP_Algorithm with the MDP Parameters for states s, actions a, 
            transitions t and rewards r
        """

        # Initialize states
        self.s = np.arange(5)

        # Initialize actions
        self.a = np.arange(2)

       # Initialize temporal transitions with zeros and shape of [length(states),length(actions)]
        temp=np.zeros([len(self.s),len(self.a)])
        # computer rewards and write array to r
        self.r=[self.computeRewards(i[0], i[1]) for i,x in np.ndenumerate(temp)]
        # reshape rewards array after computing
        self.r=np.reshape(self.r, np.shape(temp))
 
        # Initialize temporal transition matrix with zeros and shape of [length(states),length(actions]
        temp = np.zeros([len(self.s),len(self.a), 2])

        # compute and reshape transition matrix 
        self.t = [self.computeTransitions(i[0], i[1],i[2]) for i,x in np.ndenumerate(temp)] 
        self.t = np.reshape(self.t, np.shape(temp))
       
       
    def getValues(self):
        return self.s, self.t, self.a,self.r       

    def computeRewards(self, current_state, action):

        """ 
            computing the rewards matrix with probabilities

            Params:
            current_state 
            action
        """
        # if current state smaller then last state 4 and action is a then reward is 0
        if current_state<4 and action==0:
            return 0
        # if current state is last state 4 and action is a then reward is 10
        elif current_state==4 and action==0:
            return 10
        #if action is b then it gets always the same reward, no matter which current state
        elif action==1:
            return 2
     
      
            

    def computeTransitions(self, current_state, action, next_state):

        """ 
          computing the transition matrix with probabilities

            Params:
            current_state 
            action

        """
        #probability for taking action a (=0) and going forward
        if  action==0 and next_state==1:
            return 0.8
        #probability for taking action a (=0) but act like b and going to the wrong direction (going back)
        elif  action==0 and next_state==0:
            return 0.2
         #probability for taking action a (=0) and going back 
        elif  action==1 and next_state==0:
            return 0.8
         #probability for taking action a (=0) but act like a and going to the wrong direction
        elif  action==1 and next_state==1:
            return 0.2
      
      
            

