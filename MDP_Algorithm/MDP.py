'''
Created on 09.07.2018

@author: Hussam Saoud
'''
import numpy as np
import random


class MDP:
    """ 
        Markov Decision Process
        
        params:
        States s 
        Actions a
        Rewards r(s,a)
        Transition Matrix t(s,a,s')

    """

    
    
    def __init__(self, states,transitions,actions,rewards):

        self.s = states
        self.a = actions
        self.r = rewards
        self.t = transitions
        
        # MDP params
        self.discount = 0.99
        self.tau = 0.01
        self.epsilon = 0.001


    def valueIteration(self):
        """
        process value iteration with the initialized parameters

        """

        # Initialize values 
        self.values = np.zeros(len(self.s))
        
        #  loop until it converges
        while True:
            
            # values of previous iteration for checking if it has converged 
            previous_values = np.copy(self.values)
            # loop over every state
            for i in range(len(self.s)):
                # get value of action a (going one state further or staying at the end state
                # and of action b (going back to state 0) 
                if i<len(self.s)-1:
                    temp=np.array([self.values[i+1],self.values[0]])
                else:
                    # if current state is final state 4 
                    temp=np.array([self.values[i],self.values[0]])
                    
                # bellman equation for calculating the values 
                # multiplying the transition matrix at state i with rewards matrix at same state i
                # ... and adding that to the previous values for action a and b, multiplied with discount, 
                #  and getting the maximum value out of the whole result
                self.values[i] = np.max(np.dot(self.t[i][:][:], self.r[i][:]) \
                                                    + self.discount *temp)
            # Check if values converged
            if np.max(np.abs(self.values - previous_values)) <= self.epsilon:
                break


    def extractPolicy(self):
        """
            extracting policy after value iteration
        """
        # initializing policy with zeros of the shape of the states s
        self.policy = np.zeros(len(self.s))
        
        # looping over every state 
        for i in range(len(self.s)):
            # init. temporal variable of the shape of the actions
            values=np.zeros([len(self.a)])
            
            # loop over both actions a and b
            for k in range(len(self.a)):
                # get value of action a (going one state further or staying at the end state
                # and of action b (going back to state 0) 
                if i<len(self.s)-1:
                    temp_state=np.array([self.values[i+1],self.values[0]])
                else:
                    temp_state=np.array([self.values[i],self.values[0]])
                # calculate values of all action at state i
                temp_policy=np.dot(self.t[i][:][:], self.r[i][:]) \
                                            + self.discount *temp_state
            # choose action with highest value and write to policy for state i
            self.policy[i]=np.argmax(temp_policy)
