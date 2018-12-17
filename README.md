# Reinforcement-Learning
## 1 MDP - Markov Decision Process 
A simple Implementation of MDP in Python 

## 2 Q-Learning 
Implementation of the Q-Learning Algorithm 

## 3 DDPG
This implementation is based on the paper CONTINUOUS CONTROL WITH DEEP REINFORCEMENT LEARNING of Timothy P. Lillicrap et al., 2015 https://arxiv.org/pdf/1509.02971.pdf

### 3.1 Evaluation
The error of the policy is depicted as Actor- Error and the error of the Value-Function as Critic-Error. Also the Value-Function and reward (both, during training and evaluation) curve are plotted too.
Parameters. The neural network has 25 neurons and 1 layer and runs over 100 episodes. Batch Normalization is used too for stabilization. 

### 3.2 Performance 
The algorithm converges after 60 episodes with only 25 neurons. With higher number of neurons like 150 neurons it will converge much faster. 

![pendulum performance](https://github.com/saoudh/Reinforcement-Learning/blob/master/DDPG-master/screenshots/pendulum-performance.png)

### 3.3 Build
The algorithm can be executed by running ddpg.py with python3.

```
python3 -m rlrunner.ddpg
```

The requirements are:

```
gym
tensorflow
```
