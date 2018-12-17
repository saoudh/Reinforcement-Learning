import platform
print(platform.python_version())

import datetime
import matplotlib.pyplot as plt

import gym

import pprint
from matplotlib import rc
from rllab import NN, DDPG, RLBuffer
from rllab.utils import *

rc('text', usetex=True)

###########################################
# Import Default Settings
###########################################
config = getJsonDataFromConfigFile("ddpg_with_bn_1_layer.json")
pprint.pprint(config)

# Create the environment
ENV_NAME = config['environment']['name']

env = gym.make(ENV_NAME)
env_eval = gym.make(ENV_NAME)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

# if action_bound is manually set in the settings, then it should be used,
# else the value is used from the environment
action_bound = env.action_space.high[0]
cur_datetime = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
summary_dir = config["summary"]["dir"].format(ENV_NAME, cur_datetime)

if config['settings']['saving'] is not None:
    if np.load(config['settings']['saving']).shape[0] > 100:
        exit()
############################################
# Create the Neural network
############################################
opt = tf.train.AdamOptimizer
policy = NN([state_dim], action_dim, action_bound=action_bound, configuration=config[NETWORK_TYPE_ACTOR])
policy_target = NN([state_dim], action_dim, action_bound=action_bound, configuration=config[NETWORK_TYPE_ACTOR])
critic = NN([state_dim, action_dim], 1, action_bound=1., configuration=config[NETWORK_TYPE_CRITIC])
critic_target = NN([state_dim, action_dim], 1, action_bound=1., configuration=config[NETWORK_TYPE_CRITIC])
session = tf.Session()
############################################
# Initialize DDPG
############################################
ddpg = DDPG(config,
            critic,
            critic_target,
            policy,
            policy_target,
            opt(config[NETWORK_TYPE_CRITIC]['learning_rate']),
            opt(config[NETWORK_TYPE_ACTOR]['learning_rate']),
            session=session,
            gamma=config[NETWORK_TYPE_CRITIC]["discount_factor"])

session.run(tf.global_variables_initializer())
############################################
# Copy the initialization
############################################
# policy_target.loadValues(session, policy.getValues(session))
# policy_target.loadValues(session, policy.getValues(session))

writer = tf.summary.FileWriter(summary_dir, session.graph)

# OU_Noise-Parameters:
alpha = config["noise"]["alpha"]
beta = config["noise"]["beta"]
epsilon_decay = config["noise"]["epsilon_decay"]

# Initialize the noise
noise = Brownian2([action_dim], alpha=alpha, beta=beta, epsilon_decay=epsilon_decay)
# num_actions = env.action_space.shape[-1]
# noise=mu=OU_ActionNoise(np.zeros(num_actions), sigma=float(0.08) * np.ones(num_actions))


# Replay memory
rl_buffer = RLBuffer(env, batch_size=config['memory']['batch_size'], max_size=config['memory']['capacity'])
rl_buffer.mode = 'batch'

ddpg.hard_update()

exploration_steps = config["ddpg"]["exploration_steps"]
evaluation_steps = config["ddpg"]["evaluation_steps"]
max_episodes = config["ddpg"]["max_episodes"]
train_steps = config["ddpg"]["train_steps"]
is_evaluation = config["ddpg"]["is_evaluation"]



# Initialize the Runner - which interacts with the environment and stores the data in the replay memory
runner = Runner(ddpg, env, limit=action_bound, noise=noise, writer=writer,episode_length=config["ddpg"]["train_steps"])
runner_eval = Runner(ddpg, env_eval, limit=action_bound, noise=None, writer=writer,episode_length=config["ddpg"]["evaluation_steps"])

def apply(rlalg,runner, save=True, reset=False, render=True,  verbose=True, exploit=False):
    # global runner
    data, done = runner.apply(render=render, reset=reset, verbose=verbose, exploit=exploit)
    if save:
        rl_buffer.add_sample(*data)
    return done


############# with exploration #############

def evaluation():
    # evaluate
    print("Evaluation:")
    for i in range(exploration_steps):
        done = apply(ddpg,runner_eval,save=False, render=config['environment']['render_eval'],verbose=True, exploit=False)
        if done:
            runner_eval.reset()
    print("Evaluation end")


for _ in range(config['memory']['batch_size']):
    apply(ddpg, runner,save=True, render=False)



check_variables = [k for k in config["plot"]]

def reset_values(check_variables, dic_var):
    for k in check_variables:
        dic_var[k] = []

def update_var(dic_var, var_name, var_val):
        dic_var[var_name] = dic_var[var_name] + [round(var_val,3)]



def print_vars(dic_var):
    for k in dic_var:
        print("print_vars-",k, np.mean(dic_var[k])," -without mean=",dic_var[k])

def plot_vars(dic_var):
    N = len(dic_var)
    plt.clf()
    for i, k in enumerate(dic_var):
        plt.subplot(N,1,i+1)
        plt.title(k)
        plt.plot(dic_var[k])
    plt.pause(0.0000001)

dic_var = {}

reset_values(check_variables, dic_var)

import time

time_start = time.clock()
print("render_eval=",config['environment']['render_eval'])

for episode in range(max_episodes):
    print("\nepisode:", episode)
    for step in range(train_steps):
        # random steps
        # for random_step in range(exploration_steps):
        apply(ddpg, runner, save=True, render=config['environment']['render'], verbose=True, exploit=False)

        # train
        # for i in range(train_steps):
        # sample the next batch
        _, s, a, r, s_n, t = rl_buffer.next_batch()

        # update the critic and the actor
        update_var(dic_var,"critic_error", ddpg.critic_update(s, a, r, s_n, t))
        update_var(dic_var,"actor_error", ddpg.actor_update(s))
        update_var(dic_var,"values", ddpg.get_value(s,a))
        # perform the soft update
        ddpg.soft_update(tau=config['algorithm']['tau'])
        #if done:
            # noise.reset()
            #runner.reset()
    if is_evaluation:
        evaluation()
    #print_vars(dic_var)
    # perform the noise update - this is not classical DDPG ver
    noise.noise_update()
    #print("critic error", total_error/1000.)
    #add_reward_to_summary(episode, total_error, writer, "train/error_per_episode_with_bn")
    #add_reward_to_summary(episode, total_value, writer, "train/Q_per_episode_with_bn")
#####################################

time_elapsed = (time.clock() - time_start)

for ret in runner.list_reward:
    update_var(dic_var, "reward", ret)

for ret in runner_eval.list_reward:
    update_var(dic_var, "reward_eval", ret)

# save data to log-files
#for key,data_arr in dic_var.items():
#    save_np_array_to_file(data_arr,alg=config['algorithm']['type'],filename=config["plot"][key])

#for key,data_arr in dic_var.items():
# save_np_arrays_to_binary_file(dic_var,alg=config['algorithm']['type'],filename=config["plot"])
for key,data_arr in dic_var.items():
    save_single_np_array_to_binary_file(dic_var,alg=config['algorithm']['type'],filename=config["plot"][key])

# save computational time to file
#save_np_array_to_file(np.array([time_elapsed]),alg=config['algorithm']['type'],filename="computational time.txt")
save_single_np_array_to_binary_file({"computational_time":np.array([time_elapsed])},alg=config['algorithm']['type'],filename="computational_time.txt")

'''
for _ in range(exploration_steps):
    apply(ddpg,runner,save=False, render=config['environment']['render'])
'''

# Save data
experiments_store = None
if config['settings']['saving'] is not None:
    experiments_store = np.load(config['settings']['saving'])
    np.save(config['settings']['saving'],
            np.concatenate((experiments_store, [runner.list_reward[:config['settings']['n_episodes']]]), axis=0))
