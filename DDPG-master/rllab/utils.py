import json
import math
import sys
from zipfile import BadZipFile

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import variables
import os
from rllab.moving_averages import assign_moving_average
import scipy.stats as st

"""
Constants
"""

NETWORK_TYPE_ACTOR = "actor_network"
NETWORK_TYPE_CRITIC = "critic_network"
NETWORK_TYPE_MODEL = "model_network"
NETWORK_TYPE_REWARD = "reward_network"
NETWORK_TYPE_DONE = "done_network"

LOSS_FUNCTIONS_DICT = {"MSE": tf.losses.mean_squared_error,
                       "cross_entropy": tf.nn.softmax_cross_entropy_with_logits
                       }

ACTIVATION_FUNCTIONS_DICT = {"relu": tf.nn.relu,
                             "tanh": tf.nn.tanh,
                             "sigmoid": tf.nn.sigmoid}
WEIGHT_FUNCTIONS_DICT = {"sqrt": math.sqrt}

settings_path_dir = "../settings/"
#for cluster:
#settings_path_dir = "settings/"

CONFIDENCE_INTERVAL_LOG_FILE_SUFFIX = "_conf_interv"


def save_json_data_to_file(settings, file_name):
    with open(settings_path_dir + file_name, 'w+') as f:
        json.dump(dict(settings), f)


def getJsonDataFromConfigFile(file_name):
    print("current dir=", os.getcwd())
    # first argument-value is the setting-file-path
    if len(sys.argv) == 1:
        with open(settings_path_dir + file_name, 'r') as f:
            return json.load(f)
    else:
        with open(settings_path_dir + sys.argv[1], 'r') as f:
            return json.load(f)


################### Log files for plotting ####################
# if sub-directory of log-files-directory is given, then it is used, else the default one
# it is given as the second argument
# last argument-value is number of experiment-round, so the number of argument are one higher

def get_log_dir(dir_name):
    if len(sys.argv) >= 3:
        return "../logfiles/" + sys.argv[2] + "/"
    elif dir_name is not None:
        return "../logfiles/" + dir_name + "/"
    else:
        return "../logfiles/"


LOG_DIR = get_log_dir(None)


# it saves as txt-files! Better use the compressed alternative function!
def save_np_array_to_file(data_arr, alg, filename):
    filename = LOG_DIR + alg + "/" + filename
    i = 1
    # for multiple experiments runs: as long as the log-filename already exists,
    # increment the ending of the filename
    while os.path.isfile(filename):
        filename = filename + "%s" % i
        i += 1
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    np.savetxt(filename, data_arr)

import re

def save_single_np_array_to_binary_file(dic_var, alg, filename):

    DIR = LOG_DIR + alg
    #file_prefix = filename[:filename.index(".")]
    p=re.compile('[a-z_]+')
    file_prefix=p.match(filename)
    file_prefix=file_prefix.group()

    j = ""
    # for multiple experiments runs: as long as the log-filename already exists,
    # increment the ending of the filename
    while os.path.isfile(DIR+"/"+file_prefix+str(j)+".npz"):
        if j=="":
            j = 1
        j+=1

    filename = file_prefix + "%s" % str(j)+".npz"

    os.makedirs(os.path.dirname(DIR+"/"), exist_ok=True)

    np.savez_compressed(DIR+"/"+filename, **dic_var)



def save_np_arrays_to_binary_file(dic_var,alg,filename):
    # filename-format: logdir/mbddpg_1_layer_25_neurons.dat
    filename = LOG_DIR + alg +"/"+filename
    mydic={}
    # add all arrays of this experiment to a dictionary
    for key,data in dic_var.items():
        mydic.update({key:data})
    # save all arrays to one compressed file
    i=""
    # as long as the filename exists, increment it
    while os.path.isfile(filename+str(i)):
        if i=="":
            i=1
    i += 1

    filename = filename + "%s" % str(int(i))
    # create directory if not exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    np.savez_compressed(filename,**mydic)


# get the log files for actor-, critic-error
# format: Array->dict-objects {"loss-type": File(),...}
def get_log_files(alg=None, values=None, log_subdir=None):
    global LOG_DIR
    if log_subdir is not None:
        LOG_DIR = get_log_dir(log_subdir)
    log_files = {}
    for key, filename in values.items():
        file_prefix = filename[:filename.index(".")]
        p = re.compile('[a-z]+')
        file_prefix = p.match(file_prefix)
        file_prefix = file_prefix.group()
        # get first log-file which is more similiar to the default log-file name
        # i.e. actor_error.txt1 is nearer to actor_error.txt than actor_error.txt123
        for file in os.listdir(LOG_DIR + alg + "/"):
            if file.startswith(filename):
                log_files.update({key: np.load(LOG_DIR + alg + "/" + file)[file_prefix]})
                #log_files.update({key: np.loadtxt(LOG_DIR + alg + "/" + file)})
                break
    return log_files


# get the log files for actor-, critic-error
# format: Array->dict-objects {"loss-type": File(),...}
# param: log_subdir, is the sub-directory of logfiles-directory with names of
# the experiment parameters like name of environment, number of neurons and episodes
# for compressed binary files .npz
def get_log_files_with_conf_interv_from_binary_file(alg=None, values=None, log_subdir=None):
    global LOG_DIR
    mydict={}
    #print(values)
    #for key,filename in values.items:
    #    mydict.update(key)
    if log_subdir is not None:
        LOG_DIR = "../"+get_log_dir(log_subdir)
    print("logdir=",LOG_DIR)

    log_files = {}
    for key, filename in values.items():
        # filename = LOG_DIR + alg + "/" + filename
        # fill array log_files with dict-objects of format observation type like "actor_error"
        # and an array of objects with confidence interval values and format {mean: x,low:y,high:z}
        #todo: key and file_prefix are the same! sufficient to just use key without extracting regex!
        file_prefix = filename[:filename.index(".")]
        p = re.compile('[a-z_]+')
        file_prefix = p.match(file_prefix)
        file_prefix = file_prefix.group()
        try:
            if os.path.isfile(LOG_DIR +alg + "/" + file_prefix + CONFIDENCE_INTERVAL_LOG_FILE_SUFFIX+".npz"):
                #log_files.update({key: np.loadtxt(LOG_DIR + alg + "/" + filename + CONFIDENCE_INTERVAL_LOG_FILE_SUFFIX,dtype=float, delimiter=",")})
                #load array from compressed binary file which is saved in the "key"-index of the file
                try:
                    log_files.update({key: np.load(LOG_DIR + alg + "/" + file_prefix + CONFIDENCE_INTERVAL_LOG_FILE_SUFFIX+".npz")[file_prefix+CONFIDENCE_INTERVAL_LOG_FILE_SUFFIX]})
                except KeyError:
                    for k,i in np.load(LOG_DIR + alg + "/" + file_prefix + CONFIDENCE_INTERVAL_LOG_FILE_SUFFIX+".npz").items():
                        print("k=",k,"-.shape=",np.shape(i))
                    log_files.update({key: np.load(LOG_DIR + alg + "/" + file_prefix + CONFIDENCE_INTERVAL_LOG_FILE_SUFFIX+".npz")["arr_0"]})
            else:
                log_files.update({key: _calc_conf_interv(LOG_DIR + alg + "/", file_prefix)})
        except IOError:
            log_files.update({key: _calc_conf_interv(LOG_DIR + alg + "/", file_prefix)})
    return log_files


# func. for calculating the confidence interval
# for compressed binary fiiles .npz
def _calc_conf_interv(log_dir, file_prefix):
    single_row = []
    # assign logs of all experiments of an algorithm and obeservation to an array
    for cur_filename in os.listdir(log_dir):
        if cur_filename.startswith(file_prefix) and CONFIDENCE_INTERVAL_LOG_FILE_SUFFIX not in cur_filename\
                and "computational" not in cur_filename:
            try:
                single_row.append(np.load(log_dir + "/" + cur_filename)[file_prefix])
            except:
                print("badzipfile:",log_dir+cur_filename)
                os.remove(log_dir+cur_filename)
    # invert the axis to calculate the mean of the experiments and low and high bound
    stackarr = np.stack(single_row, 1)

    mean_dict_arr = []
    # calculate lower and higher bound of the confidence interval and the mean
    # and write it as a dict-object to an array
    for a in stackarr:
        myarray = st.t.interval(0.95, len(a) - 1, loc=np.mean(a), scale=st.sem(a))
        mean_dict_arr.append([np.mean(a), myarray[0], myarray[1]])
    mydict={file_prefix+CONFIDENCE_INTERVAL_LOG_FILE_SUFFIX:mean_dict_arr}

    # format of log-file mean, low bound, high bound
    np.savez_compressed(log_dir + file_prefix + CONFIDENCE_INTERVAL_LOG_FILE_SUFFIX+".npz", **mydict)

    return mean_dict_arr




"""
Noise
"""


class OU_ActionNoise:
    def __init__(self, mu, sigma, theta=.1, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(
            self.dt) * np.random.normal(size=np.shape(self.mu))
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    # def __repr__(self):


#	return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


class Brownian2:

    def __init__(self, dim, alpha=0.1, beta=0.1, epsilon=1., epsilon_decay=0.9995):
        self.alpha = alpha
        self.beta = beta
        self.dim = dim
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.noise = np.random.multivariate_normal(np.zeros(dim), np.eye(self.dim[0]))

    def draw(self):
        source = np.random.multivariate_normal(np.zeros(self.dim), np.eye(self.dim[0]))
        self.noise += -self.alpha * self.noise + self.beta * source
        return self.epsilon * self.noise

    def noise_update(self):
        self.epsilon *= self.epsilon_decay


"""
Interact with the environment and collect samples
"""


class Runner:

    def __init__(self, algorithm, environment, noise=None, limit=None, writer=None, episode_length=200):
        self.alg = algorithm
        self.env = environment
        self.noise = noise
        self.action_lim = limit
        self.state_dim = environment.observation_space.low.shape[0]
        self.action_dim = environment.action_space.low.shape[0]
        self.tot_reward = 0
        self.list_reward = []
        self.writer = writer
        self.state = self.env.reset()
        self.n_steps = 0
        self.episode_length = episode_length
        self.episode_nr = 0

    def apply(self, reset=False, render=True, verbose=True, exploit=False, draw_from_model=False):

        self.n_steps += 1
        # get action from policy
        action = self.alg.applyAction(self.state.reshape((1, self.state_dim)), exploit=exploit)
        # if self.noise is not None and not exploit:
        # action += self.noise.draw()
        if render:
            self.env.render()

        if draw_from_model:
            next_state, reward, done, _ = self.alg.model_step(self.state,
                                                              np.clip(action.reshape([self.action_dim]),
                                                                      -self.action_lim, self.action_lim))
        else:
            next_state, reward, done, _ = self.env.step(
                np.clip(action.reshape([self.action_dim]), -self.action_lim, self.action_lim))

        if hasattr(reward, "item"):
            self.tot_reward += np.asscalar(reward)
        else:
            self.tot_reward += reward

        ret = self.state.reshape((1, self.state_dim))[:], \
              action, \
              np.array([reward]), \
              next_state.reshape((1, self.state_dim)), \
              np.array([1 - done])
        # np.array([1 - done])

        self.state = next_state[:]
        if done or reset or self.n_steps >= self.episode_length:
            self.n_steps = 0
            self.state = self.env.reset()
            # print("episode_nr:", episode_nr)
            if self.writer is not None and self.episode_nr is not None:
                add_reward_to_summary(self.episode_nr, self.tot_reward, self.writer,
                                      tag="train/reward_per_episode_with_bn")

            if not reset:
                if verbose:
                    print("episode_reward:", self.tot_reward)
                self.list_reward.append(self.tot_reward)

            self.tot_reward = 0.
            self.episode_nr += 1

        return ret, done

    def reset(self):
        self.state = self.env.reset()
        self.tot_reward = 0
        self.n_steps = 0


def add_reward_to_summary(episode_nr, value=None, writer=None, tag=None):
    # print("writer:", writer)
    writer = writer
    train_episode_summary = tf.Summary()
    train_episode_summary.value.add(simple_value=value, tag=tag)
    writer.add_summary(train_episode_summary, episode_nr)
    writer.flush()


class EarlyStopping:

    def __init__(self, improvement=0.01, patience=30, conf=lambda x, y: x < y, iter_limit=5000):
        self.improvement = improvement
        self.patience = patience
        self.non_impr = 0
        self.conf = conf
        self.previous_error = np.infty
        self.iter = 0
        self.iter_limit = iter_limit

    def signal(self, val):
        self.iter += 1
        if self.conf(self.previous_error - val, self.improvement):
            self.non_impr += 1
        else:
            self.non_impr = 0
        self.previous_error = val
        if self.non_impr >= self.patience or self.iter >= self.iter_limit:
            return True
        return False

    def reset(self):
        self.non_impr = 0
        self.previous_error = np.infty
        self.iter = 0


def Setting(default_file):
    import json, sys

    if len(sys.argv) == 2:
        with open(sys.argv[1]) as json_data:
            return json.load(json_data)
    elif len(sys.argv) > 2:
        with open(sys.argv[1]) as json_data:
            return json.load(json_data), int(sys.argv[2])
    else:
        with open(default_file) as json_data:
            return json.load(json_data)


############################ Batch Normalization ##################################

def batch_norm_layer(x, scope, is_training, epsilon=0.001, decay=0.95, reuse=None, writer=None):
    with tf.variable_scope(scope, reuse=reuse):
        shape = x.get_shape().as_list()

        # gamma: a trainable scale factor
        gamma = tf.get_variable("gamma", shape[-1], initializer=tf.random_uniform_initializer(0.9, 1.1), trainable=True)
        # gamma=tf.Print(gamma,[gamma],"my gamma")
        # gamma = tf.get_variable("gamma", shape[-1], initializer=tf.constant_initializer(1.0), trainable=True)
        # beta: a trainable shift value
        # beta = tf.get_variable("beta", shape[-1], initializer=tf.constant_initializer(0.0), trainable=True)
        beta = tf.get_variable("beta", shape[-1], initializer=tf.random_uniform_initializer(-0.03, 0.03),
                               trainable=True)

        # beta=tf.Print(beta,[beta],"mybeta")
        moving_avg = tf.get_variable("moving_avg", shape[-1], initializer=tf.constant_initializer(0.0), trainable=False)
        moving_var = tf.get_variable("moving_var", shape[-1], initializer=tf.constant_initializer(1.0), trainable=False)
        if is_training:

            # tf.nn.moments == Calculate the mean and the variance of the tensor x
            avg, var = tf.nn.moments(x, list(range(len(shape) - 1)))
            update_moving_avg = assign_moving_average(moving_avg, avg, decay)
            update_moving_var = assign_moving_average(moving_var, var, decay)
            control_inputs = [update_moving_avg, update_moving_var]
        else:
            avg = moving_avg
            var = moving_var
            control_inputs = []
        with tf.control_dependencies(control_inputs):
            output = tf.nn.batch_normalization(x, avg, var, offset=beta, scale=gamma, variance_epsilon=epsilon)

        return output


def batch_norm2(x, scope, is_training, epsilon=0.001, decay=0.99):
    """
	Returns a batch normalization layer that automatically switch between train and test phases based on the
	tensor is_training

	Args:
		x: input tensor
		scope: scope name
		is_training: boolean tensor or variable
		epsilon: epsilon parameter - see batch_norm_layer
		decay: epsilon parameter - see batch_norm_layer

	Returns:
		The correct batch normalization layer based on the value of is_training
	"""
    assert isinstance(is_training, (ops.Tensor, variables.Variable)) and is_training.dtype == tf.bool

    cond = tf.cond(
        is_training,
        lambda: batch_norm_layer(x=x, scope=scope, epsilon=epsilon, decay=decay, is_training=True, reuse=False),
        lambda: batch_norm_layer(x=x, scope=scope, epsilon=epsilon, decay=decay, is_training=False, reuse=True),
    )
    # cond=tf.Print(cond,[cond],"condition")
    return cond
