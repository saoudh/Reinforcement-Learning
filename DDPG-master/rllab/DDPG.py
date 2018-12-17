import tensorflow as tf
import numpy as np
from rllab import getJsonDataFromConfigFile, NETWORK_TYPE_CRITIC


class DDPG:

    def __init__(self,configuration,  Q, Q_target, PI, PI_target, critic_opt, actor_opt, session, gamma=0.99):
        state_dim = Q.getInputDimensions()[0]
        action_dim = Q.getInputDimensions()[1]
        self.training_phase = tf.placeholder(dtype=tf.bool)
        self.config = configuration
        is_L2_regularization = self.config[NETWORK_TYPE_CRITIC]["L2_regularization"]["used"]
        # init of L2 loss
        lossL2 = 0

        #######################################################
        # Neural networks
        #######################################################
        self.Q = Q
        self.Q_target = Q_target
        self.PI_target = PI_target
        self.PI = PI

        #######################################################
        # Placeholders - for the minibatches
        #######################################################
        self.r = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="reward")
        self.s = tf.placeholder(dtype=tf.float32, shape=[None, state_dim], name="state")
        self.a = tf.placeholder(dtype=tf.float32, shape=[None, action_dim], name="action")
        self.s_n = tf.placeholder(dtype=tf.float32, shape=[None, state_dim], name="next_state")
        self.t = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="t")


        # The action chosen by the policy
        self.a_opt = self.PI(self.s, training_phase=self.training_phase, scope="a_eval")
        # set the networks in the graph
        self.a_target = self.PI_target(self.s_n, training_phase=self.training_phase, scope="a_target")
        self.q_eval = self.Q(self.s, self.a_opt, training_phase=self.training_phase, scope="q_eval")
        self.q_eval2 = self.Q(self.s, self.a, training_phase=self.training_phase, scope="q_eval")
        self.q_target = self.Q_target(self.s_n, self.a_target, scope="q_target",
                                      training_phase=self.training_phase)

        # # get parameters for batch normalization of the targets
        # q_target_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="q_target")
        # a_target_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="a_target")
        # # set function for soft and hard update the targets in the graph
        # self.PI_target.createAssigners_bn(a_target_params)
        # self.Q_target.createAssigners_bn(q_target_params)
        #
        # # get parameters for batch normalization of the evaluation networks
        # self.q_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="q_eval")
        # self.a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="a_eval")
        # q_params_total = list(self.Q.getParams())# + list(self.q_params)
        #
        # if is_L2_regularization == "True":
        #     weight_decay = self.config[NETWORK_TYPE_CRITIC]["L2_regularization"]["weight_decay"]
        #     # calculating the L2 loss of every weight except the biases
        #     lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in q_params_total
        #                        if not v.name.startswith("b")]) * weight_decay

        # bellman error: ||r(s,a) + gamma * Q'(s',u'(s')) - Q(s,a)||^2
        # if state is done then ||r(s,a) - Q(s,a)||^2
        # TODO: reinsert (1 - self.t) *
        self.bellman_residual = tf.losses.mean_squared_error(self.r +  gamma *
                                                             tf.stop_gradient(self.q_target), self.q_eval2)# + lossL2

        # Expectation under policy PI
        self.value = tf.reduce_mean(self.q_eval)

        # Optimizations
        # Q.getParams(): batch norm. weights
        # q_params1 and q_params2: weights of the Q-Network
        self.critic_optimizer = critic_opt.minimize(self.bellman_residual, var_list=list(self.Q.getParams()))
        self.actor_optimizer = actor_opt.minimize(-self.value,
                                                  var_list=list(self.PI.getParams()))# + list(self.a_params))

        self.session = session

    def critic_update(self, s, a, r, s_n, t):
        a, _ = self.session.run([self.bellman_residual, self.critic_optimizer],
                                feed_dict={self.s: s, self.a: a, self.r: r, self.s_n: s_n, self.t: t,
                                           self.training_phase: True})
        return a

    def actor_update(self, s):
        a, _ = self.session.run([self.value, self.actor_optimizer], feed_dict={self.s: s, self.training_phase: True})
        return a


    def get_value(self, s,a):
        return np.mean(self.session.run([self.q_eval2],
            feed_dict={self.s: s,self.a:a,self.training_phase:False})[0])

    """
	Bellman Error Value - For Debug Purposes - you could of course use TensorBoard, you don't need this method
	"""

    def br_value(self, s, a, r, s_n, t):
        return self.session.run(self.bellman_residual,
                                feed_dict={self.s: s, self.a: a, self.r: r, self.s_n: s_n, self.t: t,
                                           self.training_phase: False})

    """
	Bellman Error Value - For Debug Purposes - you could of course use TensorBoard, you don't need this method
	"""

    def get_actor_improvement(self, s):
        return self.session.run(self.value, feed_dict={self.s: s, self.training_phase: False})

    """
	this is how the algorithm behaves
	"""

    def applyAction(self, s, **kwargs):
        return self.session.run(self.a_opt, feed_dict={self.s: s, self.training_phase: False})

    def soft_update(self, tau):
        self.Q_target.softSet(self.session, tau, self.Q)
        self.PI_target.softSet(self.session, tau, self.PI)
        #self.PI_target.softSet_bn(self.session, tau, self.a_params)
        #self.Q_target.softSet_bn(self.session, tau, self.q_params)

    def hard_update(self):
        self.Q_target.hardSet(self.session, self.Q)
        self.PI_target.hardSet(self.session, self.PI)
        #self.Q_target.hardSet_bn(self.session, self.q_params)
        #self.PI_target.hardSet_bn(self.session, self.a_params)
