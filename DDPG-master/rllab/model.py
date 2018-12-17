import numpy as np
import tensorflow as tf

from rllab import ACTIVATION_FUNCTIONS_DICT, WEIGHT_FUNCTIONS_DICT, NETWORK_TYPE_ACTOR, NETWORK_TYPE_CRITIC, batch_norm2


def get_weight(input_dim, output_dim, weights=None):
    if weights is None:
        if output_dim is not None:
            glorot_w = np.sqrt(6. / (input_dim + output_dim))
            return tf.random_uniform(shape=[input_dim, output_dim],
                                                      minval=-glorot_w, maxval=glorot_w,
                                                      dtype=tf.float32)
        else:
            glorot_w = np.sqrt(6. / (input_dim + 1))
            return tf.random_uniform(shape=[input_dim],
                                     minval=-glorot_w, maxval=glorot_w,
                                     dtype=tf.float32)
    else:
        if output_dim is not None:
            return tf.random_uniform(shape=[input_dim, output_dim],
                                                      minval=-weights, maxval=weights,
                                                      dtype=tf.float32)
        else:
            glorot_w = np.sqrt(6. / (input_dim + 1))
            return tf.random_uniform(shape=[input_dim],
                                     minval=-weights, maxval=weights,
                                     dtype=tf.float32)


# hidden layer and output layer have different weight initialization
# def get_weight(input_dim, output_dim, layer, configuration):
#     # set weights of hidden layers
#     if layer is not None:
#         # set weight function for hidden layer
#         weight_fn = WEIGHT_FUNCTIONS_DICT[
#             configuration["layers_weight_fn"]["layer" + str(layer)]]
#         weight = (1 / weight_fn(input_dim))
#
#         if output_dim is not None:
#             # set weights
#             return tf.random_uniform(shape=[input_dim, output_dim], minval=-weight, maxval=weight, dtype=tf.float32)
#         else:
#             # set bias
#             return tf.random_uniform(shape=[input_dim], minval=-weight, maxval=weight, dtype=tf.float32)
#     # set output layer weights:
#     else:
#         # set weight function for output layer
#         weight = configuration["output_layer"]["weight"]
#         #
#         if output_dim is not None:
#             return tf.random_uniform(shape=[input_dim, output_dim], minval=-weight, maxval=weight, dtype=tf.float32)
#         else:
#             # set weights of bias
#             return tf.random_uniform(shape=[input_dim], minval=-weight, maxval=weight, dtype=tf.float32)


class TFModel:

    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        raise Exception("Not implemented yet")

    def getInputDimensions(self):
        raise Exception("Not implemented yet")

    def getOutputDimensions(self):
        raise Exception("Not implemented yet")

    def getParams(self):
        raise Exception("Not implemented yet")

    def _createAssigners(self):
        self.pl = {}
        self.assigners = {}
        for k in self.params:
            self.pl[k] = tf.placeholder(dtype=tf.float32, shape=self.params[k].shape, name="pl" + str(k))
            self.assigners[k] = tf.assign(self.params[k], self.pl[k])

    def createAssigners_bn(self, target_params):
        self.target_params = target_params
        self.pl_bn = {}
        self.assigners_bn = {}

        for k in range(len(target_params)):
            self.pl_bn[k] = tf.placeholder(dtype=tf.float32, shape=target_params[k].shape, name="pl_bn" + str(k))
            self.assigners_bn[k] = tf.assign(target_params[k], self.pl_bn[k])

    def setParam(self, session, name, value):
        session.run(self.assigners[name], feed_dict={self.pl[name]: value})

    def getValues(self, session):
        return session.run(self.params)

    def loadValues(self, session, values):
        for k in values:
            session.run(self.assigners[k], feed_dict={self.pl[k]: values[k]})

    def softSet(self, session, tau, model):
        params = model.params
        for k in params:
            my_val = session.run(self.params[k])
            par_val = session.run(model.params[k])
            session.run(self.assigners[k], feed_dict={self.pl[k]: (1 - tau) * my_val + tau * par_val})

    def softSet_bn(self, session, tau, params_eval):
        params_target = self.target_params

        for k in range(len(params_eval)):
            eval_val = session.run(params_eval[k])
            target_val = session.run(params_target[k])
            session.run(self.assigners_bn[k], feed_dict={self.pl_bn[k]: (1 - tau) * eval_val + tau * target_val})

    def hardSet(self, session, model):
        params = model.params
        for k in params:
            par_val = session.run(model.params[k])
            session.run(self.assigners[k], feed_dict={self.pl[k]: par_val})

    def hardSet_bn(self, session, params_eval):
        params_target = self.target_params

        for k in range(len(params_eval)):
            eval_val = session.run(params_eval[k])
            target_val = session.run(params_target[k])
            session.run(self.assigners_bn[k], feed_dict={self.pl_bn[k]: target_val})


class NN(TFModel):
    def __init__(self, input_dims, output_dim, action_bound=1, configuration=None):
        # create the parameters
        TFModel.__init__(self)
        self.target_params = []
        self.pl_bn = []
        self.input_dims = input_dims
        self.params = {}

        # get values for hidden layers
        self.config = configuration
        # assign number of neurons of each layer to self.layers
        #TODO: remove the commented code if it is useless
        self.layers = self.config["layers"]
        # assign Adaption of batch normalization of each layer to self.layers_is_batch_norm
        self.layers_is_batch_norm = self.config["layers_batch_norm"]
        # activation functions of every layer
        self.layers_activation_fn = self.config["layers_activation_fn"]

        self.output_layer = self.config["output_layer"]

        self.differential = self.config['differential']

        # nr of layers
        self.n_layer = len(self.layers)

        self.output_dim = output_dim
        self.action_dim = output_dim
        self.action_bound = action_bound

        # assign state to prev_dim
        prev_dim = input_dims[0]

        for i in range(self.n_layer):
            cur_dim = self.layers.get("layer" + str(i))

            if "layer_nr_concat" in self.config:
                if self.config['layer_nr_concat'] == i:
                    prev_dim += sum(input_dims[1:])
            self.params["W" + str(i)] = tf.Variable(get_weight(prev_dim, cur_dim),
                                                    name="W" + str(i))
            self.params["b" + str(i)] = tf.Variable(get_weight(cur_dim, None), name="b" + str(i))
            prev_dim = cur_dim

        weight = None
        if 'weight' in self.config['output_layer']:
            weight = self.config['output_layer']['weight']
        self.params["W" + str(self.n_layer)] = tf.Variable(get_weight(cur_dim,
                                                                      output_dim, weight),
                                                           name="W" + str(self.n_layer))
        self.params["b" + str(self.n_layer)] = tf.Variable(get_weight(output_dim, None, weight),
                                                           name="b" + str(self.n_layer))

        self._createAssigners()

    def __call__(self, *args, **kwargs):
        # assign state-value to layer-variable
        layer = args[0]

        # get action from parameters
        if len(args) > 1:
            action = args[1]

        scope_name = kwargs["scope"]

        if "training_phase" in kwargs:
            training_phase = kwargs["training_phase"]

        # nr of layer-index beginning with 0 at which the state is concatenated with the action input
        layer_nr_concat = None

        # get hidden layer number at which the action is injected into critic network
        if "layer_nr_concat" in self.config:
            layer_nr_concat = self.config["layer_nr_concat"]

        # batch norm. at the input-layer
        if self.layers_is_batch_norm["input_layer"]:
            with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
                layer = batch_norm2(layer, "input_layer_bn", training_phase)

        for i in range(self.n_layer):
            # assign the right activation for this layer number, set in the settings
            activation_fn = ACTIVATION_FUNCTIONS_DICT[
                self.config["layers_activation_fn"]["layer" + str(i)]]

            if "action" in locals():
                if i == layer_nr_concat:
                    layer = tf.concat([layer, action], axis=1)
            is_batch_norm = self.layers_is_batch_norm["layer" + str(i)]
            if is_batch_norm:
                # in the critic network there is no batch norm. at the layer where the action is injected,
                # -> has to be set in the settings in the respective layer
                # apply batch norm.

                print(scope_name, "-layer:", i, "-batch_norm=true, no concat")

                layer = tf.matmul(layer, self.params['W' + str(i)], name="mult_layer_with_batch_norm") + self.params[
                        'b' + str(i)]
                layer = activation_fn(layer)


                with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
                    layer = batch_norm2(layer, "layer" + str(i) + "_bn", training_phase)

            # if batch norm. is disabled in the settings:
            else:
                print(scope_name,"-layer:",i,"-batch_norm=false")

                layer = tf.matmul(layer, self.params['W' + str(i)], name="mult_layer_ohne_batch_norm") + self.params[
                    'b' + str(i)]
                layer = activation_fn(layer)

        out = tf.matmul(layer, self.params['W' + str(self.n_layer)], name="output_layer") + self.params[
            'b' + str(self.n_layer)]

        # assign activation function for the output-layer, set in the settings
        output_activation_fn = self.config["output_layer"]["activation_fn"]

        # if output activation func. is given, then apply it to the output layer
        if output_activation_fn != "None":
            output_activation_fn = ACTIVATION_FUNCTIONS_DICT[output_activation_fn]
            out = output_activation_fn(out)
        # the actor network scales the output by multiplying with the action bound
        out = self.action_bound * out
        if self.differential:
            out = out - args[0]
        return out

    def setEvalParams(self, eval_params):
        self.eval_params = eval_params

    def getOutputDimensions(self):
        return self.output_dim

    def getInputDimensions(self):
        return self.input_dims

    def getParams_old(self):
        return self.params

    def getParams(self):
        return self.params.values()
