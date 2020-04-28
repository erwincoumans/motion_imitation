# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
import numpy as np
import tensorflow as tf

from stable_baselines.common.distributions import make_proba_dist_type, spaces, \
    DiagGaussianProbabilityDistributionType
from stable_baselines.common.policies import FeedForwardPolicy, nature_cnn, mlp_extractor, linear


def make_proba_dist_type(ac_space):
    """
    return an instance of ProbabilityDistributionType for the correct type of action space

    :param ac_space: (Gym Space) the input action space
    :return: (ProbabilityDistributionType) the appropriate instance of a ProbabilityDistributionType
    """
    if isinstance(ac_space, spaces.Box):
        assert len(ac_space.shape) == 1, "Error: the action space must be a vector"
        return DiagGaussianFixedVarProbabilityDistributionType(ac_space.shape[0])
    else:
        return make_proba_dist_type(ac_space)


class DiagGaussianFixedVarProbabilityDistributionType(DiagGaussianProbabilityDistributionType):
    def __init__(self, size):
        super(DiagGaussianFixedVarProbabilityDistributionType, self).__init__(size)
        return

    def proba_distribution_from_latent(self, pi_latent_vector, vf_latent_vector,
                                        pi_init_scale=1.0, pi_init_bias=0.0, pi_init_std=1.0,
                                        vf_init_scale=1.0, vf_init_bias=0.0):
        mean = linear(pi_latent_vector, 'pi', self.size, init_scale=pi_init_scale, init_bias=pi_init_bias)
        logstd = tf.get_variable(name='pi/logstd', shape=[1, self.size], initializer=tf.constant_initializer(np.log(pi_init_std)), trainable=False)
        pdparam = tf.concat([mean, mean * 0.0 + logstd], axis=1)
        q_values = linear(vf_latent_vector, 'q', self.size, init_scale=vf_init_scale, init_bias=vf_init_bias)
        return self.proba_distribution_from_flat(pdparam), mean, q_values


class ImitationPolicy(FeedForwardPolicy):
    """
    Policy object that implements actor critic, using a feed forward neural network.

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param layers: ([int]) (deprecated, use net_arch instead) The size of the Neural network for the policy
        (if None, default to [64, 64])
    :param net_arch: (list) Specification of the actor-critic policy network architecture (see mlp_extractor
        documentation for details).
    :param act_fun: (tf.func) the activation function to use in the neural network.
    :param cnn_extractor: (function (TensorFlow Tensor, ``**kwargs``): (TensorFlow Tensor)) the CNN feature extraction
    :param feature_extraction: (str) The feature extraction type ("cnn" or "mlp")
    :param kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, layers=None, net_arch=None,
                 act_fun=tf.tanh, cnn_extractor=nature_cnn, feature_extraction="mlp", **kwargs):
        super(FeedForwardPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse,
                                                scale=(feature_extraction == "cnn"))

        self._pdtype = make_proba_dist_type(ac_space)

        self._kwargs_check(feature_extraction, kwargs)

        if layers is not None:
            warnings.warn("Usage of the `layers` parameter is deprecated! Use net_arch instead "
                          "(it has a different semantics though).", DeprecationWarning)
            if net_arch is not None:
                warnings.warn("The new `net_arch` parameter overrides the deprecated `layers` parameter!",
                              DeprecationWarning)

        if net_arch is None:
            if layers is None:
                layers = [64, 64]
            net_arch = [dict(vf=layers, pi=layers)]

        with tf.variable_scope("model", reuse=reuse):
            if feature_extraction == "cnn":
                pi_latent = vf_latent = cnn_extractor(self.processed_obs, **kwargs)
            else:
                pi_latent, vf_latent = mlp_extractor(tf.layers.flatten(self.processed_obs), net_arch, act_fun)

            self._value_fn = linear(vf_latent, 'vf', 1)

            self._proba_distribution, self._policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent,
                                                           pi_init_scale=1.0, pi_init_bias=0.0, pi_init_std=0.125,
                                                           vf_init_scale=1.0, vf_init_bias=0.0)

        self._setup_init()
        return
