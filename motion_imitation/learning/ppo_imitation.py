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

import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import time
from collections import deque

import numpy as np
import tensorflow as tf
from mpi4py import MPI

from stable_baselines.common import Dataset, explained_variance, fmt_row, zipsame, ActorCriticRLModel, SetVerbosity, \
  TensorboardWriter
from stable_baselines import logger
import stable_baselines.common.tf_util as tf_util
from stable_baselines.common.tf_util import total_episode_reward_logger
from stable_baselines.common.policies import ActorCriticPolicy
from stable_baselines.common.mpi_adam import MpiAdam
from stable_baselines.common.mpi_moments import mpi_moments
from stable_baselines.common.misc_util import flatten_lists
from stable_baselines.common.runners import traj_segment_generator
from stable_baselines.trpo_mpi.utils import add_vtarg_and_adv
from stable_baselines.ppo1 import pposgd_simple

from motion_imitation.learning.imitation_runners import traj_segment_generator


def add_vtarg_and_adv(seg, gamma, lam):
  """
  Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)

  :param seg: (dict) the current segment of the trajectory (see traj_segment_generator return for more information)
  :param gamma: (float) Discount factor
  :param lam: (float) GAE factor
  """
  # last element is only used for last vtarg, but we already zeroed it if last new = 1
  episode_starts = np.append(seg["episode_starts"], False)
  vpred = seg["vpred"]
  nexvpreds = seg["nextvpreds"]
  rew_len = len(seg["rewards"])
  seg["adv"] = np.empty(rew_len, 'float32')
  rewards = seg["rewards"]
  lastgaelam = 0
  for step in reversed(range(rew_len)):
    nonterminal = 1 - float(episode_starts[step + 1])
    delta = rewards[step] + gamma * nexvpreds[step] - vpred[step]
    seg["adv"][step] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
  seg["tdlamret"] = seg["adv"] + seg["vpred"]

  return

class PPOImitation(pposgd_simple.PPO1):
    """
    Proximal Policy Optimization algorithm (MPI version).
    Paper: https://arxiv.org/abs/1707.06347

    :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
    :param policy: (ActorCriticPolicy or str) The policy model to use (MlpPolicy, CnnPolicy, CnnLstmPolicy, ...)
    :param timesteps_per_actorbatch: (int) timesteps per actor per update
    :param clip_param: (float) clipping parameter epsilon
    :param entcoeff: (float) the entropy loss weight
    :param optim_epochs: (float) the optimizer's number of epochs
    :param optim_stepsize: (float) the optimizer's stepsize
    :param optim_batchsize: (int) the optimizer's the batch size
    :param gamma: (float) discount factor
    :param lam: (float) advantage estimation
    :param adam_epsilon: (float) the epsilon value for the adam optimizer
    :param schedule: (str) The type of scheduler for the learning rate update ('linear', 'constant',
        'double_linear_con', 'middle_drop' or 'double_middle_drop')
    :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    :param tensorboard_log: (str) the log location for tensorboard (if None, no logging)
    :param _init_setup_model: (bool) Whether or not to build the network at the creation of the instance
    :param policy_kwargs: (dict) additional arguments to be passed to the policy on creation
    :param full_tensorboard_log: (bool) enable additional logging when using tensorboard
        WARNING: this logging can take a lot of space quickly
    :param seed: (int) Seed for the pseudo-random generators (python, numpy, tensorflow).
        If None (default), use random seed. Note that if you want completely deterministic
        results, you must set `n_cpu_tf_sess` to 1.
    :param n_cpu_tf_sess: (int) The number of threads for TensorFlow operations
        If None, the number of cpu of the current machine will be used.
    """
    def __init__(self, policy, env, gamma=0.99, timesteps_per_actorbatch=256, clip_param=0.2, entcoeff=0.01,
                 optim_epochs=4, optim_stepsize=1e-3, optim_batchsize=64, lam=0.95, adam_epsilon=1e-5,
                 schedule='linear', verbose=0, tensorboard_log=None, _init_setup_model=True,
                 policy_kwargs=None, full_tensorboard_log=False, seed=None, n_cpu_tf_sess=1):

        super().__init__(policy=policy,
                         env=env,
                         gamma=gamma,
                         timesteps_per_actorbatch=timesteps_per_actorbatch,
                         clip_param=clip_param,
                         entcoeff=entcoeff,
                         optim_epochs=optim_epochs,
                         optim_stepsize=optim_stepsize,
                         optim_batchsize=optim_batchsize,
                         lam=lam,
                         adam_epsilon=adam_epsilon,
                         schedule=schedule,
                         verbose=verbose,
                         tensorboard_log=tensorboard_log,
                         _init_setup_model=_init_setup_model,
                         policy_kwargs=policy_kwargs,
                         full_tensorboard_log=full_tensorboard_log,
                         seed=seed,
                         n_cpu_tf_sess=n_cpu_tf_sess)
        return


    def setup_model(self):
      with SetVerbosity(self.verbose):

        self.graph = tf.Graph()
        with self.graph.as_default():
          self.set_random_seed(self.seed)
          self.sess = tf_util.make_session(num_cpu=self.n_cpu_tf_sess, graph=self.graph)

          # Construct network for new policy
          self.policy_pi = self.policy(self.sess, self.observation_space, self.action_space, self.n_envs, 1,
                                       None, reuse=False, **self.policy_kwargs)

          # Network for old policy
          with tf.variable_scope("oldpi", reuse=False):
            old_pi = self.policy(self.sess, self.observation_space, self.action_space, self.n_envs, 1,
                                 None, reuse=False, **self.policy_kwargs)

          with tf.variable_scope("loss", reuse=False):
            # Target advantage function (if applicable)
            atarg = tf.placeholder(dtype=tf.float32, shape=[None])

            # Empirical return
            ret = tf.placeholder(dtype=tf.float32, shape=[None])

            # learning rate multiplier, updated with schedule
            lrmult = tf.placeholder(name='lrmult', dtype=tf.float32, shape=[])

            # Annealed cliping parameter epislon
            clip_param = self.clip_param * lrmult

            obs_ph = self.policy_pi.obs_ph
            action_ph = self.policy_pi.pdtype.sample_placeholder([None])

            kloldnew = old_pi.proba_distribution.kl(self.policy_pi.proba_distribution)
            ent = self.policy_pi.proba_distribution.entropy()
            meankl = tf.reduce_mean(kloldnew)
            meanent = tf.reduce_mean(ent)
            pol_entpen = (-self.entcoeff) * meanent

            # pnew / pold
            ratio = tf.exp(self.policy_pi.proba_distribution.logp(action_ph) -
                           old_pi.proba_distribution.logp(action_ph))

            # surrogate from conservative policy iteration
            surr1 = ratio * atarg
            surr2 = tf.clip_by_value(ratio, 1.0 - clip_param, 1.0 + clip_param) * atarg

            clip_frac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), clip_param)))

            # PPO's pessimistic surrogate (L^CLIP)
            pol_surr = - tf.reduce_mean(tf.minimum(surr1, surr2))
            vf_loss = tf.reduce_mean(tf.square(self.policy_pi.value_flat - ret))
            total_loss = pol_surr + pol_entpen + vf_loss
            losses = [pol_surr, pol_entpen, vf_loss, meankl, meanent]
            self.loss_names = ["pol_surr", "pol_entpen", "vf_loss", "kl", "ent"]

            tf.summary.scalar('entropy_loss', pol_entpen)
            tf.summary.scalar('policy_gradient_loss', pol_surr)
            tf.summary.scalar('value_function_loss', vf_loss)
            tf.summary.scalar('approximate_kullback-leibler', meankl)
            tf.summary.scalar('clip_factor', clip_param)
            tf.summary.scalar('loss', total_loss)
            tf.summary.scalar('clip_frac', clip_frac)

            self.params = tf_util.get_trainable_vars("model")

            self.assign_old_eq_new = tf_util.function(
                [], [], updates=[tf.assign(oldv, newv) for (oldv, newv) in
                                 zipsame(tf_util.get_globals_vars("oldpi"), tf_util.get_globals_vars("model"))])

          with tf.variable_scope("Adam_mpi", reuse=False):
            self.adam = MpiAdam(self.params, epsilon=self.adam_epsilon, sess=self.sess)

          with tf.variable_scope("input_info", reuse=False):
            tf.summary.scalar('discounted_rewards', tf.reduce_mean(ret))
            tf.summary.scalar('learning_rate', tf.reduce_mean(self.optim_stepsize))
            tf.summary.scalar('advantage', tf.reduce_mean(atarg))
            tf.summary.scalar('clip_range', tf.reduce_mean(self.clip_param))

            if self.full_tensorboard_log:
              tf.summary.histogram('discounted_rewards', ret)
              tf.summary.histogram('learning_rate', self.optim_stepsize)
              tf.summary.histogram('advantage', atarg)
              tf.summary.histogram('clip_range', self.clip_param)
              if tf_util.is_image(self.observation_space):
                tf.summary.image('observation', obs_ph)
              else:
                tf.summary.histogram('observation', obs_ph)

          self.step = self.policy_pi.step
          self.proba_step = self.policy_pi.proba_step
          self.initial_state = self.policy_pi.initial_state

          tf_util.initialize(sess=self.sess)

          self.summary = tf.summary.merge_all()

          self.lossandgrad = tf_util.function([obs_ph, old_pi.obs_ph, action_ph, atarg, ret, lrmult],
                                              [self.summary, tf_util.flatgrad(total_loss, self.params)] + losses)
          self.compute_losses = tf_util.function([obs_ph, old_pi.obs_ph, action_ph, atarg, ret, lrmult],
                                                 losses)

      return

    def learn(self, total_timesteps, callback=None, log_interval=100, tb_log_name="PPO1",
              reset_num_timesteps=True, save_path=None, save_iters=20):
        is_root = (MPI.COMM_WORLD.Get_rank() == 0)
        new_tb_log = self._init_num_timesteps(reset_num_timesteps)
        callback = self._init_callback(callback)

        with SetVerbosity(self.verbose), TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name, new_tb_log) \
                as writer:
            self._setup_learn()

            assert issubclass(self.policy, ActorCriticPolicy), "Error: the input policy for the PPO1 model must be " \
                                                               "an instance of common.policies.ActorCriticPolicy."

            with self.sess.as_default():
                self.adam.sync()
                callback.on_training_start(locals(), globals())

                # Prepare for rollouts
                seg_gen = traj_segment_generator(self.policy_pi, self.env, self.timesteps_per_actorbatch,
                                                 callback=callback)

                episodes_so_far = 0
                timesteps_so_far = 0
                iters_so_far = 0
                t_start = time.time()

                # rolling buffer for episode lengths
                len_buffer = deque(maxlen=100)
                # rolling buffer for episode rewards
                reward_buffer = deque(maxlen=100)

                while True:
                    if timesteps_so_far >= total_timesteps:
                        break

                    if self.schedule == 'constant':
                        cur_lrmult = 1.0
                    elif self.schedule == 'linear':
                        cur_lrmult = max(1.0 - float(timesteps_so_far) / total_timesteps, 0)
                    else:
                        raise NotImplementedError

                    if is_root:
                        logger.log("********** Iteration %i ************" % iters_so_far)

                    seg = seg_gen.__next__()

                    # Stop training early (triggered by the callback)
                    if not seg.get('continue_training', True):  # pytype: disable=attribute-error
                        break

                    add_vtarg_and_adv(seg, self.gamma, self.lam)

                    # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
                    observations, actions = seg["observations"], seg["actions"]
                    atarg, tdlamret = seg["adv"], seg["tdlamret"]

                    # true_rew is the reward without discount
                    if writer is not None:
                        total_episode_reward_logger(self.episode_reward,
                                                    seg["true_rewards"].reshape((self.n_envs, -1)),
                                                    seg["dones"].reshape((self.n_envs, -1)),
                                                    writer, self.num_timesteps)

                    # predicted value function before udpate
                    vpredbefore = seg["vpred"]

                    # standardized advantage function estimate
                    atarg = (atarg - atarg.mean()) / atarg.std()
                    dataset = Dataset(dict(ob=observations, ac=actions, atarg=atarg, vtarg=tdlamret),
                                      shuffle=not self.policy.recurrent)
                    optim_batchsize = self.optim_batchsize or observations.shape[0]

                    # set old parameter values to new parameter values
                    self.assign_old_eq_new(sess=self.sess)
                    
                    if is_root:
                        logger.log("Optimizing...")
                        logger.log(fmt_row(13, self.loss_names))

                    # Here we do a bunch of optimization epochs over the data
                    for k in range(self.optim_epochs):
                        # list of tuples, each of which gives the loss for a minibatch
                        losses = []
                        for i, batch in enumerate(dataset.iterate_once(optim_batchsize)):
                            steps = (self.num_timesteps +
                                     k * optim_batchsize +
                                     int(i * (optim_batchsize / len(dataset.data_map))))
                            if writer is not None:
                                # run loss backprop with summary, but once every 10 runs save the metadata
                                # (memory, compute time, ...)
                                if self.full_tensorboard_log and (1 + k) % 10 == 0:
                                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                                    run_metadata = tf.RunMetadata()
                                    summary, grad, *newlosses = self.lossandgrad(batch["ob"], batch["ob"], batch["ac"],
                                                                                 batch["atarg"], batch["vtarg"],
                                                                                 cur_lrmult, sess=self.sess,
                                                                                 options=run_options,
                                                                                 run_metadata=run_metadata)
                                    writer.add_run_metadata(run_metadata, 'step%d' % steps)
                                else:
                                    summary, grad, *newlosses = self.lossandgrad(batch["ob"], batch["ob"], batch["ac"],
                                                                                 batch["atarg"], batch["vtarg"],
                                                                                 cur_lrmult, sess=self.sess)
                                writer.add_summary(summary, steps)
                            else:
                                _, grad, *newlosses = self.lossandgrad(batch["ob"], batch["ob"], batch["ac"],
                                                                       batch["atarg"], batch["vtarg"], cur_lrmult,
                                                                       sess=self.sess)

                            self.adam.update(grad, self.optim_stepsize * cur_lrmult)
                            losses.append(newlosses)
                        
                        if is_root:
                            logger.log(fmt_row(13, np.mean(losses, axis=0)))

                    if is_root:
                        logger.log("Evaluating losses...")

                    losses = []
                    for batch in dataset.iterate_once(optim_batchsize):
                        newlosses = self.compute_losses(batch["ob"], batch["ob"], batch["ac"], batch["atarg"],
                                                        batch["vtarg"], cur_lrmult, sess=self.sess)
                        losses.append(newlosses)
                    mean_losses, _, _ = mpi_moments(losses, axis=0)

                    if is_root:
                        logger.log(fmt_row(13, mean_losses))

                    for (loss_val, name) in zipsame(mean_losses, self.loss_names):
                        logger.record_tabular("loss_" + name, loss_val)
                    logger.record_tabular("ev_tdlam_before", explained_variance(vpredbefore, tdlamret))

                    # local values
                    lrlocal = (seg["ep_lens"], seg["ep_rets"])

                    # list of tuples
                    listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal)
                    lens, rews = map(flatten_lists, zip(*listoflrpairs))
                    len_buffer.extend(lens)
                    reward_buffer.extend(rews)
                    if len(len_buffer) > 0:
                        logger.record_tabular("EpLenMean", np.mean(len_buffer))
                        logger.record_tabular("EpRewMean", np.mean(reward_buffer))
                    logger.record_tabular("EpThisIter", len(lens))
                    episodes_so_far += len(lens)
                    current_it_timesteps = MPI.COMM_WORLD.allreduce(seg["total_timestep"])
                    timesteps_so_far += current_it_timesteps
                    self.num_timesteps += current_it_timesteps

                    if is_root and (save_path is not None) and (iters_so_far % save_iters == 0):
                      self.save(save_path)

                    iters_so_far += 1
                    logger.record_tabular("EpisodesSoFar", episodes_so_far)
                    logger.record_tabular("TimestepsSoFar", self.num_timesteps)
                    logger.record_tabular("TimeElapsed", time.time() - t_start)
                    if self.verbose >= 1 and is_root:
                        logger.dump_tabular()
        callback.on_training_end()

        if is_root:
            self.save(save_path)

        return self
