#!/usr/bin/env python

import gym
import numpy as np
import tensorflow as tf
import time

_actions = (1, 4, 5)


class PolicyGradientAgent(object):

    def __init__(self, hparams, sess):

        # initialization
        self._s = sess

        # build the graph  tf.float32, [None, envSize], name="input_x"  ---tf.float32,  shape=[None, hparams['input_size']]
        self._input = tf.placeholder(tf.float32,
                                     shape=[None, hparams['input_size']])

        hidden1 = tf.contrib.layers.fully_connected(
            inputs=self._input,
            num_outputs=hparams['hidden_size'],
            activation_fn=tf.nn.relu)

        logits = tf.contrib.layers.fully_connected(
            inputs=hidden1,
            num_outputs=hparams['num_actions'],
            activation_fn=None)

        # op to sample an action
        self._sample = tf.reshape(tf.multinomial(logits, 1), [])

        # add tiny number to avoid sending zero to log
        log_prob = tf.log(tf.nn.softmax(logits) + 1e-8)

        # training part of graph
        self._acts = tf.placeholder(tf.int32)  # tf.float32, [None, 1]
        self._advantages = tf.placeholder(tf.float32)

        # get log probs of actions from episode
        indices = tf.range(0, tf.shape(log_prob)[
                           0]) * tf.shape(log_prob)[1] + self._acts
        act_prob = tf.gather(tf.reshape(log_prob, [-1]), indices)

        # surrogate loss
        loss = -tf.reduce_sum(tf.multiply(act_prob, self._advantages))
        self._debug = loss

        # update
        optimizer = tf.train.RMSPropOptimizer(hparams['learning_rate'])
        self._train = optimizer.minimize(loss)

    def act(self, observation):
        # get one action, by sampling  HACER Q ESTE METODO DEVUELVA UN NUMBER DEL 0 A 4
        return self._s.run(self._sample, feed_dict={self._input: [observation]})

    def train_step(self, obs, acts, advantages):
        batch_feed = {self._input: obs,
                      self._acts: acts,
                      self._advantages: advantages}
        self._s.run(self._train, feed_dict=batch_feed)


def policy_rollout(env, agent):
    """Run one episode."""

    observation, reward, done = env.reset(), 0, False
    obs, acts, rews = [], [], []

    while not done:

        env.render()
        obs.append(observation)

        action = agent.act(observation)
        print("action: ", action)
        observation, reward, done, _ = env.step(
            action)  # action   _actions[action]

        acts.append(action)
        rews.append(reward)

    return obs, acts, rews


def process_rewards(rews):
    """Rewards -> Advantages for one episode. """

    # total reward: length of episode
    return [len(rews)] * len(rews)


def main():

    env = gym.make('MsPacman-ram-v0')

    monitor_dir = '/tmp/cartpole_exp1'
    # env.monitor.start(monitor_dir, force=True)

    # hyper parameters
    hparams = {
        # env.observation_space.shape[0]
        'input_size': env.observation_space.shape[0],
        'hidden_size': 128,  # 36
        'num_actions': env.action_space.n,  # env.action_space.n
        'learning_rate': 0.05
    }
# ValueError: Cannot feed value of shape (1, 210, 160, 3) for Tensor u'Placeholder:0', which has shape '(?, 210)
    # environment params
    eparams = {
        'num_batches': 300,
        'ep_per_batch': 50
    }

    with tf.Session() as sess:  # tf.Session() as sess:  tf.Graph().as_default(), tf.Session() as sess

        agent = PolicyGradientAgent(hparams, sess)

        sess.run(tf.initialize_all_variables())

        for batch in xrange(eparams['num_batches']):
            time.sleep(1)

            print '=====\nBATCH {}\n===='.format(batch)

            b_obs, b_acts, b_rews = [], [], []

            for _ in xrange(eparams['ep_per_batch']):

                obs, acts, rews = policy_rollout(env, agent)
                print ('Ations: ', acts)

                print 'Episode steps: {}'.format(len(obs))

                b_obs.extend(obs)
                b_acts.extend(acts)

                advantages = process_rewards(rews)
                b_rews.extend(advantages)

            # update policy
            # normalize rewards; don't divide by 0
            b_rews = b_rews - np.mean(b_rews)
            std = np.std(b_rews)
            if std != 0.0:
                b_rews = b_rews / std

            # nuevas observaciones procesadas mandar
            agent.train_step(b_obs, b_acts, b_rews)

        env.monitor.close()


# if __name__ == "__main__":
    # main()
main()
