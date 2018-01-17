# -*- coding: utf-8 -*-
from __future__ import division
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import gym
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('Blackjack-v0')
graph, = plt.plot([], [])

# (player sum, dealer sum, player usable Ace?)
num_observations = len(env.reset())
# 2 actions: 1 hit 0 stick
num_actions = env.action_space.n

# feed-forward
X = tf.placeholder(shape=[1, num_observations], dtype=tf.float32)
W = tf.get_variable('W', shape=[num_observations, num_actions],
                    initializer=tf.contrib.layers.xavier_initializer())
Q = tf.nn.tanh(tf.matmul(X, W))
# predicted maximizing action
Qmax = tf.argmax(Q)

# stochastic gradient descent
futureQ = tf.placeholder(shape=[1, num_actions], dtype=tf.float32)
reward = tf.placeholder(shape=[], dtype=tf.float32)
# minimizing squared loss
loss = tf.reduce_sum(tf.square(futureQ - Q))  # + regularization_cost
adamoptimizer = tf.train.AdamOptimizer(0.01)
update = adamoptimizer.minimize(loss)

init = tf.global_variables_initializer()

# learning params
num_episodes = 1000000
num_steps = 5
epsilon = 0.1
decay = 0.99

saver = tf.train.Saver()

wins = 0
draws = 0
loses = 0

winlist = []
drawlist = []
loselist = []

with tf.Session() as sess:
    sess.run(init)
    #done_action = np.zeros([1,num_actions])
    for ith_episode in range(num_episodes):
            # random input Q
        s = env.reset()
        r = 0
        for step in range(num_steps):
            s = np.array(s).reshape(1, num_observations)
            action, allQ = sess.run([Qmax, Q], feed_dict={X: s})
            # random action with probability epsilon
            if np.random.rand(1) < epsilon:
                action[0] = env.action_space.sample()
            # observe results
            s_prime, r_prime, done, _ = env.step(action[0])
            # get Q prime
            s_prime = np.array(s_prime).reshape(1, num_observations)
            Q_togo = sess.run(Qmax, feed_dict={X: s_prime})
            # get max Q prime for gradient update
            maxQ_togo = np.max(Q_togo)
            bellman = allQ
            bellman[0, action[0]] = r_prime + decay * maxQ_togo
            # update weights
            _, W1 = sess.run([update, W], feed_dict={X: s, futureQ: bellman})
            r += r_prime
            s = s_prime
            if done:
                epsilon = 1. / ((ith_episode / 50) + 10)
                break
        if r == 1:
            wins += 1
        elif r == 0:
            draws += 1
        else:
            loses += 1
        if (ith_episode % 1000 == 0):
            winpercent = wins / (ith_episode + 1) * 100
            winlist.append(winpercent)
            drawpercent = draws / (ith_episode + 1) * 100
            drawlist.append(drawpercent)
            losepercent = loses / (ith_episode + 1) * 100
            loselist.append(losepercent)
            print '\n'
            print "Percent winning: " + str(winpercent) + "%"
            print "Percent drawing: " + str(drawpercent) + "%"
            print "Percent losing: " + str(losepercent) + "%"
            print "out of " + str(ith_episode) + " episodes"
    save_path = saver.save(sess, "test.ckpt")

    plt.plot(winlist)
    plt.plot(drawlist)
    plt.plot(loselist)
    plt.legend(['wins', 'draws', 'losses'], loc='upper right')
    plt.show()
