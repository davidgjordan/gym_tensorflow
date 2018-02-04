import tensorflow as tf
import numpy as np
import gym
import sys
def play(nameGame,tf_path):
     # # # # # # # # # # # # # # # # # # # # # # # # # # #
    env = gym.make(nameGame)
    xInputObservations = tf.placeholder(tf.float32, [None, 128])
    yExpectedActions = tf.placeholder(tf.float32, [None, 9])
    weigth = tf.Variable(tf.zeros([128, 9]))
    bias = tf.Variable(tf.zeros([9]))  # Vector con bias
    yOperForNodes = tf.matmul(xInputObservations, weigth) + bias
    funSoftmaxError = tf.nn.softmax_cross_entropy_with_logits(
        labels=yExpectedActions, logits=yOperForNodes)
    cost = tf.reduce_mean(funSoftmaxError)
    optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(cost)
    production = tf.argmax(yOperForNodes, 1)
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    # # # # # # # # # # #  # # # # # # # # # # # # # # # #
    saver = tf.train.Saver()
    aux_reward = 0
    with tf.Session() as sess:
        #saver.restore(sess,"./POC/agent/tmp_dos/model.ckpt")#tf._path
        saver.restore(sess,tf._path)#tf._path
        tf.global_variables_initializer()
        for i_episode in range(1):
            observation = env.reset()
            done = False
            while not done:
                env.render()
                action = sess.run(production, feed_dict={
                                xInputObservations: observation.reshape(1, 128)})
                observation, reward, done, info = env.step(action)
                aux_reward += reward
        sys.stdout.write(str(aux_reward))

#play("MsPacman-ram-v0", "../../bin/Games/pacman/T1/TensorFlow/model.ckpt")