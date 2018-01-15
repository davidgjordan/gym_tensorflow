"""
Solve OpenAI Gym Cartpole V1 with DQN.
"""
import gym

import numpy as np
import tensorflow as tf
import math


# Hyperparameters
envSize = 4
H = 100  # number of neurons in hidden layer
batch_number = 50  # size of batches for training
learn_rate = .01
gamma = 0.99


def reduced_rewards(r):
    reduced_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(xrange(0, r.size)):
        running_add = running_add * gamma + r[t]
        reduced_r[t] = running_add
    return reduced_r


if __name__ == '__main__':

    env = gym.make('CartPole-v1')
    #env.monitor.start('training_dir', force=True)
    # Setup tensorflow
    tf.reset_default_graph()

    observations = tf.placeholder(tf.float32, [None, envSize], name="input_x")
    w1 = tf.get_variable("w1", shape=[envSize, H],
                         initializer=tf.contrib.layers.xavier_initializer())
    hidden_layer_1 = tf.nn.relu(tf.matmul(observations, w1))
    w15 = tf.get_variable("w15", shape=[H, H],
                          initializer=tf.contrib.layers.xavier_initializer())
    hidden_layer_2 = tf.nn.relu(tf.matmul(hidden_layer_1, w15))
    w2 = tf.get_variable("w2", shape=[H, 1],
                         initializer=tf.contrib.layers.xavier_initializer())
    result_score = tf.matmul(hidden_layer_2, w2)
    probablility = tf.nn.sigmoid(result_score)

    training_variables = tf.trainable_variables()

    input_y = tf.placeholder(tf.float32, [None, 1], name="input_y")#cambie de 1
    advantage = tf.placeholder(tf.float32, name="reward_signal")

    # Loss Function
    loss = -tf.reduce_mean((tf.log(input_y - probablility)) * advantage)

    new_gradients = tf.gradients(loss, training_variables)

    # Training

    adam = tf.train.AdamOptimizer(learning_rate=learn_rate)
    #adam = tf.train.GradientDescentOptimizer(0.1).minimize(cost)
    #adam = tf.train.GradientDescentOptimizer(learning_rate=learn_rate)

    w1_gradent = tf.placeholder(tf.float32, name="batch_gradent1")

    w2_gradent = tf.placeholder(tf.float32, name="batch_gradent2")

    batch_gradent = [w1_gradent, w2_gradent]
    update_gradent = adam.apply_gradients(
        zip(batch_gradent, training_variables))

    max_episodes = 2000
    max_steps = 500

    xs, hs, dlogps, drs, ys, tfps = [], [], [], [], [], []
    running_reward = None
    reward_sum = 0
    episode_number = 1

    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)

        # setting up the training variables
        gradBuffer = sess.run(training_variables)
        for ix, grad in enumerate(gradBuffer):
            gradBuffer[ix] = grad * 0

        for episode in xrange(max_episodes):
            observation = env.reset()
            for step in xrange(max_steps):
                if(step == (max_steps - 1)):
                    print 'Made 500 steps!'
                env.render()
                x = np.reshape(observation, [1, envSize])

                # get action from policy
                tfprob = sess.run(probablility, feed_dict={observations: x})

                action = 1 if np.random.uniform() < tfprob else 0

                #action = env.action_space.sample()
                print("np.random.uniform(): ", np.random.uniform())
                print("tfprob: ", tfprob)
                print("action: ", action)



                # will need to rework action to be more generic, not just 1 or 0

                xs.append(x)  # observation
                y = 1 if action == 0 else 0  # something about fake lables, need to investigate
                ys.append(y)

                # run an action
                observation, reward, done, info = env.step(action)
                reward_sum += reward

                drs.append(reward)

                if done:
                    episode_number += 1
                    print 'Episode %f: Reward: %f' % (episode_number, reward_sum)
                    # putting together all inputs, is there a better way to do this?
                    epx = np.vstack(xs)
                    epy = np.vstack(ys)
                    epr = np.vstack(drs)
                    tfp = tfps
                    xs, hs, dlogpr, drs, ys, tfps = [], [], [], [], [], []  # reset for next episode

                    # compute reward
                    discounted_epr = reduced_rewards(epr)
                    discounted_epr -= np.mean(discounted_epr)
                    discounted_epr /= np.std(discounted_epr)

                    # get gradient, save in gradent_buffer
                    tGrad = sess.run(new_gradients, feed_dict={
                                     observations: epx, input_y: epy, advantage: discounted_epr})
                    for ix, grad in enumerate(tGrad):
                        gradBuffer[ix] += grad

                    if episode_number % batch_number == 0:
                        sess.run(update_gradent, feed_dict={
                                 w1_gradent: gradBuffer[0], w2_gradent: gradBuffer[1]})
                        for ix, grad in enumerate(gradBuffer):
                            gradBuffer[ix] = grad * 0

                        running_reward = reward_sum if running_reward is None else (
                            ((running_reward * episode_number - 50) + (reward_sum * 50)) / episode_number)
                        print 'Average reward for episode %f. total average reward %f' % (reward_sum / batch_number, running_reward / batch_number)

                        if reward_sum / batch_number > 475:
                            print 'Task solved in', episode_number, 'episodes!'
                            reward_sum = 0
                            break
                        reward_sum = 0
                    break

    env.monitor.close()

('np.random.uniform(): ', 0.7534640259766103)
('tfprob: ', array([[0.500481]], dtype=float32))
('action: ', 1)
('np.random.uniform(): ', 0.9189687708312633)
('tfprob: ', array([[0.51253]], dtype=float32))
('action: ', 1)
('np.random.uniform(): ', 0.22598798418651)
('tfprob: ', array([[0.52526385]], dtype=float32))
('action: ', 1)
('np.random.uniform(): ', 0.8207490984718717)
('tfprob: ', array([[0.5382662]], dtype=float32))
('action: ', 1)
('np.random.uniform(): ', 0.654106041145196)
('tfprob: ', array([[0.5516434]], dtype=float32))
('action: ', 0)
('np.random.uniform(): ', 0.3444672236068924)
('tfprob: ', array([[0.5402331]], dtype=float32))
('action: ', 1)
('np.random.uniform(): ', 0.7220401751941746)
('tfprob: ', array([[0.554033]], dtype=float32))
('action: ', 0)
('np.random.uniform(): ', 0.08173039188343689)
('tfprob: ', array([[0.54262364]], dtype=float32))
('action: ', 1)
('np.random.uniform(): ', 0.9334915721881869)
('tfprob: ', array([[0.5568565]], dtype=float32))
('action: ', 0)
('np.random.uniform(): ', 0.6824650517988199)
('tfprob: ', array([[0.5451785]], dtype=float32))
('action: ', 1)
('np.random.uniform(): ', 0.08064703674564855)
('tfprob: ', array([[0.5599441]], dtype=float32))
('action: ', 1)
('np.random.uniform(): ', 0.47745689650216006)
('tfprob: ', array([[0.57484484]], dtype=float32))
('action: ', 0)
('np.random.uniform(): ', 0.9944547830747118)
('tfprob: ', array([[0.5640294]], dtype=float32))
('action: ', 0)
Episode 2.000000: Reward: 13.000000
('np.random.uniform(): ', 0.16392049823397437)
('tfprob: ', array([[0.5016257]], dtype=float32))
('action: ', 0)
('np.random.uniform(): ', 0.9084019675971512)
('tfprob: ', array([[0.50850356]], dtype=float32))
('action: ', 1)
('np.random.uniform(): ', 0.8848029968650759)
('tfprob: ', array([[0.5012737]], dtype=float32))
('action: ', 1)
('np.random.uniform(): ', 0.6532941331445336)
('tfprob: ', array([[0.51342654]], dtype=float32))
('action: ', 1)
('np.random.uniform(): ', 0.15800446703328264)
('tfprob: ', array([[0.5261485]], dtype=float32))
('action: ', 0)
('np.random.uniform(): ', 0.3951076380738663)
('tfprob: ', array([[0.5136844]], dtype=float32))
('action: ', 1)
('np.random.uniform(): ', 0.9118166185423064)
('tfprob: ', array([[0.52661586]], dtype=float32))
('action: ', 1)
('np.random.uniform(): ', 0.09320027945628762)
('tfprob: ', array([[0.5398866]], dtype=float32))
('action: ', 1)
('np.random.uniform(): ', 0.3595280878104159)
('tfprob: ', array([[0.5535285]], dtype=float32))
('action: ', 1)
('np.random.uniform(): ', 0.25369144558003487)
('tfprob: ', array([[0.5675467]], dtype=float32))
('action: ', 0)
('np.random.uniform(): ', 0.3697969019433289)
('tfprob: ', array([[0.5562303]], dtype=float32))
('action: ', 0)
('np.random.uniform(): ', 0.6529918009814187)
('tfprob: ', array([[0.5445305]], dtype=float32))
('action: ', 1)
('np.random.uniform(): ', 0.054664072133043806)
('tfprob: ', array([[0.5591081]], dtype=float32))
('action: ', 0)
('np.random.uniform(): ', 0.19360226228709254)
('tfprob: ', array([[0.5476964]], dtype=float32))
('action: ', 1)
('np.random.uniform(): ', 0.05330001607048962)
('tfprob: ', array([[0.56266516]], dtype=float32))
('action: ', 0)
('np.random.uniform(): ', 0.4637281027453366)
('tfprob: ', array([[0.55167997]], dtype=float32))
('action: ', 1)
('np.random.uniform(): ', 0.7600725657383434)
('tfprob: ', array([[0.5670093]], dtype=float32))
('action: ', 1)
Episode 3.000000: Reward: 30.000000
('np.random.uniform(): ', 0.9064248292138697)
('tfprob: ', array([[0.4989275]], dtype=float32))
('action: ', 0)