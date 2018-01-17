#! /usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import gym
from keras.models import Model
from keras.layers import Input, Dense, BatchNormalization, PReLU, Dropout
from keras.regularizers import l2
from keras.optimizers import RMSprop
from keras.engine import Layer
from keras import backend as K

env_name = 'Breakout-ram-v0'
_actions = (1, 4, 5)


def main():
    env = gym.make(env_name)
    #env = gym.wrappers.Monitor(env, env_name + '_history')

    n_max_episode = 10000
    play_length = 2**30
    warmingup_episode = 100
    memory_length = 1024 * 256
    gamma = 0.99
    epsilon = 1.0
    epsilon_decay = 0.998
    epsilon_min = 0.1
    learning_rate = 0.0001
    n_units = (128 * 8, 1024, 1024, len(_actions))

    agent = Agent(n_units, learning_rate, gamma, memory_length)

    for n_episode in range(n_max_episode):
        play(env, agent, play_length, epsilon)
        agent.train()
        if n_episode > warmingup_episode:
            epsilon = (epsilon - epsilon_min) * epsilon_decay + epsilon_min

    env.close()


def play(env, agent, play_length, epsilon=0):
    total_reward = 0.0
    observation = env.reset()
    ss = preprocess(observation)
    for n_frame in range(play_length):
        s = ss
        if n_frame < 20 or np.random.uniform() < epsilon:
            action = np.random.choice(len(_actions))
        else:
            action = agent.predict(s)
        observation, reward, done, info = env.step(_actions[action])
        total_reward += reward
        ss = preprocess(observation)
        agent.memory_append((s, action, np.sign(reward), done, ss))
        if done:
            break
    return total_reward


def preprocess(x):
    return np.asarray([(n >> i) & 1 for n in x for i in range(8)], dtype=np.bool)


class Agent(object):
    def __init__(self, n_units, sgd_lr, gamma, memory_length):
        nD, nV = n_units[0], n_units[-1]
        Wreg_l2 = 1e-6
        drop_proba = 0.2
        self.gamma = gamma
        self.train_batch_size = 128
        self.mem = PlayMemory(memory_length, nD)
        self.train_count = 0

        s = Input(shape=(nD,))
        a = Input(shape=(1,), dtype='uint8')

        z = s
        for nH in n_units[1:-1]:
            z = Dense(nH, W_regularizer=l2(Wreg_l2))(z)
            z = BatchNormalization()(z)
            z = PReLU()(z)
            z = Dropout(drop_proba)(z)
        z = Dense(nV + 1, W_regularizer=l2(Wreg_l2), init='zero')(z)

        self.Q_model = Model(input=s, output=DuelQ()(z))

        self.train_model = Model(input=[s, a], output=DuelQa()([z, a]))
        self.train_model.compile(
            loss=lambda y_true, y_pred: K.sqrt((y_true - y_pred)**2 + 1) - 1,
            optimizer=RMSprop(lr=sgd_lr))

    def train(self):
        m = self.mem
        n = len(m)
        if n < 16384 or m.add_count < 1024:
            return
        m.add_count = 0
        self.train_count += 1

        for i in range(0, n, self.train_batch_size):
            j = i + self.train_batch_size
            m.Q[i:j] += self.gamma * np.max(self.Q_model.predict_on_batch(
                m.S[i:j, 1]), axis=1, keepdims=True) * (m.T[i:j] == False)
            m.Q[i:j] /= 2

        idx = np.random.permutation(n)
        for n in range(self.train_batch_size, n, self.train_batch_size):
            i = idx[n - self.train_batch_size:n]
            self.train_model.train_on_batch(
                [m.S[i, 0], m.A[i]], m.R[i] + m.Q[i])

    def predict(self, s):
        return np.argmax(self.Q_model.predict_on_batch(s[np.newaxis]).squeeze())

    def memory_append(self, sars):
        self.mem.append(sars)


class DuelQ(Layer):
    def get_output_shape_for(self, input_shape):
        return None, input_shape[1] - 1

    def call(self, x, mask=None):
        return K.expand_dims(x[:, -1] - K.mean(x[:, :-1], axis=1)) + x[:, :-1]


class DuelQa(Layer):
    def get_output_shape_for(self, input_shape):
        return None, 1

    def call(self, z, mask=None):
        x, a = z
        return K.expand_dims(x[:, -1] - K.mean(x[:, :-1], axis=1) + x[K.arange(x.shape[0]), K.flatten(a)])


class PlayMemory(object):
    def __init__(self, length, s_size):
        self.max_length = length
        self.S = np.zeros((self.max_length, 2, s_size), dtype=np.bool)
        self.A = np.zeros((self.max_length, 1), dtype=np.int8)
        self.R = np.zeros((self.max_length, 1), dtype=np.float32)
        self.T = np.zeros((self.max_length, 1), dtype=np.bool)
        self.Q = np.zeros((self.max_length, 1), dtype=np.float32)
        self.length = 0
        self.add_count = 0
        self._i = self._rand_index()

    def __len__(self):
        return self.length

    def _rand_index(self):
        for i in range(self.max_length):
            self.length = i + 1
            yield i
        while True:
            for i in np.random.permutation(self.max_length):
                yield i

    def append(self, sars):
        (s, a, r, t, ss) = sars
        i = next(self._i)
        self.S[i, 0] = s
        self.S[i, 1] = ss
        self.A[i] = a
        self.R[i] = r
        self.T[i] = t
        self.Q[i] = 0
        self.add_count += 1


main()
