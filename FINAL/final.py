import sys
sys.path.insert(0,'/home/ubuntu/Desktop')
import gym
import tensorflow as tf
import numpy as np
import pygame


list_obs_por_juego, list_action_esperadas_por_juego = [], []

env = gym.make('MsPacman-ram-v0')
tam_teclas_disponibles = env.action_space.n  # el pacman es  9
print tam_teclas_disponibles


def correr_episodios_gym():
    for i_episode in range(15):
        observation = env.reset()
        aux_reward = 0
        reward = 0
        done = False
        aux_action, aux_obs = [], []
        while not done:
            env.render()
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            aux_reward += reward
            #print(observation)
            #print("*****************************************************************************")
            aux_action.append(action)
            aux_obs.append(observation)
            #print(aux_obs)
            #print("*****************************************************************************")

        if aux_reward > 250.0:
            print("este juego paso: ", i_episode)
            global list_obs_por_juego
            global list_action_esperadas_por_juego
            #list_action_esperadas_por_juego.append(aux_action)

            global list_action_esperadas_por_juego
            if len(list_obs_por_juego) == 0:
                list_obs_por_juego =  aux_obs
                list_action_esperadas_por_juego =  aux_action

            if len(list_obs_por_juego) > 0:
                 list_obs_por_juego = list_obs_por_juego +  aux_obs
                 list_action_esperadas_por_juego = list_action_esperadas_por_juego +  aux_action

            print("*****************************************************************************")  
            print(list_obs_por_juego)
            print("*****************************************************************************")  
        aux_reward = 0


#correr_episodios_gym()
#print(len(list_obs_por_juego))
#print(len(list_action_esperadas_por_juego))


a_0 = tf.placeholder(tf.float32, [None, 128])  # obser
y = tf.placeholder(tf.float32, [None, tam_teclas_disponibles])  # actions


#w_1 = tf.Variable(tf.truncated_normal([128, middle]))
#b_1 = tf.Variable(tf.truncated_normal([1 ,middle]))
#w_2 = tf.Variable(tf.truncated_normal([middle, tam_teclas_disponibles]))
#b_2 = tf.Variable(tf.truncated_normal([1 , tam_teclas_disponibles]))


middle = 30
w_1 = tf.Variable(tf.truncated_normal([128, middle]))
b_1 = tf.Variable(tf.truncated_normal([1, middle]))
w_2 = tf.Variable(tf.truncated_normal([middle, tam_teclas_disponibles]))
b_2 = tf.Variable(tf.truncated_normal([1, tam_teclas_disponibles]))





#np.vstack(vecAuxAct)

a = [[1,2],[3,4]]
b = [[5,6],[7,8]]
c = [[9,10],[11,12]]
d = [[13,14],[15,16]]
e = [[17,18],[19,20]]

f = a+b
f = f + c
f = f + d
f = f + e
print(f)