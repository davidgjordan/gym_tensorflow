#!/usr/bin/env python
import sys
sys.path.insert(0, '/home/ubuntu/Desktop')
import gym
import tensorflow as tf
import numpy as np
import pygame
import random
from time import sleep

import os
# raw_input('Gracias') # esta linea permite que el programa no termine hasta que se de Enter


### # # # # COLORS  # # # # # # ## # #


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    Cyan = '\033[96m'
    White = '\033[97m'
    Magenta = '\033[95m'


# #  #  # # # # # # # # # GYM # # # # # # # # # # # # # # # # # # # # # # # # # #

list_obs_por_juego, list_action_esperadas_por_juego = [], []
gameName = 'MsPacman-ram-v0'
env = gym.make(gameName)
os.system('clear')
tam_teclas_disponibles = env.action_space.n  # el pacman es  9
print bcolors.Cyan + bcolors.BOLD + gameName + '\033[0m'


def printPacman():
    print bcolors.Magenta + bcolors.BOLD + ' _____   ______   _____   __   __   ______   __    _'
    sleep(0.23)
    print '|  _  | |  __  | |  ___| |  \_/  \ |  __  | |  \  | |'
    sleep(0.23)
    print '| |_| | | |__| | | |     | |\_/| | | |__| | | |\\\ | |'
    sleep(0.23)
    print '|  ___| |  __  | | |     | |   | | |  __  | | | \\\| |'
    sleep(0.23)
    print '| |     | |  | | | |___  | |   | | | |  | | | |  \  |'
    sleep(0.23)
    print '|_|     |_|  |_| |_____| |_|   |_| |_|  |_| |_|   \_|\033[0m'
    print ''


printPacman()


def display_arr(screen, arr, transpose, video_size):
    arr_min, arr_max = arr.min(), arr.max()
    arr = 255.0 * (arr - arr_min) / (arr_max - arr_min)
    pyg_img = pygame.surfarray.make_surface(
        arr.swapaxes(0, 1) if transpose else arr)
    pyg_img = pygame.transform.scale(pyg_img, video_size)
    screen.blit(pyg_img, (0, 0))


# # # # # # #RECOLECCION DE DATOS PARA EL ENTRENAMIENTO########################################
espectedReward = 150
successGamesCount = 2
sleep(1)
print bcolors.Cyan + bcolors.BOLD + 'RECOLECCION DE DATOS DE {0} JUEGOS CON RECOMPENSA MAYOR A {1} PARA EL ENTRENAMIENTO . . .\033[0m'.format(successGamesCount, espectedReward)


def recolectar_datos():
    global list_obs_por_juego
    global list_action_esperadas_por_juego
    successGame = 1
    gameNumber = 1
    while successGame <= successGamesCount:
        observation = env.reset()
        aux_reward = 0
        reward = 0
        done = False
        aux_action, aux_obs = [], []
        ########################################
        velocity = 15000
        clock = pygame.time.Clock()
        screen = pygame.display.set_mode((480, 630))  # 480, 630
        pygame.display.set_caption(u'OBTENIENDO DATA DE ENTRENAMIENTO')
        #########################################
        while not done:
            rgb_array = env.render(mode='rgb_array')  # env.render()
            # action = env.action_space.sample()
            action = random.choice([1, 2, 3, 4, 5, 6, 7, 8])
            observation, reward, done, info = env.step(action)
            aux_reward += reward

            aux_action.append(action)
            aux_obs.append(observation)
            #############################################
            if observation is not None:
                display_arr(screen, rgb_array, True, (480, 630))
            pygame.display.flip()
            clock.tick(velocity)
        ###################################################

        if aux_reward > espectedReward:
            print bcolors.OKGREEN + 'Juego Numero: {0} PASO - recompensa: {1} - juego: {2}/{3} \033[0m'.format(gameNumber, aux_reward, successGame, gameNumber)
            successGame = successGame + 1

            list_obs_por_juego.append(aux_obs)
            list_action_esperadas_por_juego.append(aux_action)
        else:
            print bcolors.WARNING + 'Juego Numero: {0} NO PASO\033[0m'.format(gameNumber)
        gameNumber = gameNumber + 1
        aux_reward = 0
    pygame.quit()


sleep(1)
recolectar_datos()  # DATA


##################NORMALIZE DATA################################
mat_normalize_obs = None
mat_normalize_actions = None
sleep(1)
print bcolors.Cyan + bcolors.BOLD + 'NORMALIZANDO LOS DATOS DE {0} JUEGOS PARA EL ENTRENAMIENTO . . .\033[0m'.format(successGamesCount)


def get_normalizar_actions():
    vec_aux_ac = []
    for vec_action_por_juego in list_action_esperadas_por_juego:
        for act in vec_action_por_juego:
            aux = np.zeros((tam_teclas_disponibles))
            aux[act] = 1
            vec_aux_ac.append(aux)
    mat_normalize_actions = np.vstack(vec_aux_ac)
    return mat_normalize_actions, vec_aux_ac


get_normalizar_actions()


def get_normalizar_observations():
    vec_aux_obs = []
    for vec_obs_por_juego in list_obs_por_juego:
        for obs in vec_obs_por_juego:
            vec_aux_obs.append(obs)
    mat_normalize_obs = np.vstack(vec_aux_obs)
    return mat_normalize_obs, vec_aux_obs


get_normalizar_observations()


tam_mat_act, _ = get_normalizar_actions()
tam_mat_obs, _ = get_normalizar_observations()
sleep(1)
print '...'  # len(tam_mat_act)
sleep(1)
print '...'  # len(tam_mat_obs)
sleep(1)
print '...'  # len(tam_mat_obs)


_, aux_obs_copy_pila = get_normalizar_observations()
_, aux_act_copy_pila = get_normalizar_actions()


def get_lote(tam_lote):
    lis_aux_obs = []
    lis_aux_act = []
    for i in range(tam_lote):
        data_o = aux_obs_copy_pila.pop()
        data_a = aux_act_copy_pila.pop()
        if data_o is not None:
            lis_aux_obs.append(data_o)
            lis_aux_act.append(data_a)
        else:
            break
    mat_normalize_obs = np.vstack(lis_aux_obs)
    mat_normalize_actions = np.vstack(lis_aux_act)
    return mat_normalize_obs, mat_normalize_actions


########### NEURONAL NETWORK###########
sleep(1)
print bcolors.Cyan + bcolors.BOLD + 'CREANDO Y CONFIGURANDO LA RED NEURONAL . . .\033[0m'

_sizeInputX = 128
_sizeNumberKeysY = 9

a_0 = tf.placeholder(tf.float32, [None, _sizeInputX])
y = tf.placeholder(tf.float32, [None, _sizeNumberKeysY])

middle = 30
w_1 = tf.Variable(tf.truncated_normal([_sizeInputX, middle]))
b_1 = tf.Variable(tf.truncated_normal([1, middle]))
w_2 = tf.Variable(tf.truncated_normal([middle, _sizeNumberKeysY]))
b_2 = tf.Variable(tf.truncated_normal([1, _sizeNumberKeysY]))


def sigma(x):
    return tf.div(tf.constant(1.0),
                  tf.add(tf.constant(1.0), tf.exp(tf.negative(x))))


z_1 = tf.add(tf.matmul(a_0, w_1), b_1)
a_1 = sigma(z_1)
z_2 = tf.add(tf.matmul(a_1, w_2), b_2)
a_2 = sigma(z_2)

diff = 0.5 * (tf.subtract(a_2, y))**2
###########################
cost = tf.multiply(diff, diff)
optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(cost)
###########################

prod = tf.argmax(a_2, 1)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())


#####RUNING  TRAINING#############
for i_episode in range(successGamesCount + 1):
    sleep(1.5)
    print bcolors.HEADER + bcolors.BOLD + 'Entrenando la red . . .\033[0m'

epoca_i = 0
while len(aux_obs_copy_pila) != 0:
    lotex, lotey = get_lote(1)  # mejor
    # print "**************************************************************"
    # print lotey
    # print "**************************************************************"

    if len(list(lotex)) != 0:
        lotex = np.divide(lotex, 255.0)
        lotey = np.divide(lotey, 255.0)
        sess.run(optimizer, feed_dict={
            a_0: lotex, y: lotey})
    epoca_i += 1
sleep(1)
print bcolors.OKGREEN + bcolors.BOLD + 'Red entrenada!!!\033[0m'

########PLAY GAME AFTER TRAIN##################

testGames = 5

sleep(1)
print bcolors.Cyan + bcolors.BOLD + 'PROBANDO LA RED ENTRENADA EN {0} JUEGOS . . .\033[0m'.format(testGames)


def play_red_entrenada():
    gameNumber = 1
    successGame = 1
    for i_episode in range(testGames):
        observation = env.reset()
        aux_reward = 0
        reward = 0
        done = False
        list_aux_ac = []
        ########################################
        velocity = 30
        clock = pygame.time.Clock()
        screen = pygame.display.set_mode((480, 630))
        pygame.display.set_caption(
            u'JUEGOS DE PRUEBA DESPUES DEL ENTRENAMIENTO')
        #########################################
        while not done:
            # env.render()
            rgb_array = env.render(mode='rgb_array')
            observation = np.divide(observation, 255.0)

            action = sess.run(prod, feed_dict={
                a_0: observation.reshape(1, 128)})
            # print 'action elegida: {0}'.format(action)

            observation, reward, done, info = env.step(action)
            aux_reward += reward
            list_aux_ac.append(action[0])
            #############################################
            if observation is not None:
                display_arr(screen, rgb_array, True, (480, 630))
            pygame.display.flip()
            clock.tick(velocity)
        ###################################################
        if aux_reward > espectedReward:
            print bcolors.OKGREEN + bcolors.BOLD + 'Juego Numero: {0} PASO - recompensa: {1} - juego: {2}/{3}\033[0m'.format(gameNumber, aux_reward, successGame, gameNumber)
            successGame = successGame + 1
        else:
            print bcolors.WARNING + 'Juego Numero: {0} NO PASO\033[0m'.format(gameNumber)
        gameNumber = gameNumber + 1
        aux_reward = 0
    pygame.quit()


sleep(1)
play_red_entrenada()
saver = tf.train.Saver()
# el ocho dio bien no mas
save_path = saver.save(sess, "./tmp_doce_f/model.ckpt")
sleep(1)
print bcolors.HEADER + 'Model saved in path: {0}\033[0m'.format(save_path)
