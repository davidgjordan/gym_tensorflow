#!/usr/bin/env python
import sys
sys.path.insert(0, '/home/ubuntu/Desktop')
import gym
import tensorflow as tf
import numpy as np
import pygame
import random
from time import sleep
import time

import os
# esta linea permite que el programa no termine hasta que se de Enter
# raw_input('Gracias')


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


##################################################config GYM###############################################


list_obs_por_juego, list_action_esperadas_por_juego = [], []
gameName = 'MsPacman-ram-v0'
env = gym.make(gameName)
os.system('clear')
tam_teclas_disponibles = env.action_space.n  # el pacman es  9
print bcolors.Cyan + bcolors.BOLD + gameName + '\033[0m'
##################################################config GYM###############################################


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

##################################################IMAGEN###############################################


def display_arr(screen, arr, transpose, video_size):
    arr_min, arr_max = arr.min(), arr.max()
    arr = 255.0 * (arr - arr_min) / (arr_max - arr_min)
    pyg_img = pygame.surfarray.make_surface(
        arr.swapaxes(0, 1) if transpose else arr)
    pyg_img = pygame.transform.scale(pyg_img, video_size)
    screen.blit(pyg_img, (0, 0))
##################################################IMAGEN###############################################



######################################RECOLECCION DE DATOS PARA EL ENTRENAMIENTO########################################
espectedReward = 150
successGamesCount = 3
sleep(1)
raw_input(bcolors.Cyan + bcolors.BOLD +
          '1. RECOLECCCION DE DATOS. . . PULSE ENTER PARA CONTINUAR ')
print bcolors.Cyan + bcolors.BOLD + 'RECOLECTANDO DATOS DE {0} JUEGOS CON RECOMPENSA MAYOR A {1} PARA EL ENTRENAMIENTO . . .\033[0m'.format(successGamesCount, espectedReward)


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
        #velocity = 15000
        #clock = pygame.time.Clock()
        # screen = pygame.display.set_mode((480, 630))  # 480, 630
        #pygame.display.set_caption(u'OBTENIENDO DATA DE ENTRENAMIENTO')
        #########################################
        while not done:
            rgb_array = env.render(mode='rgb_array')  # env.render()
            # action = env.action_space.sample()
            action = random.choice([1, 2, 3, 4, 5, 6, 7, 8])
            # CUANTO DE REWARD OBTENGO POR N CANTIDAS DE ACCIONES, CUANTO DE REWARD PUEDO CONSEGUIR CON 1 O 2 O 3 VIDAS
            observation, reward, done, info = env.step(action)
            aux_reward += reward

            aux_action.append(action)
            aux_obs.append(observation)
            #############################################
            # if observation is not None:
            #    display_arr(screen, rgb_array, True, (480, 630))
            # pygame.display.flip()
            # clock.tick(velocity)
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
    # pygame.quit()


sleep(1)
recolectar_datos()  # DATA
print bcolors.OKGREEN + bcolors.BOLD + 'Recoleccion de datos exitosa\033[0m'
print ''
######################################RECOLECCION DE DATOS PARA EL ENTRENAMIENTO########################################


################################################NORMALIZE DATA#######################################################
mat_normalize_obs = None
mat_normalize_actions = None
sleep(1)
raw_input(bcolors.Cyan + bcolors.BOLD +
          '2. NORMALIZACION DE DATOS. . . PULSE ENTER PARA CONTINUAR ')
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
for i_episode in range(3):
    sleep(1.3)
    print bcolors.HEADER + bcolors.BOLD + 'Normalizando datos . . .\033[0m'


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


print bcolors.OKGREEN + bcolors.BOLD + 'Normalizacion exitosa . . .\033[0m'
print ''
################################################NORMALIZE DATA#######################################################


#################################################### NEURONAL NETWORK################################################
sleep(1)
raw_input(bcolors.Cyan + bcolors.BOLD +
          '3. CREACION DE LA RED NEURONAL. . . PULSE ENTER PARA CONTINUAR ')
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

# sacamos el error cuadratico de las salidas esperadas y el numero de nodos de salida
diff = 0.5 * (tf.subtract(a_2, y))**2
###########################
cost = tf.multiply(diff, diff)
# minimizamos el error con el metodo Gradient de tensorflow el cual reajusta los pesos segun un error o cost
optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(cost)
###########################

prod = tf.argmax(a_2, 1)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
sleep(1)
print bcolors.OKGREEN + bcolors.BOLD + 'Red creada con exito . . .\033[0m'
print ''
sleep(1)
#################################################### NEURONAL NETWORK################################################


#########################################################RUNING  TRAIN ING###########################################
raw_input(bcolors.Cyan + bcolors.BOLD +
          '4. ENTRENAMIENTO DE LA RED NEURONAL. . . PULSE ENTER PARA CONTINUAR ')
print bcolors.Cyan + bcolors.BOLD + 'ENTRENANDO LA RED NEURONAL . . .\033[0m'
for i_episode in range(successGamesCount + 1):
    sleep(1.5)
    print bcolors.HEADER + bcolors.BOLD + 'Entrenando la red . . .\033[0m'

epoca_i = 0
while len(aux_obs_copy_pila) != 0:
    lotex, lotey = get_lote(1)  # mejor
    if len(list(lotex)) != 0:
        lotex = np.divide(lotex, 255.0)
        lotey = np.divide(lotey, 255.0)
        sess.run(optimizer, feed_dict={
            a_0: lotex, y: lotey})
    epoca_i += 1
sleep(1)
print bcolors.OKGREEN + bcolors.BOLD + 'Red entrenada!!!\033[0m'
print ''
#########################################################RUNING  TRAIN ING###########################################


#########################################################PLAY GAME AFTER TRAIN#######################################
testGames = 5
sleep(1)
raw_input(bcolors.Cyan + bcolors.BOLD +
          '5. PRUEBA DE LA RED NEURONAL ENTRENADA. . . PULSE ENTER PARA CONTINUAR ')
print bcolors.Cyan + bcolors.BOLD + 'PROBANDO LA RED ENTRENADA PARA UNA RECOMPENSA MAYOR A {0} EN {1} JUEGOS . . .\033[0m'.format(espectedReward, testGames)
# CUANTO DE REWARD OBTENGO POR N CANTIDAS DE ACCIONES, CUANTO DE REWARD PUEDO CONSEGUIR CON 1 O 2 O 3 VIDAS


def play_red_entrenada():
    gameNumber = 1
    successGame = 1
    lives = 3
    banderaNumeroJuegos = True
    banderaVidas = True
    rewarPorVida = 0
    rewardActual = 0
    teclaActual = 3
    secuenciaDeTeclasPresionadas = []
    tiempoInicio = 0
    tiempoFin = 0
    try:
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
            tiempoInicio = time.time()
            while not done:
                # env.render()
                rgb_array = env.render(mode='rgb_array')
                observation = np.divide(observation, 255.0)

                action = sess.run(prod, feed_dict={
                    a_0: observation.reshape(1, 128)})
                # print 'action elegida: {0}'.format(action)
                teclaActual = action
                secuenciaDeTeclasPresionadas.append(teclaActual)

                observation, reward, done, info = env.step(action)
                aux_reward += reward
                rewardActual = aux_reward
                list_aux_ac.append(action[0])
                rewarPorVida = rewarPorVida + reward
                #############################################
                if observation is not None:
                    display_arr(screen, rgb_array, True, (480, 630))
                pygame.display.flip()
                clock.tick(velocity)
            ###################################################
                if banderaNumeroJuegos:
                    sleep(0.7)
                    print ''
                    print bcolors.WARNING + bcolors.BOLD + '----------------------Informacion Juego Numero: {0}--------------------------\033[0m'.format(gameNumber)
                    sleep(0.7)
                banderaNumeroJuegos = False

                if banderaVidas:
                    print bcolors.WARNING + bcolors.BOLD + 'Numero de vidas disponibles: {0}\033[0m'.format(lives)
                    rewarPorVida = 0
                    banderaVidas = False

                if lives != info['ale.lives']:
                    print bcolors.WARNING + bcolors.BOLD + 'Recompensa obtenida: {0}\033[0m'.format(rewarPorVida)
                    sleep(0.7)
                    lives = info['ale.lives']
                    banderaVidas = True

                tiempoFin = int(time.time() - tiempoInicio)

            lives = 3
            banderaNumeroJuegos = True
            secuenciaDeTeclasPresionadas = []
            tiempoInicio = 0
            if aux_reward > espectedReward:
                print bcolors.OKGREEN + bcolors.BOLD + 'Juego Numero: {0} PASO - recompensa total: {1} - juego: {2}/{3}\033[0m'.format(gameNumber, aux_reward, successGame, gameNumber)
                print bcolors.WARNING + bcolors.BOLD + '---------------------------------------------------------------------------\033[0m'
                successGame = successGame + 1
            else:
                print bcolors.WARNING + 'Juego Numero: {0} NO PASO\033[0m'.format(gameNumber)
            gameNumber = gameNumber + 1
            aux_reward = 0
    except KeyboardInterrupt as e:
        print bcolors.FAIL + bcolors.BOLD + '\rRERROR EN EL PACMAN. . . PARAMETROS DEL ERROR\033[0m'
        sleep(1)
        print ''
        print bcolors.FAIL + bcolors.BOLD + '\tNro de juego: {0}\033[0m'.format(gameNumber)
        sleep(0.3)
        print bcolors.FAIL + bcolors.BOLD + '\tUltima tecla presionada: {0}\033[0m'.format(teclaActual)
        sleep(0.3)
        print bcolors.FAIL + bcolors.BOLD + '\tRecompensa total Obtenida: {0}\033[0m'.format(rewardActual)
        sleep(0.3)
        print bcolors.FAIL + bcolors.BOLD + '\tCantidad de vidas restantes: {0}\033[0m'.format(lives)
        sleep(0.3)
        print bcolors.FAIL + bcolors.BOLD + '\tTiempo de duracion del juego: {0} segundos\033[0m'.format(tiempoFin)
        sleep(0.3)
        print bcolors.FAIL + bcolors.BOLD + '\tCantidad de teclas presionadas: {0}\033[0m'.format(len(secuenciaDeTeclasPresionadas))
        sleep(0.3)
        print bcolors.FAIL + bcolors.BOLD + '\tSecuencia de teclas presionadas: \033[0m'
        sleep(0.3)
        c = 0
        fin = '\t\t'
        for d in secuenciaDeTeclasPresionadas:
            ac = '[ {0} ] '.format(d)
            if c <= 10:
                fin = fin + ac
                c = c + 1
            else:
                fin = '{0}\n\t\t'.format(fin)
                c = 0

        print bcolors.FAIL + '{0}\033[0m'.format(fin)
        sleep(0.5)
        # manejar el error y decir en q numero de juego fue, en q vida cuando se presionaba q tecla cuando quedaban x vidas etc
        # print bcolors.Cyan + bcolors.BOLD + 'Desea seguir probando la red?  si/no \033[0m'
        continuar = ''
        continuar = raw_input(bcolors.Cyan + bcolors.BOLD +
                              'Desea seguir probando la red?  si/no \033[0m')
        if continuar == 'si' or continuar == 'SI' or continuar == 'Si' or continuar == 'sI':
            play_red_entrenada()
        else:
            pass

    finally:
        print ''
    pygame.quit()


sleep(1)
play_red_entrenada()
print bcolors.WARNING + bcolors.BOLD + 'FIN, BOT PACMAN!\033[0m'
print ''
#########################################################PLAY GAME AFTER TRAIN########################################################


#########################################################SAVE MODEL########################################################
sleep(1)
raw_input(bcolors.Cyan + bcolors.BOLD +
          '6. SALVAR EL MODELO DE LA RED NEURONAL ENTRENADA. . . PULSE ENTER PARA CONTINUAR')
print bcolors.Cyan + bcolors.BOLD + 'SALVANDO EL MODELO DE LA RED NEURONAL ENTRENADA . . .\033[0m'
saver = tf.train.Saver()
# el ocho dio bien no mas
save_path = saver.save(sess, "./tmp_tres_f3/model.ckpt")
sleep(1)
print bcolors.OKGREEN + 'Modelo salvado en la direccion: {0}\033[0m'.format(save_path)
print ''
#########################################################SAVE MODEL########################################################


# while True:
#   try:
#       x = int(raw_input("Please enter a number: "))
#       break
#   except ValueError:
#       print "Oops!  That was no valid number.  Try again..."
