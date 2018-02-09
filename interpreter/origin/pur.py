#!/usr/bin/env python
import gym
import tensorflow as tf
import numpy as np
import sys
sys.path.insert(0,'/home/ubuntu/Desktop')
import pygame


# #  #  # # # # # # # # # GYM # # # # # # # # # # # # # # # # # # # # # # # # # #

list_obs_por_juego, list_action_esperadas_por_juego = [], []

env = gym.make('MsPacman-ram-v0')
tam_teclas_disponibles = env.action_space.n  # el pacman es  9
print tam_teclas_disponibles
def display_arr(screen, arr, transpose, video_size):
    arr_min, arr_max = arr.min(), arr.max()
    arr = 255.0 * (arr - arr_min) / (arr_max - arr_min)
    pyg_img = pygame.surfarray.make_surface(arr.swapaxes(0, 1) if transpose else arr)
    pyg_img = pygame.transform.scale(pyg_img, video_size)
    screen.blit(pyg_img, (0,0))

def correr_episodios_gym():
    global list_obs_por_juego
    global list_action_esperadas_por_juego
    for i_episode in range(50):
        observation = env.reset()
        aux_reward = 0
        reward = 0
        done = False
        aux_action, aux_obs = [], []
        ########################################
        velocity = 15000
        clock = pygame.time.Clock()
        screen = pygame.display.set_mode((480,630))
        pygame.display.set_caption(u'DEVINT-24 GAMES - NNAgent')
        #########################################
        while not done:
            #env.render()
            action = env.action_space.sample()
            observation, reward, done, rgb_array = env.step(action)
            aux_reward += reward

            aux_action.append(action)
            aux_obs.append(observation)
            #############################################
            if observation is not None:################
                display_arr(screen, rgb_array, True, (480,630))
            pygame.display.flip()
            clock.tick(velocity)
        ###################################################

        if aux_reward > 300:
            list_obs_por_juego = np.vstack(aux_obs)
            list_action_esperadas_por_juego = np.vstack(aux_action)
            print("este juego paso: ", i_episode)
            print("rear: ", aux_reward)
        aux_reward = 0
    pygame.quit()


#correr_episodios_gym()
#print(list_obs_por_juego)
#print(list_action_esperadas_por_juego)

# #  #  # # # # # # # # # FIN  GYM # # # # # # # # # # # # # # # # # # # # # # # # # #


# #  #  # # # # # # # # # NORMALIZE# # # # # # # # # # # # # # # # # # # # # # # # # #

def getNormalizeActionsDefault(listExpectedActionForGame, numberKeysY):
    vecAuxAct = []
    for act in listExpectedActionForGame:
        aux = np.zeros((numberKeysY))
        aux[int(act)] = 1
        vecAuxAct.append(aux)
    matNormalizeActions = np.vstack(vecAuxAct)
    return matNormalizeActions, vecAuxAct


def getNormalizeObservationsDefault(listExpectedObsForGame):
    vecAuxObs = []
    for obs in listExpectedObsForGame:
        obs = np.divide(obs, 255.0)
        vecAuxObs.append(obs)
    matNormalizeObs = np.vstack(vecAuxObs)
    return matNormalizeObs, vecAuxObs


def getNormalizeObservationsFilter(listExpectedObsForGame, vecIndexFilter):
    vecAuxObs = []
    for obs in listExpectedObsForGame:
        obs = np.divide(obs, 255.0)
        obs = np.delete(obs, vecIndexFilter)
        vecAuxObs.append(obs)
    matNormalizeObs = np.vstack(vecAuxObs)
    return matNormalizeObs, vecAuxObs
pilaO, pilaA = [], []

def getBatch(sizeBatch):
    lisAuxObs = []
    lisAuxAct = []
    global pilaA
    global pilaO
    for i in range(sizeBatch):
        if len(pilaO) != 0:
            dataO = pilaO.pop()
            dataA = pilaA.pop()
            lisAuxObs.append(dataO)
            lisAuxAct.append(dataA)
        else:
            break
    matNormalizeObs = []
    matNormalizeActions = []
    if len(lisAuxAct) != 0:
        matNormalizeObs = np.vstack(lisAuxObs)
        matNormalizeActions = np.vstack(lisAuxAct)
    else:
        matNormalizeObs = []
        matNormalizeActions = []
    return matNormalizeObs, matNormalizeActions
# #  #  # # # # # # # # # FIN NORMALIZE # # # # # # # # # # # # # # # # # # # # # # # # # #


##############FUNCION DE APLICACION O VIABILIDAD###########
def sigma(x):
    return tf.div(tf.constant(1.0),
                  tf.add(tf.constant(1.0), tf.exp(tf.negative(x))))




def train(numberCiclos, numGames,  sizeInputX,  sizeNumberKeysY, observations, actions, vecIndexFilter, nameGame, expectedReward, path):
    print len(observations)
    print len(actions)
    ###########################RED####################################
    a_0 = tf.placeholder(tf.float32, [None, sizeInputX])
    y = tf.placeholder(tf.float32, [None, sizeNumberKeysY])
    middle = 30
    w_1 = tf.Variable(tf.truncated_normal([sizeInputX, middle]))
    b_1 = tf.Variable(tf.truncated_normal([1, middle]))
    w_2 = tf.Variable(tf.truncated_normal([middle, sizeNumberKeysY]))
    b_2 = tf.Variable(tf.truncated_normal([1, sizeNumberKeysY]))
    z_1 = tf.add(tf.matmul(a_0, w_1), b_1)
    a_1 = sigma(z_1)
    z_2 = tf.add(tf.matmul(a_1, w_2), b_2)
    a_2 = sigma(z_2)
    diff = 0.5 * (tf.subtract(a_2, y))**2
    ###########################BACK PROPAGATION###################
    cost = tf.multiply(diff, diff)
    optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(cost)
    ###########################################################
    prod = tf.argmax(a_2, 1)
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    env = gym.make(nameGame)
    ########################TRAINIGN#######################
    # # # # # # # # # # #  # # # # # # # # # # # # # # # #
    _, auxObsCopyPila = getNormalizeObservationsDefault(observations)
    _, auxActCopyPila = getNormalizeActionsDefault(actions, sizeNumberKeysY)
    matchesObs = []
    matchesAct = []
    matchesObs.extend(auxObsCopyPila)
    matchesAct.extend(auxActCopyPila)
    pilaO.extend(matchesObs)
    pilaA.extend(matchesAct)
    for i in range(int(numberCiclos)):  
        print("cicle: ", i)
        while len(pilaO) != 0:
            lotex, lotey = getBatch(10)  # mejor
            if len(list(lotex)) != 0:
                sess.run(optimizer, feed_dict={
                            a_0: lotex, y: lotey})
        for j in range(int(numGames)):  # 50
            fullReward = 0
            done = False
            observation = env.reset()
            listAuxObs, listAuxAct = [], []
            while not done:
                # Rd
                observation = np.delete(observation , vecIndexFilter)#aumente
                auxObservation = np.divide(observation, 255.0)    #aumente            
                action = sess.run(prod, feed_dict={
                                        a_0: auxObservation.reshape(1, sizeInputX)})
                #print action
                observation, reward, done, info = env.step(action)
                fullReward += reward
                listAuxObs.append(observation)
                listAuxAct.append(action)
            if fullReward > expectedReward:
                _, auxObs = getNormalizeObservationsFilter(  # normalize OBS for each game
                    listAuxObs, vecIndexFilter)
                matchesObs.extend(auxObs)
                _, auxAct = getNormalizeActionsDefault(
                    listAuxAct, sizeNumberKeysY)  # normalize actions for each game
                matchesAct.extend(auxAct)
                expectedReward+=50###############
        if len(matchesObs) == 0:
            matchesObs.extend(auxObsCopyPila)
            matchesAct.extend(auxActCopyPila)
        pilaO.extend(matchesObs)
        pilaA.extend(matchesAct)
    #############################SAVE############################
    saver = tf.train.Saver()
    save_path = saver.save(sess, path)
    message = "Model saved in path: "+ save_path
    sys.stdout.write(message)

#list_obs_por_juego, list_action_esperadas_por_juego = [], []
#numberCiclos, numGames,  sizeInputX,  sizeNumberKeysY, observations, actions, vecIndexFilter, nameGame, expectedReward, path    
#train(3,5,128,9,list_obs_por_juego, list_action_esperadas_por_juego ,[],"MsPacman-ram-v0",250,"./save/uno/model.kpt")

def display_arr(screen, arr, transpose, video_size):
    arr_min, arr_max = arr.min(), arr.max()
    arr = 255.0 * (arr - arr_min) / (arr_max - arr_min)
    pyg_img = pygame.surfarray.make_surface(arr.swapaxes(0, 1) if transpose else arr)
    pyg_img = pygame.transform.scale(pyg_img, video_size)
    screen.blit(pyg_img, (0,0))

def play(sizeInputX,sizeNumberKeysY ,nameGame ,vecIndexFilter, path):
    ###########################RED####################################
    a_0 = tf.placeholder(tf.float32, [None, sizeInputX])
    y = tf.placeholder(tf.float32, [None, sizeNumberKeysY])
    middle = 30
    w_1 = tf.Variable(tf.truncated_normal([sizeInputX, middle]))
    b_1 = tf.Variable(tf.truncated_normal([1, middle]))
    w_2 = tf.Variable(tf.truncated_normal([middle, sizeNumberKeysY]))
    b_2 = tf.Variable(tf.truncated_normal([1, sizeNumberKeysY]))
    z_1 = tf.add(tf.matmul(a_0, w_1), b_1)
    a_1 = sigma(z_1)
    z_2 = tf.add(tf.matmul(a_1, w_2), b_2)
    a_2 = sigma(z_2)
    diff = 0.5 * (tf.subtract(a_2, y))**2
    ###########################BACK PROPAGATION###################
    cost = tf.multiply(diff, diff)
    optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(cost)
    prod = tf.argmax(a_2, 1)
    sess = tf.InteractiveSession()
    ###########################################################
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()    
    ########################PLAY#######################
    env = gym.make(nameGame)

    with tf.Session() as sess:
        saver.restore(sess, path)
        for i_episode in range(1):
            observation = env.reset()
            aux_reward = 0
            reward = 0
            done = False
            list_aux_ac = []
            #######################################3
            velocity = 50
            clock = pygame.time.Clock()
            screen = pygame.display.set_mode((480,630))
            pygame.display.set_caption(u'DEVINT-24 GAMES - NNAgent')
            ################################
            rgb_array = []
            while not done:
                #env.render()
                observation = np.divide(observation, 255.0)
                observation = np.delete(observation,vecIndexFilter)
                action = sess.run(prod, feed_dict={
                                        a_0: observation.reshape(1, sizeInputX)})
                observation, reward, done, rgb_array = env.step(action)
                aux_reward += reward
                list_aux_ac.append(action[0])
                #############################################
                if observation is not None:################
                    display_arr(screen, rgb_array, True, (480,630))
                pygame.display.flip()
                clock.tick(velocity)
            pygame.quit()
        ###################################################
            if aux_reward > 300:
                print("este juego paso o:: ", i_episode)
                print("reward:: ", aux_reward)
            sys.stdout.write(str(aux_reward))
            aux_reward = 0

#./save/uno/model.kpt
#sizeInputX,sizeNumberKeysY ,nameGame ,vecIndexFilter, path
play(128,9,"MsPacman-ram-v0",[],"./save/uno/model.kpt")
