import tensorflow as tf
import numpy as np
import gym
import sys


def configNetwork(inputX, numberKeysY, nameGame):
    global _inputX  
    _inputX = inputX
    global _numberKeysY 
    _numberKeysY = numberKeysY
    global _nameGame 
    _nameGame = nameGame


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

 # json {
    #    method_1: train:
    #               parameters:[
    #                   ciclos: configtTrain.ciclos,
    #                   numgames:configtTrain.num_games
    #                   input_obs_a_0: observations[0].size()
    #                   input_salida_y: configTrain.num_teclas
    #                   obser: game.observations
    #                   actions: game.actions
    #                   vector_indices: configTrain.vecIndices
    #               ]
    # }

sess = None
production = None
optimizer = None
env  = None
xInputObservations = None
yExpectedActions = None
saver = None
weigth =None
bias =None
yOperForNodes =None
funSoftmaxError =None
cost =None
optimizer =None
prediction =None
accuracy =None
production =None
init =None

def train(numberCiclos, numGames,  sizeInputX,  sizeNumberKeysY, observations, actions, vecIndexFilter, nameGame, expectedReward, path):

    global sess
    global production
    global optimizer
    global env 
    global xInputObservations
    global yExpectedActions
    global saver
    global pilaO
    global pilaA
    global weigth    
    global bias
    global yOperForNodes
    global funSoftmaxError
    global cost
    global optimizer
    global prediction    
    global accuracy
    global production
    global init
    global tf
    # # # # # # # # # # # # # # # # # # # # # # # # # # #
    env = gym.make(nameGame)
    xInputObservations = tf.placeholder(tf.float32, [None, sizeInputX])
    yExpectedActions = tf.placeholder(tf.float32, [None, sizeNumberKeysY])
    weigth = tf.Variable(tf.zeros([sizeInputX, sizeNumberKeysY]))
    bias = tf.Variable(tf.zeros([sizeNumberKeysY]))  # Vector con bias
    yOperForNodes = tf.matmul(xInputObservations, weigth) + bias
    funSoftmaxError = tf.nn.softmax_cross_entropy_with_logits(
        labels=yExpectedActions, logits=yOperForNodes)
    cost = tf.reduce_mean(funSoftmaxError)
    optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(cost)
    prediction = tf.equal(tf.argmax(yOperForNodes, 1), tf.argmax(
        yExpectedActions, 1))
    accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
    production = tf.argmax(yOperForNodes, 1)
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    # # # # # # # # # # #  # # # # # # # # # # # # # # # #
    configNetwork(sizeInputX, sizeNumberKeysY, nameGame)
    _, auxObsCopyPila = getNormalizeObservationsDefault(observations)
    _, auxActCopyPila = getNormalizeActionsDefault(actions, sizeNumberKeysY)
    matchesObs = []
    matchesAct = []
    matchesObs.extend(auxObsCopyPila)
    matchesAct.extend(auxActCopyPila)
    
    pilaO.extend(matchesObs)
    pilaA.extend(matchesAct)

    optimizerR()

    for i in range(int(numberCiclos)):  # 3
        for j in range(int(numGames)):  # 50
            fullReward = 0
            done = False
            observation = env.reset()
            listAuxObs, listAuxAct = [], []
            while not done:
                # Rd
                observation = np.delete(observation , vecIndexFilter)#aumente
                action = solve(observation, sizeInputX)
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
        if len(matchesObs) == 0:
            matchesObs.extend(auxObsCopyPila)
            matchesAct.extend(auxActCopyPila)
        pilaO.extend(matchesObs)
        pilaA.extend(matchesAct)
        optimizerR()
    saver = tf.train.Saver()
    save(path)


def save(path):
    save_path = saver.save(sess, path+"model.ckpt")  # path
    message = "Model saved in file: " + save_path
    sys.stdout.write(message)
    sess.close()
    


def solve(observation,_inputX):#aumente este parametro
    action = sess.run(production, feed_dict={
                      xInputObservations: observation.reshape(1, _inputX)})
    return action

def optimizerR():
    epoca_i = 0
    while len(pilaO) != 0:
        lotex, lotey = getBatch(1)  # mejor
        if len(list(lotex)) != 0:
            sess.run(optimizer, feed_dict={
                     xInputObservations: lotex, yExpectedActions: lotey})
        epoca_i+=1
# ## # # #play # # # # # # # # ## # # # #  # # # #
