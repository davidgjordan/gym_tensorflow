import tensorflow as tf
import numpy as np
import gym

_inputX = 128
_numberKeysY = 9
_nameGame = "MsPacman-ram-v0"


def configNetwork(inputX, numberKeysY, nameGame):
    _inputX = inputX
    _numberKeysY = numberKeysY
    _nameGame = nameGame


_env = gym.make(_nameGame)

# # # # # # # # # # # # # # # # # # # # # # # # # # #
xInputObservations = tf.placeholder(tf.float32, [None, _inputX])
yExpectedActions = tf.placeholder(tf.float32, [None, _numberKeysY])
weigth = tf.Variable(tf.zeros([_inputX, _numberKeysY]))
bias = tf.Variable(tf.zeros([_numberKeysY]))  # Vector con bias
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
saver = tf.train.Saver()

sess = tf.Session()
sess.run(init)
# # # # # # # # # # #  # # # # # # # # # # # # # # # #


# def getNormalizeActions(listExpectedActionForGame, numberKeysY):
#     vecAuxAct = []
#     for vecActionForGame in listExpectedActionForGame:
#         for act in vecActionForGame:
#             aux = np.zeros((numberKeysY))
#             aux[act] = 1
#             vecAuxAct.append(aux)
#     matNormalizeActions = np.vstack(vecAuxAct)
#     return matNormalizeActions, vecAuxAct


def getNormalizeActionsDefault(listExpectedActionForGame, numberKeysY):
    vecAuxAct = []
    for act in listExpectedActionForGame:
        aux = np.zeros((numberKeysY))
        aux[act] = 1
        vecAuxAct.append(aux)
    matNormalizeActions = np.vstack(vecAuxAct)
    return matNormalizeActions, vecAuxAct


# def getNormalizeObservations(listExpectedObsForGame):
#     vecAuxObs = []
#     for vecObsForGame in listExpectedObsForGame:
#         for obs in vecObsForGame:
#             vecAuxObs.append(obs)
#     matNormalizeObs = np.vstack(vecAuxObs)
#     return matNormalizeObs, vecAuxObs


def getNormalizeObservationsDefault(listExpectedObsForGame):
    vecAuxObs = []
    for obs in listExpectedObsForGame:
        obs = np.divide(obs, 256.0)
        vecAuxObs.append(obs)
    matNormalizeObs = np.vstack(vecAuxObs)
    return matNormalizeObs, vecAuxObs


def getNormalizeObservationsFilter(listExpectedObsForGame, vecIndexFilter):
    vecAuxObs = []
    for obs in listExpectedObsForGame:

        obs = np.divide(obs, 256.0)
        obs = np.delete(obs, vecIndexFilter)
        vecAuxObs.append(obs)

    matNormalizeObs = np.vstack(vecAuxObs)
    return matNormalizeObs, vecAuxObs


pilaO, pilaA = [], []


def getBatch(sizeBatch):
    lisAuxObs = []
    lisAuxAct = []
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


def train(numberCiclos, numGames,  sizeInputX,  sizeNumberKeysY, observations, actions, vecIndexFilter, nameGame, expectedReward, path):

    sess.run(init)
    tf.global_variables_initializer()

    configNetwork(sizeInputX, sizeNumberKeysY, nameGame)
    # tambien deveria normalizar con las nuevas columnas del vector de indices
    _, auxObsCopyPila = getNormalizeObservationsDefault(observations)
    #auxObsCopyPila = observations
    _, auxActCopyPila = getNormalizeActionsDefault(actions, sizeNumberKeysY)
    matchesObs = []
    matchesAct = []
    matchesObs.extend(auxObsCopyPila)
    matchesAct.extend(auxActCopyPila)
    pilaO.extend(matchesObs)
    pilaA.extend(matchesAct)

    optimizerR()

    for i in range(numberCiclos):  # 3
        for j in range(numGames):  # 50
            fullReward = 0
            done = False
            observation = _env.reset()
            listAuxObs, listAuxAct = [], []
            while not done:
                # Rd
                action = solve(observation)
                observation, reward, done, info = _env.step(action)
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

    save(path)


def save(path):
    save_path = saver.save(sess, "./save/tmp_cuatro/model.ckpt")  # path
    print("Model saved in file: %s" % save_path)


def solve(observation):

    action = sess.run(production, feed_dict={
                      xInputObservations: observation.reshape(1, _inputX)})
    return action


def optimizerR():

    epoca_i = 0
    while len(pilaA) != 0:
        lotex, lotey = getBatch(1)  # mejor
        if len(list(lotex)) != 0:
            #print("lotex: ", lotex)
            # print("lotey: ", lotey)
            sess.run(optimizer, feed_dict={
                     xInputObservations: lotex, yExpectedActions: lotey})

# ## # # #play # # # # # # # # ## # # # #  # # # #


def play(path):
    saver.restore(sess, "./save/tmp_dos/model.ckpt")
    for i_episode in range(5):
        observation = _env.reset()
        aux_reward = 0
        reward = 0
        done = False
        list_aux_ac = []

        while not done:
            _env.render()
            action = sess.run(production, feed_dict={
                              xInputObservations: observation.reshape(1, _inputX)})
            observation, reward, done, info = _env.step(action)
            aux_reward += reward
            list_aux_ac.append(action)
        if aux_reward > 250.0:
            print('Este juego paso: {} - Reward: {}'.format(i_episode, aux_reward))
        aux_reward = 0


# # # # # # # # # # # # # # #GENERATE EXAMPLE DATA # ## ### # # # # # # # # ####


list_obs_por_juego, list_action_esperadas_por_juego = [], []

# tam_teclas_disponibles = _env.action_space.n  # el pacman es  9
#
# print tam_teclas_disponibles


def correr_episodios_gym():
    for i_episode in range(1):
        observation = _env.reset()
        aux_reward = 0
        reward = 0
        done = False
        aux_action, aux_obs = [], []
        cont = 0
        while not done:
            _env.render()
            action = _env.action_space.sample()
            observation, reward, done, info = _env.step(action)
            aux_reward += reward

            aux_action.append(action)
            aux_obs.append(observation)

        # if aux_reward > 200.0:
            # list_obs_por_juego.append(aux_obs)
            # list_action_esperadas_por_juego.append(aux_action)
            # print("este juego paso: ", i_episode)

            list_obs_por_juego.append(observation)
            list_action_esperadas_por_juego.append(action)
            # if cont == 5:
            #     break
            cont += 1
        aux_reward = 0


correr_episodios_gym()


vecIndexFilter = []
#deleteV = [0, 2]
# numberCiclos, numGames,  sizeInputX,  sizeNumberKeysY, observations, actions, vecIndexFilter, nameGame, expectedReward, path):
train(2, 3, len(list_obs_por_juego[0]), 9, list_obs_por_juego,
      list_action_esperadas_por_juego, vecIndexFilter, "MsPacman-ram-v0", 250, "aqui")
play("aqui")

#vecIndexFilter = np.divide(vecIndexFilter, 2.0)
# print ("vecIndexFilter: ", vecIndexFilter)
# vecIndexFilter = np.delete(vecIndexFilter, deleteV)
# print ("vecIndexFilter: ", vecIndexFilter)
