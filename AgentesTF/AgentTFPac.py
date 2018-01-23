#!/usr/bin/env python

import numpy as np
import gym
# La imagenes tienen dimension de 28x28


# #  #  # # # # # # # # # GYM # # # # # # # # # # # # # # # # # # # # # # # # # #

list_obs_por_juego, list_action_esperadas_por_juego, list_reward_por_juego = [], [], []

env = gym.make('MsPacman-ram-v0')
tam_teclas_disponibles = env.action_space.n  # el pacman es  9
print tam_teclas_disponibles


def correr_episodios_gym():
    for i_episode in range(15):
        observation = env.reset()
        aux_reward = 0
        reward = 0
        done = False
        aux_action, aux_obs, aux_reward_list = [], [], []
        while not done:
            env.render()
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            aux_reward += reward

            aux_action.append(action)
            aux_obs.append(observation)
            aux_reward_list.append(reward)

        if aux_reward > 250.0:
            list_obs_por_juego.extend(aux_obs)
            list_action_esperadas_por_juego.extend(aux_action)
            list_reward_por_juego.append(aux_reward_list)
            print("este juego paso: ", i_episode)
        aux_reward = 0
    env.close()


correr_episodios_gym()

mat_normalize_obs = None
mat_normalize_actions = None


def get_normalizar_actions():
    vec_aux_ac = []
    for vec_action_por_juego in list_action_esperadas_por_juego:
        for act in vec_action_por_juego:
            aux = np.zeros((tam_teclas_disponibles))
            aux[act] = 1
            vec_aux_ac.append(aux)
    mat_normalize_actions = np.vstack(vec_aux_ac)
    return mat_normalize_actions, vec_aux_ac


# get_normalizar_actions()


def get_normalizar_observations():
    vec_aux_obs = []
    for vec_obs_por_juego in list_obs_por_juego:
        for obs in vec_obs_por_juego:
            vec_aux_obs.append(obs)
    mat_normalize_obs = np.vstack(vec_aux_obs)
    return mat_normalize_obs, vec_aux_obs


# get_normalizar_observations()


# tam_mat_act, _ = get_normalizar_actions()
# tam_mat_obs, _ = get_normalizar_observations()
# print len(tam_mat_act)
# print len(tam_mat_obs)


# _, aux_obs_copy_pila = get_normalizar_observations()
# _, aux_act_copy_pila = get_normalizar_actions()


def get_lote(tam_lote):
    lis_aux_obs = []
    lis_aux_act = []
    #print("tam pila: ", len(aux_obs_copy_pila))
    for i in range(tam_lote):
        if aux_obs_copy_pila:
            data_o = aux_obs_copy_pila.pop()
            data_a = aux_act_copy_pila.pop()
            lis_aux_obs.append(data_o)
            lis_aux_act.append(data_a)
        else:
            break
    mat_normalize_obs = None
    mat_normalize_actions = None
    if lis_aux_act:
        mat_normalize_obs = np.vstack(lis_aux_obs)
        mat_normalize_actions = np.vstack(lis_aux_act)
    return mat_normalize_obs, mat_normalize_actions


#mat_obs, mat_ac = get_lote(3)
#
#print("Actions: ")
# print(mat_ac)
#print("Observations: ")
# print(mat_obs)
#
#
#mat_obs, mat_ac = get_lote(5)
#
#print("Actions: ")
# print(mat_ac)
#print("Observations: ")
# print(mat_obs)
# #  #  # # # # # # # # # FIN GYM # # # # # # # # # # # # # # # # # # # # # # # # # #


import tensorflow as tf
import time


class PolicyGradientAgent(object):

    def __init__(self, hparams, sess):

        # inicializamos prompidad seess(sension tf)

        self._s = sess

        # construir tensor de entrada tf.float32, [None, envSize], name="input_x"  ---tf.float32,  shape=[None, hparams['input_size']]
        self._input = tf.placeholder(tf.float32,
                                     shape=[None, hparams['input_size']])  # 128

        hidden1 = tf.contrib.layers.fully_connected(
            inputs=self._input,
            # 100  salidas de la capa oculta
            num_outputs=hparams['hidden_size'],
            activation_fn=tf.nn.relu)

        # crea una variable llamada weights, lo que representa una matriz de peso completamente
        # conectado, que se multiplica
        # por el inputspara producir una tensor de las unidades ocultas   logits
        weights = tf.contrib.layers.fully_connected(  # se vincula directamente con la anterior capa hidden1   # logits
            inputs=hidden1,
            num_outputs=hparams['num_actions'],   # creo por defecto 4
            activation_fn=None)

        # para generar una accion   op to sample an action  xample
        self._reshapeModel = tf.reshape(
            tf.multinomial(weights, 1), [])  # remodela el tensor
        # de mmuestras obtenidas q devuelve la mulotinomail segun los pesos  al tamano de una lista

# get log probabilities
        # calcula el logaritmo natural de los pesos,   agregue un numero pequeo para evitar enviar cero al registro
        log_prob = tf.log(tf.nn.softmax(weights) + 1e-8)  # logits w

        # auxiliar para las actiones
        # tf.float32, [None, 1]  es un tensor de entradas
        self._acts = tf.placeholder(tf.int32)
        # auxiliar para las ventajas
        self._advantages = tf.placeholder(
            tf.float32)  # auxiliar para las ventajas

        # creamos una matriz de acciones probables segun este indicie de acciones aleatores
        #  para iniciar el step
        indices = tf.range(0, tf.shape(log_prob)[  # desde cero hasta
                           0]) * tf.shape(log_prob)[1] + self._acts

        act_prob = tf.gather(tf.reshape(log_prob, [-1]), indices)

        # auxiliar para guardar las perdidas calculadas en vase a las ventajas
        # calcula la suma de elementos a travs de las dimensiones de un tensor
        loss = -tf.reduce_sum(tf.multiply(act_prob, self._advantages))
        self._debug = loss

        # update GradientDescentOptimizer        optimizer = tf.train.RMSPropOptimizer(hparams['learning_rate'])
        optimizer = tf.train.GradientDescentOptimizer(hparams['learning_rate'])
        #init = tf.initialize_all_variables()
        # self._s.run(init)
        self._train = optimizer.minimize(loss)

    # metodo para obetener acciones segun las osevaciones el remodelado y las nuevas entradas
    def act(self, observation):
        # obtenemos una accin, por muestreo  HACER Q ESTE METODO DEVUELVA UN NUMBER DEL 0 A 4

        act = self._s.run(self._reshapeModel, feed_dict={
                          self._input: [observation]})
        #print("act ", act)
        return act

    # metodo para en entrenamiento por paso
    def train_step(self, obs, acts, advantages):
        batch_feed = {self._input: obs,  # alimentacion por lotes
                      self._acts: acts,
                      self._advantages: advantages}

        # feed_dictpara proporcionar los ejemplos de entrada para este paso de entrenamiento
        self._s.run(self._train, feed_dict=batch_feed)


def start_training_policy(env, agent):
    # Se ejecutara solo un episodio (asta q muera el pacman)

    observation, reward, done = env.reset(), 0, False
    obs, acts, rews = [], [], []

    while not done:

        env.render()
        obs.append(observation)

        action = agent.act(observation)

        observation, reward, done, _ = env.step(
            action)  # action   _actions[action]
        #print ('Ations: ', action)
        #print ('observations: ', observation)
        #print ('reward: ', reward)

        acts.append(action)
        rews.append(reward)

    return obs, acts, rews


def process_rewards(rews):
    # rewards -> recompensas -> ventajas para un episodio

    # recompensa total: duracion del episodio
    return [len(rews)] * len(rews)


def main():

    env = gym.make('MsPacman-ram-v0')

    # monitor_dir = '/tmp/MsPacman_exp1'  # guadar registros de juego q podrian sevir
    # env.monitor.start(monitor_dir, force=True)

    # variables auxiliares para entrenar mediante episodios actualmente se esta entrenando por
    # lotes
    max_episodes = 2000
    max_steps = 500

    # parameters para gym y tf
    hparams = {
        # env.observation_space.shape[0]
        'input_size': 128,  # tamaho de entrada
        'hidden_size': 56,  # 36  capaz ocultas
        # env.action_space.n  segun en espacio de juego
        'num_actions': env.action_space.n,
        'learning_rate': 0.5  # tasa de aprendizaje 0.05 a mayor tasa
        # de aprendizaje mayor es la modificacion de los
        # pesos que realiza tensorflow en cada iteracion
    }

    # parameters para el entrenamiento por lotes
    eparams = {
        'num_batches': 60,
        'ep_per_batch': 15
    }

    # sseion en tf
    count = 0
    with tf.Session() as sess:  # tf.Session() as sess:  tf.Graph().as_default(), tf.Session() as sess

        # objeto para el entrenamiento
        agent = PolicyGradientAgent(hparams, sess)
        #
        sess.run(tf.initialize_all_variables())

        # for batch in xrange(max_episodes):

        # while len(aux_act_copy_pila) != 0:
        # for batch in xrange(eparams['num_batches']):
        #     time.sleep(1)

        #     print '=====\nBATCHE {}\n===='.format(batch)

        #     aux_obs, aux_acts, aux_rews = [], [], []

        #     # for _ in xrange(max_steps):
        #     for _ in xrange(eparams['ep_per_batch']):
        #         obs, acts, rews = start_training_policy(env, agent)

        #         print (
        #             'OBSERVATOISOBSERVATOISOBSERVATOISOBSERVATOISOBSERVATOISOBSERVATOIS: ', obs)
        #         #print ('Rewards: ', rews)
        #         # print 'Episode steps: {}'.format(len(obs))

        #         # anhadimos la lista de obs a nuestro auxiliar(extend concatena a continuaion de los datos existentes)
        #         aux_obs.extend(obs)
        #         aux_acts.extend(acts)

        #         # procesamos los rewards y devolvemos solo un score
        #         advantages = process_rewards(rews)
        #         aux_rews.extend(advantages)
        #         # print 'Episodio {} ==== Rewards1 {} 'count, advantages[0]

        #         print(
        #             'Episodio: {} ====  Prom Rewards * tiempo {}'.format(count, advantages[0]))

        #         #print ('Rewards1: ', advantages[0])
        #         count = count + 1
        #         #print ('Rewards2: ', aux_rews - np.mean(aux_rews))

        #     # np.MEAN devuelve el promedio de los elementos de la matriz
        #     aux_rews = aux_rews - np.mean(aux_rews)

        #     # Calcular la desviacion estandar a lo largo del eje especificado
        #     # Devuelve la desviacion estandar, una medida de la propagacion de una
        #     # distribucion, de los elementos de la matri
        #     std = np.std(aux_rews)
        #     # actualizo la politica de entrenamiento
        #     # y normalizo los rewards;
        #     if std != 0.0:  # para q no de error en caso de ser cero
        #         aux_rews = aux_rews / std  # seteamos

        #     # mandar nuevas observaciones procesadas para su refactorizacion por tf
        #     agent.train_step(aux_obs, aux_acts, aux_rews)


# # # # # # # # # # # ## # # # # # # # # # # # # # # # # ## # # # # #
        for i in range(3):

            rews = []

            for game in list_reward_por_juego:
                advantages = process_rewards(game)
                rews.extend(advantages)

            rews = rews - np.mean(rews)
            std = np.std(rews)
            # actualizo la politica de entrenamiento
            # y normalizo los rewards;

            if std != 0.0:  # para q no de error en caso de ser cero
                rews = rews / std  # seteamos

            agent.train_step(list_obs_por_juego,
                             list_action_esperadas_por_juego, rews)
# # # # # # # # # # # ## # # # # # # # # # # # # # # # # ## # # # # #

        for i_episode in range(15):
            observation = env.reset()
            aux_reward = 0
            reward = 0
            done = False
            list_aux_ac = []

            while not done:
                env.render()
                action = agent.act(observation)

                observation, reward, done, _ = env.step(
                    action)
                #print("action elegida: ", action)
                aux_reward += reward
                list_aux_ac.append(action)
            if aux_reward > 250.0:
                print("este juego paso: ", i_episode)
            aux_reward = 0
            print("action elegida list: ", list_aux_ac)

        # cerramos entorno
        # env.monitor.close()



# if __name__ == "__main__":
    # main()
main()
