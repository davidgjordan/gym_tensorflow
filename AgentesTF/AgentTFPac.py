#!/usr/bin/env python
import gym
import numpy as np
import tensorflow as tf
import time


class PolicyGradientAgent():

    def __init__(self, hparams, sess):

        # inicializamos prompidad de clase seess(sension tf)

        self._s = sess

        # construir tensor de entrada tf.float32, [None, envSize], name="input_x"  ---tf.float32,  shape=[None, hparams['input_size']]
        self._input = tf.placeholder(tf.float32,
                                     shape=[None, hparams['input_size']])

        hidden1 = tf.contrib.layers.fully_connected(
            inputs=self._input,
            num_outputs=hparams['hidden_size'],
            activation_fn=tf.nn.relu)

        # crea una variable llamada weights, lo que representa una matriz de peso completamente
        # conectado, que se multiplica
        # por el inputspara producir una tensor de las unidades ocultas
        weights = tf.contrib.layers.fully_connected(
            inputs=hidden1,
            num_outputs=hparams['num_actions'],
            activation_fn=None)

        # para generar una accion
        self._reshapeModel = tf.reshape(tf.multinomial(weights, 1), [])

        # agregue un numero pequeo para evitar enviar cero al registro
        log_prob = tf.log(tf.nn.softmax(weights) + 1e-8)

        # auxiliar para las actiones
        self._acts = tf.placeholder(tf.int32)  # tf.float32, [None, 1]
        # auxiliar para las ventajas
        self._advantages = tf.placeholder(tf.float32)

        # creamos una matroz de acciones probables segun este indicie de acciones aleatores
        #  para iniciar el step
        indices = tf.range(0, tf.shape(log_prob)[
                           0]) * tf.shape(log_prob)[1] + self._acts
        act_prob = tf.gather(tf.reshape(log_prob, [-1]), indices)

        # auxiliar para guardar las perdidas calculadas en vase a las ventajas
        # calcula la suma de elementos a travs de las dimensiones de un tensor
        loss = -tf.reduce_sum(tf.multiply(act_prob, self._advantages))
        self._debug = loss

        # update GradientDescentOptimizer        optimizer = tf.train.RMSPropOptimizer(hparams['learning_rate'])
        optimizer = tf.train.RMSPropOptimizer(hparams['learning_rate'])
        init = tf.initialize_all_variables()
        # self._s.run(init)
        self._train = optimizer.minimize(loss)

    # metodo para obetener acciones segun las osevaciones el remodelado y las nuevas entradas
    def get_action(self, observation):
        # obtenemos una accin, por muestreo  HACER Q ESTE METODO DEVUELVA UN NUMBER DEL 0 A 4

        action = self._s.run(self._reshapeModel, feed_dict={
                          self._input: [observation]})
        return action

    # metodo para en entrenamiento por paso
    def train_step(self, obs, acts, advantages):
        batch_feed = {self._input: obs,  # alimentacion por lotes
                      self._acts: acts,
                      self._advantages: advantages}

        # feed_dictpara proporcionar los ejemplos de entrada para este paso de entrenamiento
        self._s.run(self._train, feed_dict=batch_feed)


def start_training_policy(env, agent):#AUMENTAR UN PARAMETRO DE LAS OBSERVACIONES , REWARD Y DONE Q ME MANDEN PARA ENTRENAR CON ESO
    # Se ejecutara solo un episodio (asta q muera el pacman)
    #   AQUI DEVERIA MANDARME LAS OBSERVATIONS PARA EL ENTRENAMIENTO
    observation, reward, done = env.reset(), 0, False#
    obs, acts, rews = [], [], []

    while not done:

        env.render()

        #obs.append(observation)
        action = agent.get_action(observation)

        observation, reward, done, _ = env.step(
            action)  # action   _actions[action]
        #print ('Ations: ', action)
        #print ('observations: ', observation)
        obs.append(observation)#verificar si es mas conveniente antes del step o no
        acts.append(action)
        rews.append(reward)

    return obs, acts, rews


def process_rewards(rews):
    # rewards -> recompensas -> ventajas para un episodio

    # recompensa total: duracion del episodio
    return [len(rews)] * len(rews)


def main():

    env = gym.make('DemonAttack-ram-v0')

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
        'hidden_size': 100,  # 36  capaz ocultas
        # env.action_space.n  segun en espacio de juego
        'num_actions': env.action_space.n,
        'learning_rate': 0.09  # tasa de aprendizaje 0.05 a mayor tasa
        # de aprendizaje mayor es la modificacion de los
        # pesos que realiza tensorflow en cada iteracion
    }

    # parameters para el entrenamiento por lotes
    eparams = {
        "num_batches": 500,
        'ep_per_batch': 100
    }

    # sseion en tf
    count = 0
    with tf.Session() as sess:  # tf.Session() as sess:  tf.Graph().as_default(), tf.Session() as sess

        # objeto para el entrenamiento
        agent = PolicyGradientAgent(hparams, sess)
        #
        sess.run(tf.initialize_all_variables())

        # for batch in xrange(max_episodes):

        for batch in xrange(eparams['num_batches']):#ITERO EL TAMANHO DE OBS Q ME DEN
            time.sleep(1)

            print '=====\nBATCHE {}\n===='.format(batch)

            aux_obs, aux_acts, aux_rews = [], [], []

            # for _ in xrange(max_steps):
            for _ in xrange(eparams['ep_per_batch']):
                obs, acts, rews = start_training_policy(env, agent)

                #print ('Ations: ', acts)
                #print ('Rewards: ', rews)
                # print 'Episode steps: {}'.format(len(obs))

                # anhadimos la lista de obs a nuestro auxiliar(extend concatena a continuaion de los datos existentes)
                aux_obs.extend(obs)
                aux_acts.extend(acts)

                # procesamos los rewards y devolvemos una matriz con el promedio score
                advantages = process_rewards(rews)
                aux_rews.extend(advantages)
                # print 'Episodio {} ==== Rewards1 {} 'count, advantages[0]

                print(
                    'Episodio: {} ====  Prom Rewards * tiempo {}'.format(count, advantages[0]))

                #print ('Rewards1: ', advantages[0])
                count = count + 1
                #print ('Rewards2: ', aux_rews - np.mean(aux_rews))

            # np.MEAN devuelve el promedio de los elementos de la matriz
            aux_rews = aux_rews - np.mean(aux_rews)

            # Calcular la desviacion estandar a lo largo del eje especificado
            # Devuelve la desviacion estandar, una medida de la propagacion de una
            # distribucion, de los elementos de la matri
            std = np.std(aux_rews)
            # actualizo la politica de entrenamiento
            # y normalizo los rewards;
            if std != 0.0:  # para q no de error en caso de ser cero
                aux_rews = aux_rews / std  # seteamos

            # mandar nuevas observaciones procesadas para su refactorizacion por tf
            agent.train_step(aux_obs, aux_acts, aux_rews)
            print("aux_obs", aux_obs)
        # cerramos entorno
        env.monitor.close()


# if __name__ == "__main__":
    # main()
main()
