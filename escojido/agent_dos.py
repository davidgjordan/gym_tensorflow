import tensorflow as tf
import numpy as np
import gym


# # # # # # # # # # # # CLASS # # # # # # # # # # # #
class NNAgent():
    def __init__(self, env, sess, list_obs_por_juego, list_action_esperadas_por_juego):

        self._measure_of_reward = 250
        self._tam_nodos_entrada = 128
        self._sess = sess
        self._env = env  # gym.make('MsPacman-ram-v0')
        self._tam_teclas_disponibles = env.action_space.n
        self._list_obs_por_juego = list_obs_por_juego
        self._list_action_esperadas_por_juego = list_action_esperadas_por_juego

        # #  # # # # # # # ## ##  RED NEURONAL  # # # # # # #
        self._x_input_observations = tf.placeholder(
            tf.float32, [None, self._tam_nodos_entrada])
        self._yR_salidas_aciones_esperadas = tf.placeholder(
            tf.float32, [None, self._tam_teclas_disponibles])
        self._Pesos = tf.Variable(
            tf.zeros([self._tam_nodos_entrada, self._tam_teclas_disponibles]))
        self._bias = tf.Variable(tf.zeros([self._tam_teclas_disponibles]))
        self._y_oper_nodos_entradas = tf.matmul(
            self._x_input_observations, self._Pesos) + self._bias
        self._fun_softmax_medir_error = tf.nn.softmax_cross_entropy_with_logits(
            labels=self._yR_salidas_aciones_esperadas, logits=self._y_oper_nodos_entradas)
        self._costo = tf.reduce_mean(self._fun_softmax_medir_error)
        self._optimizador = tf.train.GradientDescentOptimizer(
            0.1).minimize(self._costo)
        self._prediccion = tf.equal(tf.argmax(self._y_oper_nodos_entradas, 1), tf.argmax(
            self._yR_salidas_aciones_esperadas, 1))
        self._exactitud_predicciones = tf.reduce_mean(
            tf.cast(self._prediccion, tf.float32))
        self._Produccion = tf.argmax(self._y_oper_nodos_entradas, 1)
        self._init = tf.global_variables_initializer()

        _, self.aux_obs_copy_pila = get_normalizar_observations()
        _, self.aux_act_copy_pila = get_normalizar_actions()
        self._saver = tf.train.Saver()

    def get_normalizar_actions():
        vec_aux_ac = []
        for vec_action_por_juego in self._list_action_esperadas_por_juego:
            for act in vec_action_por_juego:
                aux = np.zeros((self._tam_teclas_disponibles))
                aux[act] = 1
                vec_aux_ac.append(aux)
        mat_normalize_actions = np.vstack(vec_aux_ac)
        return mat_normalize_actions, vec_aux_ac

    def get_normalizar_observations():
        vec_aux_obs = []
        for vec_obs_por_juego in self._list_obs_por_juego:
            for obs in vec_obs_por_juego:
                vec_aux_obs.append(obs)
        mat_normalize_obs = np.vstack(vec_aux_obs)
        return mat_normalize_obs, vec_aux_obs

    def get_lote(tam_lote):
        lis_aux_obs = []
        lis_aux_act = []
        for i in range(tam_lote):
            if self.aux_obs_copy_pila:
                data_o = self.aux_obs_copy_pila.pop()
                data_a = self.aux_act_copy_pila.pop()

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

    def avance(epoca_i, _sess, last_features, last_labels):
        costoActual = self._sess.run(costo, feed_dict={
            self._x_input_observations: last_features, _yR_salidas_aciones_esperadas: last_labels})
        mat_normalize_obs, _ = get_normalizar_observations()
        mat_normalize_actions, _ = get_normalizar_actions()
        Certeza = self._sess.run(_exactitud_predicciones, feed_dict={
            self._x_input_observations: mat_normalize_obs, self._yR_salidas_aciones_esperadas: mat_normalize_actions})
        print(
            'Epoca: {:<4} - Costo: {:<8.3} Certeza: {:<5.3}'.format(epoca_i, costoActual, Certeza))

    def train():
        with tf.Session() as self._sess:
            self._sess.run(_init)
            tf.global_variables_initializer()
            epoca_i = 0
            while len(aux_act_copy_pila) != 0:
                lotex, lotey = get_lote(3)
                opt = self._sess.run(_optimizador, feed_dict={
                                     self._x_input_observations: lotex, self._yR_salidas_aciones_esperadas: lotey})
                if (epoca_i % 50 == 0):
                    avance(epoca_i, _sess, lotex, lotey)

                    lis_obs, ___ = get_normalizar_observations()
                    mat_ob = np.vstack(lis_obs[5])
                epoca_i += 1

    def probarTrain():
        for i_episode in range(10):
            observation = self._env.reset()
            aux_reward = 0
            reward = 0
            done = False
            list_aux_ac = []

            while not done:
                env.render()
                action = self._sess.run(self._Produccion, feed_dict={
                                        self._x_input_observations: observation.reshape(1, _tam_nodos_entrada)})
                #print("action elegida: ", action)
                observation, reward, done, info = env.step(action)
                aux_reward += reward
                list_aux_ac.append(action[0])
            if aux_reward > _measure_of_reward:
                print("este juego paso o:: ", i_episode)
            aux_reward = 0
        #print("action elegida list: ", list_aux_ac)

    def saveTrain(save_path):
        # SALVAR
        # "./tmp_tres/model.ckpt")
        path = self._saver.save(self._sess, save_path)
        print("Model saved in file: %s" % path)

# # # # # # # # # # # # FIN CLASS # # # # # # # # # # # #

# #  #  # # # # # # # # # GYM # # # # # # # # # # # # # # # # # # # # # # # # # #


list_obs_por_juego, list_action_esperadas_por_juego = [], []

env = gym.make('MsPacman-ram-v0')
tam_teclas_disponibles = env.action_space.n  # el pacman es  9

print tam_teclas_disponibles


def correr_episodios_gym():
    for i_episode in range(10):
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

            aux_action.append(action)
            aux_obs.append(observation)

        if aux_reward > 250.0:
            list_obs_por_juego.append(aux_obs)
            list_action_esperadas_por_juego.append(aux_action)
            print("este juego paso: ", i_episode)
        aux_reward = 0


correr_episodios_gym()
sess = tf.Session()
nnagent = NNAgent(env, sess, list_obs_por_juego,
                  list_action_esperadas_por_juego)
