import numpy as np
import gym


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
print len(tam_mat_act)
print len(tam_mat_obs)


_, aux_obs_copy_pila = get_normalizar_observations()
_, aux_act_copy_pila = get_normalizar_actions()


def get_lote(tam_lote):
    lis_aux_obs = []
    lis_aux_act = []
    # print("tam pila: ", len(aux_obs_copy_pila))
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



# 1 setup
import tensorflow
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


a_0 = tf.placeholder(tf.float32, [None, 128])
y = tf.placeholder(tf.float32, [None, tam_teclas_disponibles])

middle = 128
w_1 = tf.Variable(tf.truncated_normal([128, middle]))
b_1 = tf.Variable(tf.truncated_normal([1, middle]))
w_2 = tf.Variable(tf.truncated_normal([middle, tam_teclas_disponibles]))
b_2 = tf.Variable(tf.truncated_normal([1, tam_teclas_disponibles]))


def sigma(x):
    return tf.div(tf.constant(1.0),
                  tf.add(tf.constant(1.0), tf.exp(tf.negative(x))))


z_1 = tf.add(tf.matmul(a_0, w_1), b_1)
a_1 = sigma(z_1)
z_2 = tf.add(tf.matmul(a_1, w_2), b_2)
a_2 = sigma(z_2)

diff = tf.subtract(a_2, y)


def sigmaprime(x):
    return tf.multiply(sigma(x), tf.subtract(tf.constant(1.0), sigma(x)))


cost = tf.multiply(diff, diff)
step = tf.train.GradientDescentOptimizer(0.1).minimize(cost)


# salida obtenida y salida esperada
acct_mat = tf.equal(tf.argmax(a_2, 1), tf.argmax(y, 1))
acct_res = tf.reduce_sum(tf.cast(acct_mat, tf.float32))

Prod = tf.argmax(a_2, 1)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

while len(aux_act_copy_pila) != 0:
#for i in xrange(10000):
    batch_xs, batch_ys = get_lote(10)
    sess.run(step, feed_dict={a_0: batch_xs,
                              y: batch_ys})
    print("batch_xs:  ", batch_xs)
    print("batch_ys:  ", batch_ys)
    # if i % 1000 == 0:
        # res = sess.run(acct_res, feed_dict={a_0: mnist.test.images[:1000],
                                            #y: mnist.test.labels[:1000]})
        # print ('Resultado de una imagen', sess.run(
            # acct_mat, feed_dict={a_0: mnist.test.images[2].reshape(1, 784), y: mnist.test.labels[2].reshape(1, 10)}))

        # print ('Resultado de una imagen2222 ', sess.run(
            # Prod, feed_dict={a_0: mnist.test.images[6].reshape(1, 784)}))
        # print res


for i_episode in range(15):
    observation=env.reset()
    aux_reward=0
    reward=0
    done=False
    list_aux_ac=[]
    while not done:
        env.render()
        action=sess.run(Prod, feed_dict = {
                          a_0: observation.reshape(1, 128)})
        #print("action elegida: ", action)
        observation, reward, done, info = env.step(action)
        aux_reward += reward
        list_aux_ac.append(action[0])

    if aux_reward > 250.0:
        print("este juego paso: ", i_episode)
    aux_reward = 0
    # print("action elegida list: ", list_aux_ac)
