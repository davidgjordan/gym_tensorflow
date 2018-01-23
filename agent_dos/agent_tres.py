import numpy as np
import gym
# La imagenes tienen dimension de 28x28

# #  #  # # # # # # # # # GYM # # # # # # # # # # # # # # # # # # # # # # # # # #

list_obs_por_juego, list_action_esperadas_por_juego = [], []

env = gym.make('MsPacman-ram-v0')
tam_teclas_disponibles = env.action_space.n  # el pacman es  9
print tam_teclas_disponibles


def correr_episodios_gym():
    for i_episode in range(20):
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



# 1 setup
import tensorflow
import tensorflow as tf
#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

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

# 1.1.The sigmoid function


def sigma(x):
    return tf.div(tf.constant(1.0),
                  tf.add(tf.constant(1.0), tf.exp(tf.negative(x))))


# 1.2.The forward propagation
z_1 = tf.add(tf.matmul(a_0, w_1), b_1)
a_1 = sigma(z_1)
z_2 = tf.add(tf.matmul(a_1, w_2), b_2)
a_2 = sigma(z_2)


# 1.3.Difference
diff = tf.subtract(a_2, y)


# 1.4.The sigmoid prime function
# def sigmaprime(x):
#    return tf.multiply(sigma(x), tf.subtract(tf.constant(1.0), sigma(x)))


############## REMPLAZE FOR EL BACK PROPAGATION ###########

# 1.5.Backward propagation
# d_z_2 = tf.multiply(diff, sigmaprime(z_2))
# d_b_2 = d_z_2
# d_w_2 = tf.matmul(tf.transpose(a_1), d_z_2)

# d_a_1 = tf.matmul(d_z_2, tf.transpose(w_2))
# d_z_1 = tf.multiply(d_a_1, sigmaprime(z_1))
# d_b_1 = d_z_1
# d_w_1 = tf.matmul(tf.transpose(x_input_observations), d_z_1)


# # 1.6.Updating the network
# eta = tf.constant(0.5)
#     optimizador = [
#         tf.assign(w_1,
#                   tf.subtract(w_1, tf.multiply(eta, d_w_1))), tf.assign(b_1,
#                                                                         tf.subtract(b_1, tf.multiply(eta,
#                                                                                                      tf.reduce_mean(d_b_1, axis=[0])))), tf.assign(w_2,
#                                                                                                                                                    tf.subtract(w_2, tf.multiply(eta, d_w_2))), tf.assign(b_2,
#                                                                                                                                                                                                          tf.subtract(b_2, tf.multiply(eta,
#                                                                                                                                                                                                                                       tf.reduce_mean(d_b_2, axis=[0]))))
#
#     ]
cost = tf.multiply(diff, diff)
step = tf.train.GradientDescentOptimizer(0.1).minimize(cost)


# costo = tf.reduce_mean(fun_softmax_medir_error)
# step = tf.train.GradientDescentOptimizer(0.1).minimize(costo)

# 1.7.Running and testing the training process
acct_mat = tf.equal(tf.argmax(a_2, 1), tf.argmax(y, 1))
acct_res = tf.reduce_sum(tf.cast(acct_mat, tf.float32))

Prod = tf.argmax(a_2, 1)
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

i = 0
# for i in xrange(10000):
while len(aux_act_copy_pila) != 0:
    #batch_xs, batch_ys = mnist.train.next_batch(10)
    batch_xs, batch_ys = get_lote(10)
    opt = sess.run(step, feed_dict={a_0: batch_xs,
                                    y: batch_ys})
    print("OPT: ", opt)
    if i % 1000 == 0:
        obs, _o = get_normalizar_observations()
        act, _a = get_normalizar_actions()

        res_a = sess.run(acct_res, feed_dict={a_0: obs, y: act})

        # for data in res_a:
        print res_a
        #    pass

        #print ("batch x : ", batch_ys)
    i += 1

for i_episode in range(15):
    observation = env.reset()
    aux_reward = 0
    reward = 0
    done = False
    list_aux_ac = []
    while not done:
        env.render()
        action = sess.run(Prod, feed_dict={
                          a_0: observation.reshape(1, 128)})
        #print("action elegida: ", action)
        observation, reward, done, info = env.step(action)
        aux_reward += reward
        list_aux_ac.append(action[0])

    if aux_reward > 250.0:
        print("este juego paso: ", i_episode)
    aux_reward = 0
    print("action elegida list: ", list_aux_ac)
