#!/usr/bin/env python
import sys
sys.path.insert(0,'/home/ubuntu/Desktop')
import gym
import tensorflow as tf
import numpy as np
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
    i_episode = 0
    while i_episode != 3:
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
            rgb_array = env.render(mode='rgb_array')#env.render()            
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            aux_reward += reward

            aux_action.append(action)
            aux_obs.append(observation)
            #############################################
            if observation is not None:################
                display_arr(screen, rgb_array, True, (480,630))
            pygame.display.flip()
            clock.tick(velocity)
        ###################################################

        if aux_reward > 250:
            i_episode = i_episode+ 1
            list_obs_por_juego.append(aux_obs)
            list_action_esperadas_por_juego.append(aux_action)
            print("este juego paso: ", i_episode)
            print("rear: ", aux_reward)
        aux_reward = 0
    pygame.quit()


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



# mat_obs, mat_ac = get_lote(5)

# print("Actions: ")
# print(mat_ac)
# print("Observations: ")
# print(mat_obs)
# #  #  # # # # # # # # # FIN GYM # # # # # # # # # # # # # # # # # # # # # # # # # #

_sizeInputX = 128
_sizeNumberKeysY = 9
def train():
    global _sizeInputX
    _sizeInputX = 128
    global _sizeNumberKeysY
    _sizeNumberKeysY = 9
train()
print("inputx: ",_sizeInputX)
print("inputy: ",_sizeNumberKeysY)
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


epoca_i = 0


while len(aux_obs_copy_pila) != 0:
    lotex, lotey = get_lote(1)  # mejor
    if len(list(lotex)) != 0:
        lotex = np.divide(lotex, 255.0)
        lotey = np.divide(lotey, 255.0)
        sess.run(optimizer, feed_dict={
                    a_0: lotex, y: lotey})
    epoca_i+=1


def play():
    for i_episode in range(5):
        observation = env.reset()
        aux_reward = 0
        reward = 0
        done = False
        list_aux_ac = []
        ########################################
        velocity = 50
        clock = pygame.time.Clock()
        screen = pygame.display.set_mode((480,630))
        pygame.display.set_caption(u'DEVINT-24 GAMES - NNAgent')
        #########################################
        while not done:
            #env.render()
            rgb_array = env.render(mode='rgb_array')
            observation = np.divide(observation, 255.0)
            
            action = sess.run(prod, feed_dict={
                                    a_0: observation.reshape(1, 128)})
            #print("action elegida: ", action)
            
            observation, reward, done, info = env.step(action)
            aux_reward += reward
            list_aux_ac.append(action[0])
            #############################################
            if observation is not None:################
                display_arr(screen, rgb_array, True, (480,630))
            pygame.display.flip()
            clock.tick(velocity)
        ###################################################
        if aux_reward > 300:
            print("este juego paso o:: ", i_episode)
            print("reward:: ", aux_reward)
        aux_reward = 0
    pygame.quit()
    
play()
saver = tf.train.Saver()
save_path = saver.save(sess, "./tmp_seis_f/model.ckpt")
print("Model saved in path: %s" % save_path)

# for i in xrange(10000):
#     batch_xs, batch_ys = mnist.train.next_batch(3)
#     sess.run(step, feed_dict={a_0: batch_xs,
#                               y: batch_ys})
#     print "##########################xx########X#######################"
#     print batch_xs
#     print "############################YYY#############################"
#     print batch_ys
#     print "#########################################################"

#     if i % 1000 == 0:
#         res = sess.run(acct_res, feed_dict={a_0: mnist.test.images[:1000],
#                                            y: mnist.test.labels[:1000]})
        # print mnist.test.labels[2]
        # print(sess.run(prod, feed_dict={
        #      a_0: mnist.test.images[2].reshape(1, 784)}))
        # print res
################################################

# lis_action, list_obs = [], []


# def correr_episodios_gym():
#     for i_episode in range(2):
#         observation = env.reset()
#         aux_reward = 0
#         reward = 0
#         done = False
#         aux_action, aux_obs = [], []
#         while not done:
#             env.render()
#             action = env.action_space.sample()
#             observation, reward, done, info = env.step(action)
#             aux_reward += reward

#             aux_action.append(action)
#             aux_obs.append(observation)
