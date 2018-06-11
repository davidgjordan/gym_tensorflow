import random
import sys
sys.path.insert(0, '/home/ubuntu/Desktop')
import gym
import tensorflow as tf
import numpy as np
import pygame


list_obs_por_juego, list_action_esperadas_por_juego = [], []

env = gym.make('MsPacman-ram-v0')
tam_teclas_disponibles = env.action_space.n  # el pacman es  9
# print tam_teclas_disponibles

# method for render image


def display_arr(screen, arr, transpose, video_size):
    arr_min, arr_max = arr.min(), arr.max()
    arr = 255.0 * (arr - arr_min) / (arr_max - arr_min)
    pyg_img = pygame.surfarray.make_surface(
        arr.swapaxes(0, 1) if transpose else arr)
    pyg_img = pygame.transform.scale(pyg_img, video_size)
    screen.blit(pyg_img, (0, 0))
###


print("RECOLECCION DE DATOS PARA EL ENTRENAMIENTO")
espectedReward = 150
successGamesCount = 3


def correr_episodios_gym():
    successGame = 1
    gameNumber = 1
    while successGame <= successGamesCount:
        observation = env.reset()
        aux_reward = 0
        reward = 0
        done = False
        aux_action, aux_obs = [], []
        ##################render images######################
        #velocity = 15000
        #clock = pygame.time.Clock()
        #screen = pygame.display.set_mode((480,630))
        #pygame.display.set_caption(u'OBTENIENDO DATA DE ENTRENAMIENTO')
        #########################################

        while not done:
            # rgb_array = env.render(mode='rgb_array')#env.render()
            #action = env.action_space.sample()
            action = random.choice([5, 6, 7, 8])
            observation, reward, done, info = env.step(
                action)  # random.randint(0, 5)
            aux_reward += reward
            aux_action.append(action)
            aux_obs.append(observation)
            ####################render images##################
            # if observation is not None:
            #    display_arr(screen, rgb_array, True, (480,630))
            # pygame.display.flip()
            # clock.tick(velocity)
            ###################################################

        if aux_reward > espectedReward:
            print 'Juego Numero: {0} PASO - recompensa: {1} - juego: {2}/{3}'.format(gameNumber, aux_reward, successGame, gameNumber)
            successGame = successGame + 1
            global list_obs_por_juego
            global list_action_esperadas_por_juego
            if len(list_obs_por_juego) == 0:
                list_obs_por_juego = aux_obs
                list_action_esperadas_por_juego = aux_action
            if len(list_obs_por_juego) > 0:
                list_obs_por_juego = list_obs_por_juego + aux_obs
                list_action_esperadas_por_juego = list_action_esperadas_por_juego + aux_action
        else:
            print 'Juego Numero: {0} NO PASO'.format(gameNumber)
        gameNumber = gameNumber + 1
        aux_reward = 0

    # pygame.quit()


correr_episodios_gym()  # DATA


#####NORMALIZE ACTIONS#############
count = 0
aux_actions = []
for data in list_action_esperadas_por_juego:
    aux = np.zeros((9))
    aux[list_action_esperadas_por_juego[count]] = 1

    aux_actions.append(aux)
    count = count + 1


list_obs_por_juego = np.divide(list_obs_por_juego, 255.0)


########### NEURONAL NETWORK###########
a_0 = tf.placeholder(tf.float32, [None, 128])
y = tf.placeholder(tf.float32, [None, 9])

middle = 30
w_1 = tf.Variable(tf.truncated_normal([128, middle]))
b_1 = tf.Variable(tf.truncated_normal([1, middle]))
w_2 = tf.Variable(tf.truncated_normal([middle, 9]))
b_2 = tf.Variable(tf.truncated_normal([1, 9]))


def sigma(x):  # FUNTION ACTIVATION
    return tf.div(tf.constant(1.0),
                  tf.add(tf.constant(1.0), tf.exp(tf.negative(x))))


z_1 = tf.add(tf.matmul(a_0, w_1), b_1)
a_1 = sigma(z_1)
z_2 = tf.add(tf.matmul(a_1, w_2), b_2)
a_2 = sigma(z_2)
diff = tf.subtract(a_2, y)
###########################
cost = tf.multiply(diff, diff)
optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(cost)
###########################

prod = tf.argmax(a_2, 1)
######################


#####RUNING  TRAINING#############
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

print("ENTRENANDO LA RED")
lote = 1
while lote <= len(list_obs_por_juego):
    # print("******************************************************")
    #print(aux_actions[lote - 1:lote])
    # print("******************************************************")

    sess.run(optimizer, feed_dict={
        a_0: list_obs_por_juego[lote - 1:lote], y: aux_actions[lote - 1:lote]})
    lote = lote + 1


########PLAY GAME AFTER TRAIN##################
testGames = 3
print("PROBANDO LA RED ENTRENADA")
def play():
    gameNumber = 1
    successGame = 1
    for i_episode in range(testGames):
        observation = env.reset()
        aux_reward = 0
        reward = 0
        done = False
        list_aux_ac = []
        ########################################
        velocity = 50
        clock = pygame.time.Clock()
        screen = pygame.display.set_mode((480, 630))
        pygame.display.set_caption(
            u'JUEGOS DE PRUEBA DESPUES DEL ENTRENAMIENTO')
        #########################################
        while not done:
            # env.render()
            rgb_array = env.render(mode='rgb_array')
            observation = np.divide(observation, 255.0)

            action = sess.run(prod, feed_dict={
                a_0: observation.reshape(1, 128)})
            print("action elegida: ", action)

            observation, reward, done, info = env.step(action)
            aux_reward += reward
            list_aux_ac.append(action[0])
            #############################################
            if observation is not None:
                display_arr(screen, rgb_array, True, (480, 630))
            pygame.display.flip()
            clock.tick(velocity)
        ###################################################
        if aux_reward > espectedReward:
            print 'Juego Numero: {0} PASO - recompensa: {1} - juego: {2}/{3}'.format(gameNumber, aux_reward, successGame, gameNumber)
            successGame = successGame + 1
        else:
            print 'Juego Numero: {0} NO PASO'.format(gameNumber)
        gameNumber = gameNumber + 1
        aux_reward = 0

    pygame.quit()


play()
saver = tf.train.Saver()
save_path = saver.save(sess, "./final_uno/model.ckpt")
print("Model saved in path: %s" % save_path)
