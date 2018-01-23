import gym

import numpy as np
import tensorflow as tf
import math


# Hyperparameters
envSize = 128
# 100 #number of neurons in hidden layer(n de neruronas en la capa oculta)
H = 100
batch_number = 50  # 50 size of batches for training (lotes para entrenamiento) de 50 a 128
learn_rate = .01  # taza de aprendizaje

# Este algoritmo se basa en los mismos principios que rigen la actualizacion de parametros
# en el algoritmo Acelerador Regresivo version Gamma. El algoritmo Acelerador Regresivo version
# Gamma con Gradiente Local de Error se valida mediante diferentes problemas relacionados con
# aproximacion de funciones y reconocimiento de patrones. Los resultados muestran buen comportamiento
# en cuanto a convergencia y generalizacion, mejorando la tasa de aprendizaje del algoritmo backpropagation
gamma = 0.99  # algoritmo acelerador regresivo version gamma con gradiente local de error para entrenamiento de 
                # redes neuronales perceptron multicapa


def reduced_rewards(r):  # recompensas reducidas
    reduced_r = np.zeros_like(r)  # setea nuestra matriz a valores cero
    running_add = 0
    for t in reversed(xrange(0, r.size)):
        running_add = running_add * gamma + r[t] # calculamos la gradiente segun la constante gamma
        reduced_r[t] = running_add
    return reduced_r # matriz refactorizada con datos actualizados 


if __name__ == '__main__':

    env = gym.make('MsPacman-ram-v0')
    #env.monitor.start('training_dir', force=True)
    # Setup tensorflow
    tf.reset_default_graph()# Borra la pila de graficos predeterminada y restablece el grafico global predeterminado

    observations = tf.placeholder(tf.float32, [None, envSize], name="input_x") # 128
    w1 = tf.get_variable("w1", shape=[envSize, H],#Esta operacion devuelve un tensor de enteros 1-D que representa la forma de input  # 128  100
                         initializer=tf.contrib.layers.xavier_initializer())# Un inicializador para una matriz de peso.
                                                                            #Este inicializador esta disenhado para mantener la escala de los 
                                                                            # degradados aproximadamente igual en todas las capas. En una distribucion 
                                                                            # uniforme, esto termina siendo el rango: x = sqrt(6. / (in + out)); 
                                                                            # [-x, x]y para la distribucion normal sqrt(2. / (in + out))se usa una 
    #capa oculta 1                                                                        # desviacion estandar de .
    hidden_layer_1 = tf.nn.relu(tf.matmul(observations, w1))# creamos la primera capa oculta multiplicando matrizes de obsevaciones por la matriz de peso 1
    w15 = tf.get_variable("w15", shape=[H, H],    # 100 100
                          initializer=tf.contrib.layers.xavier_initializer())
                          
    hidden_layer_2 = tf.nn.relu(tf.matmul(hidden_layer_1, w15))
    w2 = tf.get_variable("w2", shape=[H, 1],
                         initializer=tf.contrib.layers.xavier_initializer())

    result_score = tf.matmul(hidden_layer_2, w2)
    probablility = tf.nn.sigmoid(result_score) # Calcula sigmoide de x elemento (la funcion q permite ver el grado de avanze comprende en rangos de 0a 1)


    training_variables = tf.trainable_variables() # retorna una lista de objetos variables

    #[None, 1]La forma del tensor a alimentar (opcional). Si la forma no esta especificada, puede alimentar un tensor de cualquier forma.
    input_y = tf.placeholder(tf.float32, [None, 1], name="input_y")#de 1 a 4

    advantage = tf.placeholder(tf.float32, name="reward_signal") # tensor para almacenar ventajas(reward signal)

    # Loss Function
    loss = -tf.reduce_mean((tf.log(input_y - probablility)) * advantage) # tensor para almacenar perdidas(revuelve la matriz reducida mean Calcula la media 
                                                                        # de los elementos a traves de las dimensiones de un tensor.)
                                                                        

    new_gradients = tf.gradients(loss, training_variables) # guardamos los nuevos gradientes en base a ls y tv retorn na lista de sum(dy/dx)para cada x en xs.
                                                            # compara ambas listas buscando las variaciones

    # Training

    global_step = tf.Variable(0, trainable=False, name='global_step')


    #adam = tf.train.AdamOptimizer(learning_rate=learn_rate)
    #backP = tf.train.GradientDescentOptimizer(0.1).minimize(cost)   # learning_rate: Un tensor o un valor de coma flotante. La tasa de aprendizaje a usar
    backP = tf.train.GradientDescentOptimizer(learning_rate=learn_rate)  # decenso por gradiente ultilizada para el backpropagation

    w1_gradent = tf.placeholder(tf.float32, name="batch_gradent1") # gradiente de lote

    w2_gradent = tf.placeholder(tf.float32, name="batch_gradent2") # gradiente de lote

    batch_gradent = [w1_gradent, w2_gradent] # El descenso del gradiente por lotes calcula el gradiente utilizando todo el conjunto de datos. Esto es ideal para colectores de errores convexos o relativamente suaves.

    update_gradent = backP.apply_gradients(         # devuelve una lista de tuplas 
        zip(batch_gradent, training_variables))     #(variables de entrenamiento )

    max_episodes = 2000
    max_steps = 500

    #save
    #obs, hideR or Capas,   , rewards, actions , 
    xs, hs, dlogps, drs, ys, tfps = [], [], [], [], [], []
    running_reward = None
    reward_sum = 0
    episode_number = 1
    contador = 0

    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)

        # setting up the training variables
        gradBuffer = sess.run(training_variables)
        for ix, grad in enumerate(gradBuffer):
            gradBuffer[ix] = grad * 0

        for episode in xrange(max_episodes):
            observation = env.reset()# dataJson[i]
            for step in xrange(max_steps):
                if(step == (max_steps - 1)):
                    print 'Made 500 steps!'
                env.render()
                x = np.reshape(observation, [1, envSize])## dataJson[i]

                # get action from policy
                tfprob = sess.run(probablility, feed_dict={observations: x})

                # cambiar teclas
                random = np.random.uniform()
                print("tfprob1: ", tfprob)
                print("random: ", random)
                #tfprob = tfprob * random
                #random = np.arange(4)
                
                cuarto = tfprob/4

                if random < cuarto:
                    action = 0
                elif random > cuarto and random < cuarto*2:
                    action = 1
                elif random > cuarto*2 and random < cuarto*3:
                    action = 2
                else:
                    action = 3
                
                # cuarto = random/4
                # if tfprob < random:
                #     action = 0
                # elif tfprob > random:
                #     action = 1
                # else:
                #     action = 3

                 
                #action = 1 if random< tfprob else 0#   1   0
                
                # will need to rework action to be more generic, not just 1 or 0

                
                #action = np.random.choice(ramd)
                #print("X: ", x)
                print("np.random.uniform(): ", random)
                print("tfprob2: ", tfprob[0][0])
                print("contador: ", contador)
                contador+=1

                #q_values = sess.run(probablility, feed_dict={observations: x})
                #action = epsilon_greedy(q_values, step)

                xs.append(x)  # observation   1   0
                
                # if action == 1 :
                #     y = 4
                # elif action == 2:
                #     y = 3
                # elif action == 3:
                #     y = 2
                # else:
                #     y = 1

                y = action
                print("action: ", y)
                #y = 1 if action == 0 else 0  # something about fake lables, need to investigate
                ys.append(y)

                # run an action
                observation, reward, done, info = env.step(action)
                #print("observation ", observation)
                reward_sum += reward

                drs.append(reward)

                if done:
                    episode_number += 1
                    print 'Episode %f: Reward: %f' % (episode_number, reward_sum)
                    # putting together all inputs, is there a better way to do this?
                    epx = np.vstack(xs)  # metodo del numpy para unir matrices
                    epy = np.vstack(ys)  # metodo del numpy para unir matrices
                    epr = np.vstack(drs)  # metodo del numpy para unir matrices
                    tfp = tfps
                    xs, hs, dlogps, drs, ys, tfps = [], [], [], [], [], []  # reset for next episode

                    # compute reward
                    discounted_epr = reduced_rewards(epr)
                    discounted_epr -= np.mean(discounted_epr)
                    discounted_epr /= np.std(discounted_epr)

                    # get gradient, save in gradent_buffer
                    tGrad = sess.run(new_gradients, feed_dict={
                                     observations: epx, input_y: epy, advantage: discounted_epr})
                    for ix, grad in enumerate(tGrad):
                        gradBuffer[ix] += grad
                        print ("episode_number % batch_number",
                               episode_number % batch_number)

                    if episode_number % batch_number == 0:
                        sess.run(update_gradent, feed_dict={
                                 w1_gradent: gradBuffer[0], w2_gradent: gradBuffer[1]})
                        for ix, grad in enumerate(gradBuffer):
                            gradBuffer[ix] = grad * 0

                        #running_reward = reward_sum if running_reward is None else (
                        #    ((running_reward * episode_number - 50) + (reward_sum * 50)) / episode_number)
                        #print 'Average reward for episode %f. total average reward %f' % (reward_sum / batch_number, running_reward / batch_number)

                        # if reward_sum / batch_number > 475:
                        #     print 'Task solved in', episode_number, 'episodes!'
                        #     reward_sum = 0
                        #     break
                        # reward_sum = 0
                    break

    env.monitor.close()
