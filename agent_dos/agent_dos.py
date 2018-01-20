import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import gym
# La imagenes tienen dimension de 28x28
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# GYM

list_obs, list_action_esperadas = [], []

env = gym.make('MsPacman-ram-v0')
tam_teclas_disponibles = env.action_space.n  # el pacman es  9

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

        # print("Action: ", action)
        # print("Reward: ", reward)

    if aux_reward > 250.0:
        list_obs.append(aux_obs)
        list_action_esperadas.append(aux_action)
        print("este juego paso: ", i_episode)

    # print("juego: ", i_episode)
    # print("Total Reward por el juego: ", aux_reward)
    aux_reward = 0

for action in list_action_esperadas:
    # print("action: ", action)
    pass
    # LISTAS CARGADAS CON OBSEVATIONS Y ACCIONES ESPERADAS AHORA NORMALIZAR LAS SALIDAS ESPERADAS

list_normalize_actions, aux = [], []
for vec_action_por_juego in list_action_esperadas:
    vec_aux = []
    for act in vec_action_por_juego:
        aux = np.zeros(((tam_teclas_disponibles)))
        # aux = np.linspace(0, 0.01, (tam_teclas_disponibles + 1))
        aux[act] = 1
        vec_aux.append(aux)

    list_normalize_actions.append(vec_aux)

for vec_action in list_normalize_actions:
    #print("vec_action : ", vec_action)
    for data in vec_action:
        # print(data)
        pass


# RED NEURONAL   REVISAR LAS ENTRADAS Y  VOLVER LOS OBSERVATIONS Y ACTIONS A MATRIZES COMO EN EL AGENT UNO
# imagen del numero descompuesta a un vector
x_input_observations = tf.placeholder(tf.float32, [None, 1])
# Matriz de pesos, 784 para recibir la observ, 10 por las posible salidas de teclas
Pesos = tf.Variable(tf.zeros([1, 1]))
                                    # seria conveniete normalizae las teclas en [0,0,1,0,0,0,0,0] proporcionaron la tecla 2 etc
bias = tf.Variable(tf.zeros([1]))  # Vector con bias
# La operacion que se hara en los nodos que reciben entradas
y_oper_nodos_entradas = tf.matmul(x_input_observations, Pesos) + bias
# Matriz con las acciones esperadas de nuestro set de datos
yR_salidas_aciones_esperadas = tf.placeholder(tf.float32, [None, 1])
# Definir la funcion de costo entropia cruzada (Cross Entropy) para poder medir el error. La salida sera con Softmax
fun_softmax_medir_error = tf.nn.softmax_cross_entropy_with_logits(
    labels=yR_salidas_aciones_esperadas, logits=y_oper_nodos_entradas)
costo = tf.reduce_mean(fun_softmax_medir_error)
optimizador = tf.train.GradientDescentOptimizer(0.5).minimize(costo)
# Correr la grafica computacional
prediccion = tf.equal(tf.argmax(y_oper_nodos_entradas, 1), tf.argmax(
    yR_salidas_aciones_esperadas, 1))  # Nos da arreglo de booleanos para decirnos
                                                         # cuales estan bien y_entradas cuales no
# Nos da el porcentaje sobre el arreglo de prediccion
exactitud_predicciones = tf.reduce_mean(tf.cast(prediccion, tf.float32))
# Devuelve el indice con el valor mas grande en los ejes de un tensor. (argumentos en desuso)
# recordando que estamos trabajando sobre una matriz de acciones del tipo [0,0,1,0,0,0,0,0] proporcionaron la tecla 2 etc 1 es el mas alto
Produccion = tf.argmax(y_oper_nodos_entradas, 1)
init = tf.global_variables_initializer()
 # Entrenar algoritmo
# Funcion que usaremos para ver que tan bien va a aprendiendo nuestro modelo


def avance(epoca_i, sess, last_features, last_labels):
    costoActual = sess.run(costo, feed_dict={
                           x_input_observations: last_features, yR_salidas_aciones_esperadas: last_labels})
    Certeza = sess.run(exactitud_predicciones, feed_dict={
                       x_input_observations: list_obs, yR_salidas_aciones_esperadas: list_normalize_actions})
    print(
        'Epoca: {:<4} - Costo: {:<8.3} Certeza: {:<5.3}'.format(epoca_i, costoActual, Certeza))

with tf.Session() as sess:
    sess.run(init)
    for epoca_i in range(1):
        #lotex, lotey = mnist.train.next_batch(100)# lotex = observaciones   lotey = acciones esperadas
        lotex = list_obs
        lotey = list_normalize_actions
        #print("lotex: ", lotex)
        print("lotey: ", lotey)
        sess.run(optimizador, feed_dict={x_input_observations: lotex, yR_salidas_aciones_esperadas: lotey})
        if (epoca_i%50==0):pass
            #avance(epoca_i, sess, lotex, lotey)
    #print('RESULTADO FINAL: ',sess.run(exactitud_predicciones, feed_dict={x_input_observations: mnist.test.images,yR_salidas_aciones_esperadas: mnist.test.labels}))
    #print ('Resultado de una imagen',sess.run(Produccion,feed_dict={x_input_observations: mnist.test.images[5].reshape(1,784)}))
