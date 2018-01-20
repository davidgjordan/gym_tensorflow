import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import gym
# La imagenes tienen dimension de 28x28
#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

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

        #print("Action: ", action)
        #print("Reward: ", reward)

    if aux_reward > 250.0:
        list_obs.append(aux_obs)
        list_action_esperadas.append(aux_action)
        print("este juego paso: ", i_episode)

    #print("juego: ", i_episode)
    #print("Total Reward por el juego: ", aux_reward)
    aux_reward = 0

for action in list_action_esperadas:
    #print("action: ", action)
    pass
    # LISTAS CARGADAS CON OBSEVATIONS Y ACCIONES ESPERADAS AHORA NORMALIZAR LAS SALIDAS ESPERADAS

list_normalize_actions, aux = [], []
for action_elegida_vec in list_action_esperadas:
    for act in action_elegida_vec:
        aux = np.zeros(((tam_teclas_disponibles + 1), 1))
        #aux = np.linspace(0, 0.01, (tam_teclas_disponibles + 1))
        aux[act] = 1
        list_normalize_actions.append(aux)

for vec_action in list_normalize_actions:
    print("vec_action : ")
    for data in vec_action:
        print(data)

# x_input_observatios=tf.placeholder(tf.float32,[None,784]) #imagen del numero descompuesta a un vector
# Pesos=tf.Variable(tf.zeros([784,10])) #Matriz de pesos, 784 para recibir la observ, 10 por las posible salidas de teclas
#                                     #seria conveniete normalizae las teclas en [0,0,1,0,0,0,0,0] proporcionaron la tecla 2 etc
# bias=tf.Variable(tf.zeros([10])) #Vector con bias
# y_oper_nodos_entradas=tf.matmul(x_input_observatios,Pesos)+bias #La operacion que se hara en los nodos que reciben entradas
# yR_salidas_aciones_esperadas=tf.placeholder(tf.float32,[None,10]) # Matriz con las acciones esperadas de nuestro set de datos

# #Definir la funcion de costo entropia cruzada (Cross Entropy) para poder medir el error. La salida sera con Softmax
# fun_softmax_medir_error=tf.nn.softmax_cross_entropy_with_logits(labels=yR_salidas_aciones_esperadas,logits=y_oper_nodos_entradas)
# costo=tf.reduce_mean(fun_softmax_medir_error)
# optimizador=tf.train.GradientDescentOptimizer(0.5).minimize(costo)

# #Correr la grafica computacional
# prediccion = tf.equal(tf.argmax(y_oper_nodos_entradas, 1), tf.argmax(yR_salidas_aciones_esperadas, 1)) #Nos da arreglo de booleanos para decirnos
#                                                          #cuales estan bien y_entradas cuales no
# exactitud_predicciones = tf.reduce_mean(tf.cast(prediccion, tf.float32))#Nos da el porcentaje sobre el arreglo de prediccion

# #Devuelve el indice con el valor mas grande en los ejes de un tensor. (argumentos en desuso)
# Produccion = tf.argmax(y_oper_nodos_entradas,1) # recordando que estamos trabajando sobre una matriz de acciones del tipo [0,0,1,0,0,0,0,0] proporcionaron la tecla 2 etc 1 es el mas alto
# init=tf.global_variables_initializer()


# #Entrenar algoritmo
# #Funcion que usaremos para ver que tan bien va a aprendiendo nuestro modelo
# def avance(epoca_i, sess, last_features, last_labels):
#     costoActual = sess.run(costo,feed_dict={x_input_observatios: last_features, yR_salidas_aciones_esperadas: last_labels})
#     Certeza = sess.run(exactitud_predicciones,feed_dict={x_input_observatios:mnist.validation.images,yR_salidas_aciones_esperadas: mnist.validation.labels})
#     print('Epoca: {:<4} - Costo: {:<8.3} Certeza: {:<5.3}'.format(epoca_i,costoActual,Certeza))


# with tf.Session() as sess:
#     sess.run(init)
#     for epoca_i in range(1000):
#         lotex, lotey = mnist.train.next_batch(100)# lotex = observaciones   lotey = acciones esperadas
#         sess.run(optimizador, feed_dict={x_input_observatios: lotex, yR_salidas_aciones_esperadas: lotey})
#         if (epoca_i%50==0):
#             avance(epoca_i, sess, lotex, lotey)
#     print('RESULTADO FINAL: ',sess.run(exactitud_predicciones, feed_dict={x_input_observatios: mnist.test.images,yR_salidas_aciones_esperadas: mnist.test.labels}))
#     print ('Resultado de una imagen',sess.run(Produccion,feed_dict={x_input_observatios: mnist.test.images[5].reshape(1,784)}))
#     print("Produccion: " , sess.run(Produccion) )


''' Epoca: 0    - Costo: 1.8      Certeza: 0.111
Epoca: 50   - Costo: 0.382    Certeza: 0.875
Epoca: 100  - Costo: 0.407    Certeza: 0.898
Epoca: 150  - Costo: 0.316    Certeza: 0.896
Epoca: 200  - Costo: 0.34     Certeza: 0.908
Epoca: 250  - Costo: 0.313    Certeza: 0.912
Epoca: 300  - Costo: 0.286    Certeza: 0.907
Epoca: 350  - Costo: 0.391    Certeza: 0.914
Epoca: 400  - Costo: 0.509    Certeza: 0.909
Epoca: 450  - Costo: 0.478    Certeza: 0.914
Epoca: 500  - Costo: 0.506    Certeza: 0.914
Epoca: 550  - Costo: 0.215    Certeza: 0.917
Epoca: 600  - Costo: 0.273    Certeza: 0.914
Epoca: 650  - Costo: 0.227    Certeza: 0.917
Epoca: 700  - Costo: 0.224    Certeza: 0.919
Epoca: 750  - Costo: 0.177    Certeza: 0.922
Epoca: 800  - Costo: 0.324    Certeza: 0.919
Epoca: 850  - Costo: 0.248    Certeza: 0.924
Epoca: 900  - Costo: 0.258    Certeza: 0.923
Epoca: 950  - Costo: 0.162    Certeza: 0.918
RESULTADO FINAL:  0.92
Resultado de una imagen [1]
In [8]:
mnist.test.labels[5]
Out[8]:
array([ 0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.])
In [ ]: '''
