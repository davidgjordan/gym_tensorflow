import tensorflow as tf
import numpy as np
import gym
import pygame


# #  #  # # # # # # # # # GYM # # # # # # # # # # # # # # # # # # # # # # # # # #

list_obs_por_juego, list_action_esperadas_por_juego = [], []

env = gym.make('Asterix-ram-v0')
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

        if aux_reward > 200.0:
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
        if  aux_obs_copy_pila:
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
#print(mat_ac)
#print("Observations: ")
#print(mat_obs)
#
#
#mat_obs, mat_ac = get_lote(5)
#
#print("Actions: ")
#print(mat_ac)
#print("Observations: ")
#print(mat_obs)
# #  #  # # # # # # # # # FIN GYM # # # # # # # # # # # # # # # # # # # # # # # # # #



# #  # # # # # # # ## ##  RED NEURONAL   REVISAR LAS ENTRADAS Y  VOLVER LOS OBSERVATIONS Y ACTIONS A MATRIZES COMO EN EL AGENT UNO
# imagen del numero descompuesta a un vector
x_input_observations = tf.placeholder(tf.float32, [None, 128])
# Matriz con las acciones esperadas de nuestro set de datos
yR_salidas_aciones_esperadas = tf.placeholder(tf.float32, [None, tam_teclas_disponibles])
# Matriz de pesos, 784 para recibir la observ, 10 por las posible salidas de teclas
Pesos = tf.Variable(tf.zeros([128, tam_teclas_disponibles]))
                                    # seria conveniete normalizae las teclas en [0,0,1,0,0,0,0,0] proporcionaron la tecla 2 etc
bias = tf.Variable(tf.zeros([tam_teclas_disponibles]))  # Vector con bias
# La operacion que se hara en los nodos que reciben entradas
y_oper_nodos_entradas = tf.matmul(x_input_observations, Pesos) + bias
# Definir la funcion de costo entropia cruzada (Cross Entropy) para poder medir el error. La salida sera con Softmax
fun_softmax_medir_error = tf.nn.softmax_cross_entropy_with_logits(
    labels=yR_salidas_aciones_esperadas, logits=y_oper_nodos_entradas)
costo = tf.reduce_mean(fun_softmax_medir_error)

optimizador = tf.train.GradientDescentOptimizer(0.1).minimize(costo)
# Correr la grafica computacional
prediccion = tf.equal(tf.argmax(y_oper_nodos_entradas, 1), tf.argmax(
    yR_salidas_aciones_esperadas, 1))  # Nos da arreglo de booleanos para decirnos
                                                         # cuales estan bien y_entradas cuales no
# Nos da el porcentaje sobre el arreglo de prediccion
exactitud_predicciones = tf.reduce_mean(tf.cast(prediccion, tf.float32))
# Devuelve el indice con el valor mas grande en los ejes de un tensor. (argumentos en desuso)
# recordando que estamos trabajando sobre una matriz de acciones del tipo [0,0,1,0,0,0,0,0] proporcionaron la tecla 2 etc 1 es el mas alto
Produccion = tf.argmax(y_oper_nodos_entradas, 1)#tf.argmax retorna el indice con el valor mas grande del tensor q en esete caso es la tecla de salida del entrenamiento
init = tf.global_variables_initializer()
 # Entrenar algoritmo
# Funcion que usaremos para ver que tan bien va a aprendiendo nuestro modelo


# Add ops to save and restore all the variables.
saver = tf.train.Saver()


def avance(epoca_i, sess, last_features, last_labels):
    costoActual = sess.run(costo, feed_dict={
                           x_input_observations: last_features, yR_salidas_aciones_esperadas: last_labels})
    mat_normalize_obs , _ = get_normalizar_observations()                           
    mat_normalize_actions , _ = get_normalizar_actions()                           
    Certeza = sess.run(exactitud_predicciones, feed_dict={
                       x_input_observations: mat_normalize_obs, yR_salidas_aciones_esperadas: mat_normalize_actions})
    print(
        'Epoca: {:<4} - Costo: {:<8.3} Certeza: {:<5.3}'.format(epoca_i, costoActual, Certeza))


    

with tf.Session() as sess:
    sess.run(init)
    tf.global_variables_initializer()
    epoca_i = 0
    while len(aux_act_copy_pila) != 0:

    #for epoca_i in range(1000):
        #lotex, lotey = mnist.train.next_batch(100)# lotex = observaciones   lotey = acciones esperadas
                
        lotex, lotey = get_lote(3)
        
        opt = sess.run(optimizador, feed_dict={x_input_observations: lotex, yR_salidas_aciones_esperadas: lotey})
        print("OPT: ", opt)
        print("LOTE x: ", lotex)
        print("LOTE y: ", lotey)
        if (epoca_i%50==0):
            avance(epoca_i, sess, lotex, lotey)

            lis_obs, ___ = get_normalizar_observations()
            mat_ob = np.vstack(lis_obs[5])
            #print("mat 5: ", lis_obs[5].reshape(1,128))
            #print ('Resultado de una imagen',sess.run(Produccion,feed_dict={x_input_observations: lis_obs[5].reshape(1,128)}))
            #print ('Resultado de una imagen',sess.run(Produccion,feed_dict={x_input_observations: mat_normalize_obs[5].reshape(1,128) }))
        epoca_i+=1


## # # #  # ## # # # # # # # PROBANDO LA PREDCICION DE LA RED NEURONAL # # # # ##
    for i_episode in range(10):
        observation = env.reset()
        aux_reward = 0
        reward = 0
        done = False
        list_aux_ac = []
        
        while not done:
            env.render()
            action = sess.run(Produccion,feed_dict={x_input_observations: observation.reshape(1,128)})
            #print("action elegida: ", action)
            observation, reward, done, info = env.step(action)
            aux_reward += reward
            list_aux_ac.append(action[0])
        if aux_reward > 200.0:
            print("este juego paso o:: ", i_episode)
        aux_reward = 0
        #print("action elegida list: ", list_aux_ac)
            
    ######SALVAR
    save_path = saver.save(sess, "./tmp_tres/model.ckpt")
    print("Model saved in file: %s" % save_path)

    # aver.restore(sess, "/tmp/model.ckpt")
    # print("Model restored.")  


## # # #  # ## # # # # # # # PROBANDO LA PREDCICION DE LA RED NEURONAL # # # # ##

# Script::Train(observation , Actions, numero_games, numero_ciclos, rewards_esperado){
#     configurarNN(observation , Actions);# configurar red con los nuevos tamanos
#     NNentrenada = train(observation , Actions);# entrenar red

#     for i < numero_ciclos i++{ #numero de juegos de los cuales  se guardaran partidas de los q aprueben el rewar

#         for i < numero_games i++{ #
#             NNActions = NNentrenada.solve(OpenAI.getObs());
#             OpenAI.setActions(NNActions); #entorno.step(NNActions)
#         }
#         rewarCAlculate();
#     }

#     Nmatches([obs, action],[obs, action]);
# }
# NNotra = retrain(NNmatches)                                                                                    




#############################################################################################


## # # #  # ## # # # # # # # PROBANDO LA PREDCICION DE LA RED NEURONAL # # # # ##

ConfigTrain configTrain;
path_redentrenada propiedad de clase;
metodo privafdo  

Game{
  name_game#nombre del juego
  observation  # todas las observaciones filtradas vector<vector<double>>
  Actions   # vector
  rewards_total # double
  cantida steps # int tamano de las filas de las obs
  number_keys_available # int tamano de las teclas avilitadas por el entorno
}

ConfigTrain{
    vector_indices # vector indices a seleccionar para el filtrado
    numero_ciclos,# int cantidad de ciclos del entrenamiento
    numero_games, # int canrtidad de juegos por ciclo de los cuales se filtrara las nuevas obs
    rewards_esperado # double minimo reward para selecciuonar ese juego 
}

Train(Obser , ConfigTrain , vector_indices ){

}

Script::Train(Game game, ConfigTrain configTrain){
    
    
    configNetwork( game , configTrain);# configurar red con los nuevos tamanos
    
    string path_redentrenada = train(game);# entrenar red

    int nnactions = 0;
    OpenAI env;
    vector <Observations> list_obs_por_juego;
    reward_total = 0;

    for i=0 ; i < numero_ciclos ;  i++{ #numero de juegos de los cuales  se guardaran partidas de los q aprueben el rewar
        
        Observations obs_por_juego_aux;

        for i=0 ;i < numero_games ; i++{ #
            # el python deveria devolverme un game completo
            nnactions = solve(env.getObs(), path_redentrenada);
            observations , reward = env.setActions(nnactions); #entorno.step(NNActions)
            reward_total+= reward;
            obs_por_juego_aux.observations.push(observations);#
        }
        
        bool passed = rewarCalculate(reward_total ,rewards_esperado);
        if passed  list_obs_por_juego.push(obs_por_juego_aux);
        reward_total = 0;
        observations = list_obs_por_juego;
    }

    #NNotra = retrain(list_obs_por_juego)                                                                                    
}



bool rewarCalculate(rewards_actual, rewards_esperado){
    #calcula  si las obs pasan el rewards_esperado
    return rewards_actual > rewards_esperado ? true :false;
}

json json:

void configNetwork( game , configTrain){
    # crear un json para mandarle al script
    # json {
    #    method_1: train:
    #               parameters:[
    #                   ciclos: configtTrain.ciclos,
    #                   numgames:configtTrain.num_games
    #                   input_obs_a_0: observations[0].size() 
    #                   input_salida_y: configTrain.num_teclas
    #                   obser: game.observations
    #                   actions: game.actions
    #                   vector_indices: configTrain.vecIndices
    #                   nameGame:game.name
    #                   rewards_esperado: configTrain.rewards_esperado
    #                    path_salida_network: confiTrain.path
    #               ]
    # }
}

string train(Game game){

    string path_script  = crearFileScript();
    #metodo del padre
    _path_script = path_script;

    PythonScript::addParameter(json.dump());
    PythonBuilder::run();

    string path_red_entrenada = PythonScript::getResult();
    return path_red_entrenada;
}
