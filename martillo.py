import gym
env = gym.make('CartPole-v1')
for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        print("obs: ", observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            break


# Los valores en el parametro de observacion muestran la posicioclearn (x), la velocidad (x_dot),
# el angulo (theta) y la velocidad angular (theta_dot).
# scratch  investigar
# entender realmente lo q es un lenguaje de programacion
# docker containers(PAUSA)
# revisar tensor flow
# ARREGLO -> TODA LA INFORMACION ESTAN EN POSICIONES DE MEMORIA CONTIGUAS
# LISTA   -> MODELO DE ELEMENTOS CONTINUOS ALMACENADOS EN DISTINTOS LUGARES  (PORQ CADA UNO SABE LA REFERENCIA AL OTRO)

# REVISAR STRING LITERALS, CHAR, CONCATENAR CORTARLAS, Q SON LOS CSTRING(ARREGLOS DE CARACTERES POR PUNTEROS)
# MANEJAR CHAR *
