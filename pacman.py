# import gym
# env = gym.make('MsPacman-v0')
# env.reset()
# for _ in range(1000):
# 	env.render()
# 	env.step(env.action_space.sample()) # take a random action


import gym
env = gym.make('MsPacman-ram-v0')
# for i_episode in range(20):
observation = env.reset()
for t in range(300):
    env.render()
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    print("observation")
    print(observation)
    print("done")
    print(done)
    print("reward")
    print(reward)
    print("info")
    print(info)
    
    if done:
        print("Episode finished after {} timesteps".format(t+1))
        break

# usaban un archivo como el config.json para aplicar filtros a la observacion
# y darle el observation filtrado al tensorflow ejem. 
# q el ususario filtre solo su observacion con los datos q cambian
#para darle al tensorflow o si queire con todos los datos


#done
#False #varia cuando sucede algo en el entorno como morir el pacman
#reward  
#0.0   #varia en 10.0 si realiza una buena accion como comer un punto
#info
#{'ale.lives': 2}
#observation  
#[  0   0 112 113  51   0  42  93  94  29  74   0  43  80  73 122 116   0
#   0   3   0   0   1   0   0   1   3   6 229  10 210   0  45   2   0 177
#  39 162  60 255   0  50  33   3   2 122   0  58 107 137 154 152   1 221
#   0   0   0   0   2  80 255 255   0 255 255  80 255 255  80 255 255  80
# 255 255  80 191 191  80 191 191  80 191 191  80 171 175  80 255 239  80
# 255 239  80 255 239   0 255 239  80 255 234  20 223  43 217  59 217  51
# 217 123 217 123 217 123 217 221   0  63   0  12  32   1   0   1 215 245
# 146 215] 

