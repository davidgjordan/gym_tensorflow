import numpy as np
import gym

# #  #  # # # # # # # # # GYM # # # # # # # # # # # # # # # # # # # # # # # # # #

list_obs_por_juego, list_action_esperadas_por_juego = [], []

env = gym.make('Enduro-ram-v0')
tam_teclas_disponibles = env.action_space.n  # el pacman es  9

print tam_teclas_disponibles