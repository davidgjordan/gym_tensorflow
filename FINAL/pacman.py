import random
import sys
sys.path.insert(0,'/home/ubuntu/Desktop')
import gym
import tensorflow as tf
import numpy as np
import pygame


list_obs_por_juego, list_action_esperadas_por_juego = [], []

env = gym.make('MsPacman-ram-v0')
tam_teclas_disponibles = env.action_space.n  # el pacman es  9
#print tam_teclas_disponibles

##method for render image
def display_arr(screen, arr, transpose, video_size):
    arr_min, arr_max = arr.min(), arr.max()
    arr = 255.0 * (arr - arr_min) / (arr_max - arr_min)
    pyg_img = pygame.surfarray.make_surface(arr.swapaxes(0, 1) if transpose else arr)
    pyg_img = pygame.transform.scale(pyg_img, video_size)
    screen.blit(pyg_img, (0,0))
###



print("RECOLECCION DE DATOS PARA EL ENTRENAMIENTO")
espectedReward = 250
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
        velocity = 15000
        clock = pygame.time.Clock()
        screen = pygame.display.set_mode((480,630))
        pygame.display.set_caption(u'OBTENIENDO DATA DE ENTRENAMIENTO')
        #########################################
        
        while not done:
            rgb_array = env.render(mode='rgb_array')#env.render() 
            #action = env.action_space.sample()
            action = random.choice([5,6,7,8])
            observation, reward, done, info = env.step(action)#random.randint(0, 5)
            aux_reward += reward
            aux_action.append(action)
            aux_obs.append(observation)
            ####################render images##################
            if observation is not None:
                display_arr(screen, rgb_array, True, (480,630))
            pygame.display.flip()
            clock.tick(velocity)
            ###################################################
        
        if aux_reward > espectedReward:
            print 'Juego Numero: {0} PASO - recompensa: {1} - juego: {2}/{3}'.format(gameNumber,aux_reward, successGame,gameNumber)
            successGame =  successGame +1
            global list_obs_por_juego
            global list_action_esperadas_por_juego
            if len(list_obs_por_juego) == 0:
                list_obs_por_juego =  aux_obs
                list_action_esperadas_por_juego =  aux_action
            if len(list_obs_por_juego) > 0:
                 list_obs_por_juego = list_obs_por_juego +  aux_obs
                 list_action_esperadas_por_juego = list_action_esperadas_por_juego +  aux_action
        else:
            print 'Juego Numero: {0} NO PASO'.format(gameNumber)
        gameNumber = gameNumber +1
        aux_reward = 0

    pygame.quit()



correr_episodios_gym()####DATA 
