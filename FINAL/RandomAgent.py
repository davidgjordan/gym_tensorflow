#!/usr/bin/env python
# -*- coding: utf-8 -*
import sys
import time
import random
import pygame
sys.path.insert(0,'/home/ubuntu/Desktop')
import gym
import json
import numpy as np
from gym import wrappers

class RandomAgent(object):
    def __init__(self):
        self.video_size = (480,630)
        self.transpose = True
        self.done=False
        self.total_reward=0
        self.data_ram=dict()
        self.data_rgb=dict()
        self.output =""
        self.count=0
        self.count2=0
        self.interval_1 =0
        self.interval_2 =0 
        self.interval_3 =0
        self.interval_of_time = 0
        self.frames_by_decision =0
        self.keys_1 =0 
        self.keys_2 =0 
        self.keys_3 =0
        self.interval_deny_last_key=0  
        self.deny_key = self.Inverse(3)
        self.Steps = 0
        self.ArrayNumberKeys =[]
        self.name_game ="" 
        self.Actions=[]
        self.numberKeys=0
    def getObservations(self):
        self.output = {"Observation":self.data_ram, "Reward":self.total_reward, "Steps": self.Steps, "Actions":self.Actions, "NumberKeys":self.numberKeys,"NameGame":self.name_game}
        output_string = json.dumps(self.output)
        #print (output_string)
        sys.stdout.write(output_string) 
    def display_arr(self,screen, arr, transpose, video_size):
        arr_min, arr_max = arr.min(), arr.max()
        arr = 255.0 * (arr - arr_min) / (arr_max - arr_min)
        pyg_img = pygame.surfarray.make_surface(arr.swapaxes(0, 1) if transpose else arr)
        pyg_img = pygame.transform.scale(pyg_img, video_size)
        screen.blit(pyg_img, (0,0))
    def Inverse(self, key):
        if key==4:
            return 1
        if key==1:
            return 4
        if key==2:
            return 3
        if key==3:
            return 2
    def Generate_by_intervals(self):
        if self.interval_of_time <=self.interval_1:
            self.interval_of_time+=1
            return random.choice(self.ArrayNumberKeys)
        if self.interval_of_time > self.interval_1 and self.interval_of_time< self.interval_2:
            self.interval_of_time+=1
            return random.choice([5,6,7,8])
        if self.interval_of_time >= self.interval_2 and self.interval_of_time<= self.interval_3:
            self.interval_of_time+=1
            return random.choice([2,3,4])
        if self.interval_of_time>=self.interval_3:  
            self.interval_of_time=0
            return random.choice(self.keys_1)
    def Generate(self, last_key):
        self.count+=1
        if self.count>self.frames_by_decision:
            self.count=0
            gen = self.Generate_by_intervals()
            self.count2+=1
            if self.count2<self.interval_deny_last_key:
                while self.deny_key == gen:
                    gen = self.Generate_by_intervals()
            else:
                self.deny_key = self.Inverse(last_key)
            return gen
        return last_key
    def setRandomParameters(self, stepByInterval, key_1, key_2, key_3, DenyLastkeyBySteps, DecisionsBySecond):
        self.interval_1 = stepByInterval[0]
        self.interval_2 = stepByInterval[1]
        self.interval_3 = stepByInterval[2]
        self.interval_of_time = 0
        self.frames_by_decision = DecisionsBySecond
        self.keys_1 = key_1
        self.keys_2 = key_2
        self.keys_3 = key_3
        self.interval_deny_last_key = DenyLastkeyBySteps
    def run(self, name_game, velocity):
        self.name_game = name_game
        number_games = 5
        counter=0
        greater_reward =0
        env = gym.make(name_game)
        self.numberKeys = env.action_space.n
        for x in range(0, self.numberKeys):
            self.ArrayNumberKeys.append(x);
        self.done = False
        last_key = 3
        steps = 0
        env.reset()
        clock = pygame.time.Clock()
        screen = pygame.display.set_mode(self.video_size)
        pygame.display.set_caption(u'DEVINT-24 GAMES - Random')
        while True:
            if not self.done:           
                obs,reward, self.done, rgb_array = env.step(last_key)
                self.total_reward+=reward
                self.Steps+=1
                self.data_ram[self.Steps]=obs.tolist()
                self.Actions.append(last_key)
            else:
                if counter<number_games:
                    self.done =False
                    if greater_reward<self.total_reward:
                        greater_reward=self.total_reward
                    self.total_reward=0
                    counter+=1
                    env.reset()
                else:
                    self.total_reward=greater_reward
                    break
            randgen = self.Generate(last_key)
            if randgen!=None:
                last_key=randgen
            if obs is not None:
                self.display_arr(screen, rgb_array, self.transpose, self.video_size)
            pygame.display.flip()
            clock.tick(velocity)
        pygame.quit()


if __name__ == '__main__':
    a = RandomAgent()
    a.setRandomParameters([4,10,30],[0,1,2,3,4,5,6,7,8],[0,1,2,3,4,5],[5,6,7,8],120,5)
    #print (sys.argv[1])
    a.run("MsPacman-ram-v0", 200)
    a.getObservations()
