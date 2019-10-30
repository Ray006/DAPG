# Learning Complex Dexterous Manipulation with Deep Reinforcement Learning and Demonstrations
# This code is just for observing the action space of the hand 
#
import mj_envs
import click 
import os
import gym
import numpy as np
import pickle
from ipdb import set_trace


def demo_playback(env_name, demo_paths=0):
    e = gym.make(env_name)

    # print(e.action_space)
    # print(e.observation_space)
    # print(e.observation_space.high)
    # print(e.observation_space.low)

    aa = np.zeros_like(e.action_space.sample())
    # set_trace()
    # aa = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    # actions = e.action_space.sample()
    i=0
    count = 0

    e.reset()
    for _ in range(100000):
        for path in range(100000):

            if path%50 == 0: # frequence of taking a new action

                if count <= 10 or count>=30: # the initial value is 0, but the boundary is [-1,1], 
                    # so it goes up from 0->1(count 0-10) then down from 1->-1 finally up -1->0(count 30-40) 
                    step = 0.1
                    if i<4:             # the first three dimensions is the positin control of the whole hand. should be scaled down the value
                        step = 0.02

                if count >= 10 and count<30:
                    step = -0.1  
                    if i<4:
                        step = -0.02

                aa[i] += step 

                count += 1        # record the number of the step
                if count == 40:   # if finishing a test of one action dimension,then turn to next action dimension
                    count = 0
                    i+=1          # turn to next action dimension.
                if i == len(aa):  # if all action dimensions are tested, repeat the test.
                    i=0

                # print(path)
                print('Value of '+str(i)+'th action:',aa[i])


            actions = aa
            e.step(actions)
            e.env.mj_render()

if __name__ == '__main__':
    
    demo_playback('relocate-v0')
