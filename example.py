import os
import sys
import torch
import torch.nn as nn
import gym
from gym import wrappers
import mujoco_py
import pybullet_envs

from pso import PSO
from utils import Normalizer, mkdir

# hyper parameters
class Hp():
    def __init__(self):
        self.main_loop_size = 100
        self.horizon = 1000
        self.lr = 0.02
        self.n_directions = 8
        self.b = 8
        assert self.b<=self.n_directions, "b must be <= n_directions"
        self.std = 0.03
        self.seed = 1
        ''' chose your favourite '''
        #self.env_name = 'Reacher-v1'
        #self.env_name = 'Pendulum-v0'
        #self.env_name = 'HalfCheetahBulletEnv-v0'
        #self.env_name = 'Hopper-v1'#'HopperBulletEnv-v0'
        #self.env_name = 'Ant-v1'#'AntBulletEnv-v0'#
        self.env_name = 'HalfCheetah-v1'
        #self.env_name = 'Swimmer-v1'
        #self.env_name = 'Humanoid-v1'

def run(env, pso, normalizer, state, direction=None, side='left'):
    normalizer.observe(state)
    state = normalizer.normalize(state)
    state = torch.from_numpy(state).float()
    action = pso.evaluate(state, direction, side).numpy()
    state, reward, done, _ = env.step(action)
    reward = max(min(reward, 1), -1)
    if direction is not None:
        pso.reward(reward, direction, side)
    return state, reward, done

# training loop
def train(env,pso, normalizer, hp):
    fitness = []
    for episode in range(hp.main_loop_size):
        # init perturbations
        pso.sample()

        # perturbations left
        for k in range(hp.n_directions):
            state = env.reset()
            done = False
            num_plays = 0
            while not done and num_plays<hp.horizon:
                state, reward, done = run(env, pso, normalizer, state, k, 'left')
                num_plays += 1

        # perturbations right
        for k in range(hp.n_directions):
            state = env.reset()
            done = False
            num_plays = 0
            while not done and num_plays<hp.horizon:
                state, reward, done = run(env, pso, normalizer, state, k, 'right')
                num_plays += 1

        # update policy
        pso.update()

        # evaluate
        state = env.reset()
        done = False
        reward_evaluation = 0
        while not done :
            state, reward, done = run(env, pso, normalizer, state)
            reward_evaluation += reward

        # finish, print:
        print('episode',episode,'reward_evaluation',reward_evaluation)
        fitness.append(reward_evaluation)
    return fitness

if __name__ == '__main__':
    hp = Hp()

    work_dir = mkdir('exp', 'brs')
    monitor_dir = mkdir(work_dir, 'monitor')
    env = gym.make(hp.env_name)

    env.seed(hp.seed)
    torch.manual_seed(hp.seed)
    env = wrappers.Monitor(env, monitor_dir, force=True)

    num_inputs = env.observation_space.shape[0]
    num_outputs = env.action_space.shape[0]

    policy = nn.Linear(num_inputs, num_outputs, bias=True)
    policy.weight.data.fill_(0)
    policy.bias.data.fill_(0)

    pso = PSO(policy, hp.lr, hp.std, hp.b, hp.n_directions)
    normalizer = Normalizer(num_inputs)
    fitness = train(env, pso, normalizer, hp)
    import matplotlib.pyplot as plt
    plt.plot(fitness)
    plt.show()
