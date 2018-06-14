import torch
import torch.nn as nn
from torch.autograd import Variable
from copy import deepcopy

import numpy as np

class PSO():
    def __init__(self, module, lr, std, b, n_directions):
        assert b<=n_directions, "b must be <= n_directions"
        # hp:
        self.lr = lr
        self.std = std
        self.b = b
        self.n_directions = n_directions
        # parameters
        self.module = module
        self.module_buffer = deepcopy(module)
        self.module_st = deepcopy(module).state_dict()
        self.modules_left = [{name:None for name,_ in module.named_parameters()} for _ in range(self.n_directions)]
        self.modules_right = [{name:None for name,_ in module.named_parameters()} for _ in range(self.n_directions)]
        # memory:
        self.rewards_left = [0 for _ in range(self.n_directions)]
        self.rewards_right = [0 for _ in range(self.n_directions)]

    def sample(self):
        for k in range(self.n_directions):
            for name, p in self.module.named_parameters():
                delta = self.std*torch.randn(p.size())*(torch.rand(p.size())>0.8).float()
                self.modules_left[k][name] = deepcopy(self.module_st[name]) - delta
                self.modules_right[k][name] = deepcopy(self.module_st[name]) + delta

    def evaluate(self, state, direction=None, side='left'):
        state = Variable(state)
        if direction is None:
            self.module_buffer.load_state_dict(self.module_st)
        else:
            if side=='left':
                self.module_buffer.load_state_dict(self.modules_left[direction])
            else:
                self.module_buffer.load_state_dict(self.modules_right[direction])
        action = self.module_buffer(state)
        return action.data

    def reward(self, reward, direction, side='left'):
        if side=='left':
            self.rewards_left[direction] += reward
        else:
            self.rewards_right[direction] += reward

    def update(self):
        # sd of rewards:
        sigma_r = torch.Tensor(self.rewards_left + self.rewards_right).std()

        # sort rollouts in best score order, take the 'b' best ones:
        scores = {k:max(r_left, r_right) for k,(r_left, r_right) in enumerate(zip(self.rewards_left, self.rewards_right))}
        order = sorted(scores.keys(), key=lambda x:scores[x])[-self.b:]
        rollouts = [(self.rewards_left[k], self.rewards_right[k], k) for k in order[::-1]]

        # update main state dict:
        for name, p in self.module.named_parameters():
            step = 0
            for r_left, r_right, k in rollouts:
                delta = self.modules_right[k][name] - self.module_st[name]
                step += (r_right - r_left) * delta/self.std
            self.module_st[name] += self.lr*step / (sigma_r*self.b)

        # update module:
        #self.module.load_state_dict(self.module_st)

        # reinitialize memory
        self.rewards_left = [0 for _ in range(self.n_directions)]
        self.rewards_right = [0 for _ in range(self.n_directions)]
