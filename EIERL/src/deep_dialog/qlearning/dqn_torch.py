#!/usr/bin/env python
# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F




class EpsilonGreedyPolicy(nn.Module):
    def __init__(self, s_dim, h_dim, a_dim,tag, epsilon_spec={'start': 0.1, 'end': 0.0, 'end_epoch': 200}):
        super(EpsilonGreedyPolicy, self).__init__()



        self.linear_i2h = nn.Linear(s_dim,h_dim)
        self.linear_h2o = nn.Linear(h_dim, a_dim)

        self.epsilon = epsilon_spec['start']
        self.start = epsilon_spec['start']
        self.end = epsilon_spec['end']
        self.end_epoch = epsilon_spec['end_epoch']
        self.a_dim = a_dim
        self.tag = tag

    def forward(self, s):
        x = F.tanh(self.linear_i2h(s))
        x = self.linear_h2o(x)
        return x


    def select_action(self, s, is_train=True):
        """
        :param s: [s_dim]
        :return: [1]
        """
        # forward to get action probs
        # [s_dim] => [a_dim]
        if is_train:
            if self.epsilon > np.random.rand():
                # select a random action
                a = torch.randint(self.a_dim, (1, ))
            else:
                a = self._greedy_action(s)
        else:
            a = self._greedy_action(s)

        # transforms action index to a vector action (one-hot encoding)
        a_vec = torch.zeros(self.a_dim)
        a_vec[a] = 1.

        return a_vec



    def update_epsilon(self, epoch):
        # Linear decay
        a = -float(self.start - self.end) / self.end_epoch
        b = float(self.start)
        self.epsilon = max(self.end, a * float(epoch) + b)
        return self.epsilon


    def _greedy_action(self, s):
        """
        Select a greedy action
        """
        a_weights = self.forward(s)
        return a_weights.argmax(0, True)

    def clean_action(self, s, return_only_action=True):

        a = self._greedy_action(s)
        # # transforms action index to a vector action (one-hot encoding)
        # a_vec = torch.zeros(self.a_dim)
        # a_vec[a] = 1.
        return a

    # def noisy_action(self, obs, return_only_action=True):
    #     a = torch.randint(self.a_dim, (1,))
    #     # transforms action index to a vector action (one-hot encoding)
    #     a_vec = torch.zeros(self.a_dim)
    #     a_vec[a] = 1.
    #
    #     return a_vec
    #
    #
    # def noisy_action2(self, obs, return_only_action=True):
    #     if self.epsilon > np.random.rand():
    #         # select a random action
    #         a = torch.randint(self.a_dim, (1,))
    #     else:
    #         a = self._greedy_action(obs)
    #     # transforms action index to a vector action (one-hot encoding)
    #     a_vec = torch.zeros(self.a_dim)
    #     a_vec[a] = 1.
    #
    #     return a_vec
