#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class NoisyLinear(nn.Module):
    """
        It defines the layer noisy, which will receive p-dim x, and return q-dim y
    """

    # in_features: p, out_features: q
    # so: mu and sigma of weight is q*p; of bias is q.
    def __init__(self, in_features, out_features, sigma_init=0.4):

        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.Tensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer('bias_epsilon', torch.Tensor(out_features))
        self.reset_parameters()
        self.reset_noise()

    def forward(self, x,is_train=True):
        if is_train:
            weight = self.weight_mu + self.weight_sigma.mul(Variable(self.weight_epsilon))
            bias = self.bias_mu + self.bias_sigma.mul(Variable(self.bias_epsilon))
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)

    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / np.sqrt(self.out_features))

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def _scale_noise(self, size):
        x = torch.randn(size)
        x = x.sign().mul(x.abs().sqrt())
        return x



class NoisyModel(nn.Module):
    def __init__(self, s_dim, a_dim,h_dim):
        super(NoisyModel, self).__init__()
        self.s_dim  = s_dim
        self.a_dim  = a_dim
        self.h_dim  = h_dim
        self.fc1 = nn.Linear(s_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, h_dim)
        self.fc3 = nn.Linear(h_dim, a_dim)
        self.sigma_init = 0.0017
        self.epsilon = 0.1
        self.noisy1 = NoisyLinear(h_dim, h_dim, self.sigma_init)
        self.noisy2 = NoisyLinear(h_dim, a_dim, self.sigma_init)

    def forward(self, x,is_train=True):
        x = x.type(torch.FloatTensor)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        # x = self.fc3(x)
        x = torch.relu(self.noisy1(x,is_train))
        x = self.noisy2(x,is_train)
        return x

    def select_action(self, s,is_train=True):
        # if is_train:
        #     if self.epsilon > np.random.rand():
        #         # select a random action
        #         a = torch.randint(self.a_dim, (1, ))
        #     else:
        #         a = self._greedy_action(s, is_train=True)
        # else:
        #     a = self._greedy_action(s,is_train=False)
        a = self._greedy_action(s, is_train=is_train)
        # a = self._greedy_action(s,is_train=is_train)
        # transforms action index to a vector action (one-hot encoding)
        # a_vec = torch.zeros(self.a_dim)
        # a_vec[a] = 1.
        return a


    def _greedy_action(self, s,is_train=False):
        """
        Select a greedy action
        """
        a_weights = self.forward(s,is_train)
        return a_weights.argmax(0, True)

    def reset_noise(self):
        self.noisy1.reset_noise()
        self.noisy2.reset_noise()