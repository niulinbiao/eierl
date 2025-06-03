# coding=utf-8
import os, random,sys
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(root_dir)

from collections import namedtuple

import torch
import torch.nn.functional as F
from torch import optim
from torch import nn

import numpy as np
import pandas as pd
import logging

import os,json,copy

from tqdm import tqdm

from deep_dialog.qlearning import EpsilonGreedyPolicy


Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'term'))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device("cpu")



class DQN(object):
    def __init__(self, args):
        # with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), r'../config.json'), 'r') as f:
        #     cfg = json.load(f)
        # self.save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), cfg['save_dir'])
        # self.save_per_epoch = cfg['save_per_epoch']
        # self.training_iter = cfg['training_iter']
        # self.hidden_size = cfg['hidden_size']
        # self.training_batch_iter = cfg['training_batch_iter']
        self.batch_size =args['batch_size']
        self.epsilon = args['epsilon']
        self.gamma = args['gamma']
        self.args = args


        self.dqn = EpsilonGreedyPolicy(args['state_dim'], args['hidden_size'], args['da_dim'],tag="主模型").to(device=DEVICE)
        self.target_dqn = copy.deepcopy(self.dqn)
        self.target_dqn.load_state_dict(self.dqn.state_dict())
        self.target_dqn.eval()

        self.dqn_optim = optim.Adam(self.dqn.parameters(), lr=1e-3)

        self.loss_fn = nn.MSELoss()


    # def calc_q_loss(self,  batch):
    #     '''Compute the Q value loss using predicted and target Q values from the appropriate networks'''
    #     s = torch.from_numpy(np.stack(batch.state)).to(device=DEVICE)
    #     a = torch.from_numpy(np.stack(batch.action)).to(device=DEVICE)
    #     r = torch.from_numpy(np.stack(batch.reward)).to(device=DEVICE)
    #     next_s = torch.from_numpy(np.stack(batch.next_state)).to(device=DEVICE)
    #     mask = torch.Tensor(np.stack(batch.mask)).to(device=DEVICE)
    #     q_preds = self.net(s)
    #     with torch.no_grad():
    #         # Use online_net to select actions in next state
    #         online_next_q_preds = self.online_net(next_s)
    #         # Use eval_net to calculate next_q_preds for actions chosen by online_net
    #         next_q_preds = self.eval_net(next_s)
    #     act_q_preds = q_preds.gather(-1, a.argmax(-1).long().unsqueeze(-1)).squeeze(-1)
    #     online_actions = online_next_q_preds.argmax(dim=-1, keepdim=True)
    #     max_next_q_preds = next_q_preds.gather(-1, online_actions).squeeze(-1)
    #     max_q_targets = r + self.gamma * mask * max_next_q_preds
    #
    #     q_loss = self.loss_fn(act_q_preds, max_q_targets)
    #
    #     return q_loss

    def sample_from_buffer(self, batch_size):
        """Sample batch size examples from experience buffer and convert it to torch readable format"""
        # type: (int, ) -> Transition

        batch = [random.choice(self.running_expereince_pool) for i in xrange(batch_size)]
        np_batch = []
        for x in range(len(Transition._fields)):
            v = []
            for i in xrange(batch_size):
                v.append(batch[i][x])
            np_batch.append(np.vstack(v))

        return Transition(*np_batch)

    def update(self, epoch,memory):
        print "当前经验池数据量：", len(memory)
        self.cur_bellman_err = 0.
        self.running_expereince_pool = list(memory)
        num_batches = 1
        for iter_batch in range(num_batches):
            for iter in tqdm(range(len(self.running_expereince_pool) / ( self.batch_size))):
                self.dqn_optim.zero_grad()
                batch = self.sample_from_buffer( self.batch_size)

                state_value = self.dqn(torch.FloatTensor(batch.state)).gather(1, torch.tensor(batch.action))
                next_state_value, _ = self.target_dqn(torch.FloatTensor(batch.next_state)).max(1)
                next_state_value = next_state_value.unsqueeze(1)
                term = np.asarray(batch.term, dtype=np.float32)
                expected_value = torch.FloatTensor(batch.reward) + self.gamma * next_state_value * (
                        1 - torch.FloatTensor(term))
                loss = F.mse_loss(state_value, expected_value)
                loss.backward()
                self.dqn_optim.step()
                self.cur_bellman_err += loss.item()

            print ('<<dialog policy dqn>> epoch {}, total_loss {}'
                   .format(epoch,
                           float(self.cur_bellman_err) / (len(memory) / (float(self.batch_size)))))


        # 保存损失值
        loss_data = {}
        loss_data['epoch'] = [epoch]
        loss_data['loss'] = [self.cur_bellman_err]
        df = pd.DataFrame(loss_data)
        # 将DataFrame数据写入csv文件（如果文件不存在则创建，如果存在则追加写入）
        df.to_csv('loss_erl_dqn.csv', mode='a', index=False, header=not os.path.exists('loss_erl_dqn.csv'))

        # update the epsilon value
        self.dqn.update_epsilon(epoch)
        # update the target network
        self.target_dqn.load_state_dict(self.dqn.state_dict())



    def save(self, directory, epoch):
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(self.net.state_dict(), directory + '/' + str(epoch)  + '_erl_dqn.pol.mdl')

        print('<<dialog policy>> epoch {}: saved network to mdl'.format(epoch))
