# coding=utf-8
'''
Created on Jun 18, 2016

An DQN Agent

- An DQN
- Keep an experience_replay pool: training_data <State_t, Action, Reward, State_t+1>
- Keep a copy DQN

Command: python .\run.py --agt 9 --usr 1 --max_turn 40 --movie_kb_path .\deep_dialog\data\movie_kb.1k.json --dqn_hidden_size 80 --experience_replay_pool_size 1000 --replacement_steps 50 --per_train_epochs 100 --episodes 200 --err_method 2
'''
#-*- coding: UTF-8 -*-
import logging
import os
import sys
import random, copy, json
import cPickle as pickle
import numpy as np
import pandas as pd
import torch
from torch import optim
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F


from  deep_dialog import dialog_config

from deep_dialog.agents import Agent
from collections import namedtuple, deque

from deep_dialog.agents.agent_noise.model import NoisyModel

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'term'))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class AgentNoiseDQN(Agent):
    def __init__(self, movie_dict=None, act_set=None, slot_set=None, params=None):
        print "加载NOISY_DQN算法！"
        self.movie_dict = movie_dict
        self.act_set = act_set
        self.slot_set = slot_set
        self.act_cardinality = len(act_set.keys())
        self.slot_cardinality = len(slot_set.keys())

        self.training_iter = 10
        self.training_batch_iter = 1
        self.save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "save")
        
        self.feasible_actions = dialog_config.feasible_actions
        self.num_actions = len(self.feasible_actions)
        
        self.epsilon = params['epsilon']
        self.agent_run_mode = params['agent_run_mode']
        self.agent_act_level = params['agent_act_level']
        self.experience_replay_pool_size = params.get('experience_replay_pool_size', 1000)
        self.experience_replay_pool = deque(maxlen=self.experience_replay_pool_size)
        self.hidden_size = params.get('dqn_hidden_size', 60)
        self.gamma = params.get('gamma', 0.9)
        self.predict_mode = params.get('predict_mode', False)
        self.warm_start = params.get('warm_start', 0)
        
        self.max_turn = params['max_turn'] + 4
        self.state_dimension = 2 * self.act_cardinality + 7 * self.slot_cardinality + 3 + self.max_turn

        self.dqn = NoisyModel(self.state_dimension,  self.num_actions,self.hidden_size)
        self.target_dqn = NoisyModel(self.state_dimension,self.num_actions,self.hidden_size)
        self.target_dqn.load_state_dict(self.dqn.state_dict())

        self.target_dqn.eval()

        self.dqn_optim = optim.Adam(self.dqn.parameters(), lr=1e-3)

        self.loss_fn = nn.MSELoss()
        
        self.cur_bellman_err = 0
                
        # Prediction Mode: load trained DQN model
        if params['trained_model_path'] != None:
            self.dqn.model = copy.deepcopy(self.load_trained_DQN(params['trained_model_path']))
            self.clone_dqn = copy.deepcopy(self.dqn)
            self.predict_mode = True
            self.warm_start = 2
            
            
    def initialize_episode(self):
        """ Initialize a new episode. This function is called every time a new episode is run. """
        
        self.current_slot_id = 0
        self.phase = 0
        
        self.current_request_slot_id = 0
        self.current_inform_slot_id = 0
        
        #self.request_set = dialog_config.movie_request_slots #['moviename', 'starttime', 'city', 'date', 'theater', 'numberofpeople']
        
    def initialize_config(self, req_set, inf_set):
        """ Initialize request_set and inform_set """
        
        self.request_set = req_set
        self.inform_set = inf_set
        self.current_request_slot_id = 0
        self.current_inform_slot_id = 0
        
    def state_to_action(self, state):
        """ DQN: Input state, output action """
        
        self.representation = self.prepare_state_representation(state)
        self.action = self.run_policy(self.representation)
        act_slot_response = copy.deepcopy(self.feasible_actions[self.action])
        return {'act_slot_response': act_slot_response, 'act_slot_value_response': None}
        
    
    def prepare_state_representation(self, state):
        """ Create the representation for each state """
        
        user_action = state['user_action']
        current_slots = state['current_slots']
        kb_results_dict = state['kb_results_dict']
        agent_last = state['agent_action']
        
        ########################################################################
        #   Create one-hot of acts to represent the current user action
        ########################################################################
        user_act_rep =  np.zeros((1, self.act_cardinality))
        user_act_rep[0,self.act_set[user_action['diaact']]] = 1.0

        ########################################################################
        #     Create bag of inform slots representation to represent the current user action
        ########################################################################
        user_inform_slots_rep = np.zeros((1, self.slot_cardinality))
        for slot in user_action['inform_slots'].keys():
            user_inform_slots_rep[0,self.slot_set[slot]] = 1.0

        ########################################################################
        #   Create bag of request slots representation to represent the current user action
        ########################################################################
        user_request_slots_rep = np.zeros((1, self.slot_cardinality))
        for slot in user_action['request_slots'].keys():
            user_request_slots_rep[0, self.slot_set[slot]] = 1.0

        ########################################################################
        #   Creat bag of filled_in slots based on the current_slots
        ########################################################################
        current_slots_rep = np.zeros((1, self.slot_cardinality))
        for slot in current_slots['inform_slots']:
            current_slots_rep[0, self.slot_set[slot]] = 1.0

        ########################################################################
        #   Encode last agent act
        ########################################################################
        agent_act_rep = np.zeros((1,self.act_cardinality))
        if agent_last:
            agent_act_rep[0, self.act_set[agent_last['diaact']]] = 1.0

        ########################################################################
        #   Encode last agent inform slots
        ########################################################################
        agent_inform_slots_rep = np.zeros((1, self.slot_cardinality))
        if agent_last:
            for slot in agent_last['inform_slots'].keys():
                agent_inform_slots_rep[0,self.slot_set[slot]] = 1.0

        ########################################################################
        #   Encode last agent request slots
        ########################################################################
        agent_request_slots_rep = np.zeros((1, self.slot_cardinality))
        if agent_last:
            for slot in agent_last['request_slots'].keys():
                agent_request_slots_rep[0,self.slot_set[slot]] = 1.0
        
        turn_rep = np.zeros((1,1)) + state['turn'] / 10.

        ########################################################################
        #  One-hot representation of the turn count?
        ########################################################################
        turn_onehot_rep = np.zeros((1, self.max_turn))
        turn_onehot_rep[0, state['turn']] = 1.0

        ########################################################################
        #   Representation of KB results (scaled counts)
        ########################################################################
        kb_count_rep = np.zeros((1, self.slot_cardinality + 1)) + kb_results_dict['matching_all_constraints'] / 100.
        for slot in kb_results_dict:
            if slot in self.slot_set:
                kb_count_rep[0, self.slot_set[slot]] = kb_results_dict[slot] / 100.

        ########################################################################
        #   Representation of KB results (binary)
        ########################################################################
        kb_binary_rep = np.zeros((1, self.slot_cardinality + 1)) + np.sum( kb_results_dict['matching_all_constraints'] > 0.)
        for slot in kb_results_dict:
            if slot in self.slot_set:
                kb_binary_rep[0, self.slot_set[slot]] = np.sum( kb_results_dict[slot] > 0.)

        self.final_representation = np.hstack([user_act_rep, user_inform_slots_rep, user_request_slots_rep, agent_act_rep, agent_inform_slots_rep, agent_request_slots_rep, current_slots_rep, turn_rep, turn_onehot_rep, kb_binary_rep, kb_count_rep])

        return self.final_representation
      
    def run_policy(self, representation):
        """ epsilon-greedy policy """
        
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        else:
            if self.warm_start == 1:   #热启动
                if len(self.experience_replay_pool) > self.experience_replay_pool_size:
                    self.warm_start = 2
                return self.rule_request_inform_policy()
                # return self.rule_policy()
            else:
                action = self.DQN_policy(representation.squeeze()).item()
                return action
    
    def rule_policy(self):
        """ Rule Policy """
        
        if self.current_slot_id < len(self.request_set):
            slot = self.request_set[self.current_slot_id]
            self.current_slot_id += 1

            act_slot_response = {}
            act_slot_response['diaact'] = "request"
            act_slot_response['inform_slots'] = {}
            act_slot_response['request_slots'] = {slot: "UNK"}
        elif self.phase == 0:
            act_slot_response = {'diaact': "inform", 'inform_slots': {'taskcomplete': "PLACEHOLDER"}, 'request_slots': {} }
            self.phase += 1
        elif self.phase == 1:
            act_slot_response = {'diaact': "thanks", 'inform_slots': {}, 'request_slots': {} }
                
        return self.action_index(act_slot_response)

    def DQN_policy(self, state_representation):
        """ Return action from DQN"""

        with torch.no_grad():
            action = self.dqn.select_action(torch.FloatTensor(state_representation),is_train=self.predict_mode)
        return action
    
    def rule_request_inform_policy(self):
        """ Rule Request and Inform Policy """
        
        if self.current_request_slot_id < len(self.request_set):
            slot = self.request_set[self.current_request_slot_id]
            self.current_request_slot_id += 1

            act_slot_response = {} 
            act_slot_response['diaact'] = "request"
            act_slot_response['inform_slots'] = {}
            act_slot_response['request_slots'] = {slot: "UNK"}
        elif self.current_inform_slot_id < len(self.inform_set):
            slot = self.inform_set[self.current_inform_slot_id]
            self.current_inform_slot_id += 1

            act_slot_response = {}
            act_slot_response['diaact'] = "inform"
            act_slot_response['inform_slots'] = {slot: 'PLACEHOLDER'}
            act_slot_response['request_slots'] = {}
        elif self.phase == 0:
            act_slot_response = {'diaact': "inform", 'inform_slots': {'taskcomplete': "PLACEHOLDER"}, 'request_slots': {}}
            self.phase += 1
        elif self.phase == 1:
            act_slot_response = {'diaact': "thanks", 'inform_slots': {}, 'request_slots': {}}
        #else:
        #    raise Exception("THIS SHOULD NOT BE POSSIBLE (AGENT CALLED IN UNANTICIPATED WAY)")
        
        return self.action_index(act_slot_response) #{'act_slot_response': act_slot_response, 'act_slot_value_response': None}
        
    
    def action_index(self, act_slot_response):
        """ Return the index of action """
        
        for (i, action) in enumerate(self.feasible_actions):
            if act_slot_response == action:
                return i
        print act_slot_response
        raise Exception("action index not found")
        return None
    
    
    def register_experience_replay_tuple(self, s_t, a_t, reward, s_tplus1, episode_over):
        """ Register 将经验存放缓存池 feedback from the environment, to be stored as future training data """
        """ 将经验存放缓存池 """
        
        state_t_rep = self.prepare_state_representation(s_t)
        action_t = self.action
        reward_t = reward
        state_tplus1_rep = self.prepare_state_representation(s_tplus1)
        training_example = (state_t_rep, action_t, reward_t, state_tplus1_rep, episode_over)
        
        if self.predict_mode == False: # Training Mode
            if self.warm_start == 1:
                self.experience_replay_pool.append(training_example)
        else: # Prediction Mode
            self.experience_replay_pool.append(training_example)

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

    def train(self, epoch, batch_size=1, num_batches=100):
        """ Train DQN with experience buffer that comes from both user and world model interaction."""
        print "当前经验池数据量：", len(self.experience_replay_pool)
        self.cur_bellman_err = 0.
        self.running_expereince_pool = list(self.experience_replay_pool)
        for iter_batch in range(num_batches):
            for iter in tqdm(range(len(self.running_expereince_pool) / (batch_size))):
                self.dqn_optim.zero_grad()
                batch = self.sample_from_buffer(batch_size)

                state_value = self.dqn(torch.FloatTensor(batch.state)).gather(1, torch.tensor(batch.action))
                next_state_value, _ = self.target_dqn(torch.FloatTensor(batch.next_state)).max(1)
                next_state_value = next_state_value.unsqueeze(1)
                term = np.asarray(batch.term, dtype=np.float32)
                expected_value = torch.FloatTensor(batch.reward) + self.gamma * next_state_value * (
                        1 - torch.FloatTensor(term))
                loss = F.mse_loss(state_value, expected_value)
                loss.backward()
                self.dqn_optim.step()

                self.dqn.reset_noise()
                self.target_dqn.reset_noise()

                self.cur_bellman_err += loss.item()

            print ('<<dialog policy noise_dqn>> epoch {}, total_loss {}'
                   .format(epoch,
                           float(self.cur_bellman_err) / (len(self.experience_replay_pool) / (float(batch_size)))))

        # # update the epsilon value
        # self.dqn.update_epsilon(epoch)
        # update the target network
        self.target_dqn.load_state_dict(self.dqn.state_dict())

    ################################################################################
    #    Debug Functions
    ################################################################################
    def save_experience_replay_to_file(self, path):
        """ Save the experience replay pool to a file """
        
        try:
            pickle.dump(self.experience_replay_pool, open(path, "wb"))
            print 'saved model in %s' % (path, )
        except Exception, e:
            print 'Error: Writing model fails: %s' % (path, )
            print e         
    
    def load_experience_replay_from_file(self, path):
        """ Load the experience replay pool from a file"""
        
        self.experience_replay_pool = pickle.load(open(path, 'rb'))
    
             
    def load_trained_DQN(self, path):
        """ Load the trained DQN from a file """
        
        trained_file = pickle.load(open(path, 'rb'))
        model = trained_file['model']
        
        print "trained DQN Parameters:", json.dumps(trained_file['params'], indent=2)
        return model

    def save(self, domain, epoch,success_rate=None):
        directory = os.path.join(os.path.dirname(self.save_dir), "save/" + domain)
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(self.dqn.state_dict(), directory + '/' + str(epoch) + '_noise_dqn.pol.mdl')
        if success_rate != None:
            torch.save(self.dqn.state_dict(), directory + '/' + 'best_' + 'succ_' + str(success_rate) + '_noise_dqn.pol.mdl')


        else:
            torch.save(self.dqn.state_dict(), directory + '/' + str(epoch) + '_noise_dqn.pol.mdl')


        print ('<<dialog policy>> epoch {}: saved network to mdl'.format(epoch))

    def load(self, filename):
        dqn_mdl_candidates = [
            filename + '_dqn.pol.mdl',
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"+self.save_dir + '/' + filename + '_dqn.pol.mdl'),
        ]
        for dqn_mdl in dqn_mdl_candidates:
            if os.path.exists(dqn_mdl):
                print("加载模型：", dqn_mdl)
                self.dqn.load_state_dict(torch.load(dqn_mdl, map_location=DEVICE))
                self.target_dqn.load_state_dict(torch.load(dqn_mdl, map_location=DEVICE))
                logging.info('<<dialog policy>> loaded checkpoint from file: {}'.format(dqn_mdl))
                break


