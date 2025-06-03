# coding=utf-8
import copy

import numpy as np, os, time, random, json, sys
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(root_dir)
from deep_dialog.agents.agent_erl.algos.neuroevolution import SSNE
print root_dir
from deep_dialog.agents import Agent
from torch.multiprocessing import Process, Pipe, Manager
from collections import deque
import torch
from  deep_dialog import dialog_config

from deep_dialog.qlearning import EpsilonGreedyPolicy
from deep_dialog.agents.agent_erl.core import utils

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device("cpu")

class ERL(Agent):

	def __init__(self, movie_dict=None, act_set=None, slot_set=None, params=None):
		Agent.__init__(self, movie_dict, act_set, slot_set, params)
		self.rollout_size = params['rollout_size']
		self.pop_size = params['pop_size']
		self.best_score = -float('inf')


		self.movie_dict = movie_dict
		self.act_set = act_set
		self.slot_set = slot_set
		self.act_cardinality = len(act_set.keys())
		self.slot_cardinality = len(slot_set.keys())

		self.training_iter = 10
		self.training_batch_iter = 1
		self.save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../save")
		print "模型保存目录：",self.save_dir

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

		# DQN参数
		params["state_dim"] = self.state_dimension
		params["hidden_size"] = self.hidden_size
		params["da_dim"] = self.num_actions
		# 进化算法参数
		params["pop_size"] = self.pop_size

		#######################  Actor, Critic and ValueFunction Model Constructor ######################

		# PG Learner
		# from convlab2.policy.erl_dqn.algos.dqn import DQN
		from deep_dialog.agents.agent_erl.algos.dqn import DQN
		self.learner = DQN(params)

		self.cur_net =None

		#Evolution
		self.evolver = SSNE(params)
		#Save best policy
		self.best_policy = EpsilonGreedyPolicy(self.state_dimension, self.hidden_size, self.num_actions,tag="最优模型")



		# Initialize population
		# 创建了一个可在多个进程之间共享的Manager.list对象，用于存储种群
		self.population = []
		# 初始化种群   我们的种群就是一个个的神经网络模型
		for i in range(self.pop_size):
			self.population.append(EpsilonGreedyPolicy(self.state_dimension, self.hidden_size, self.num_actions,tag="进化个体"+str(i)))

		# # Initialize Rollout Bucket
		# 创建了一个回放桶对象rollout_bucket，用于存储当前策略下的轨迹数据。
		self.rollout_bucket = []
		for i in range(self.rollout_size):
			self.rollout_bucket.append(EpsilonGreedyPolicy(self.state_dimension, self.hidden_size, self.num_actions,tag="原始个体"+str(i)))

		#Trackers
		self.best_score = -float('inf');  # 存储最佳分数，初始值设置为负无穷大
		self.gen_frames = 0; # 追踪已生成的帧数
		self.total_frames = 0;# 追踪总共生成的帧数
		self.test_score = None;# 存储测试分数
		self.test_std = None# 存储测试标准差

	def initialize_episode(self):
		""" Initialize a new episode. This function is called every time a new episode is run. """

		self.current_slot_id = 0
		self.phase = 0

		self.current_request_slot_id = 0
		self.current_inform_slot_id = 0

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
		user_act_rep = np.zeros((1, self.act_cardinality))
		user_act_rep[0, self.act_set[user_action['diaact']]] = 1.0

		########################################################################
		#     Create bag of inform slots representation to represent the current user action
		########################################################################
		user_inform_slots_rep = np.zeros((1, self.slot_cardinality))
		for slot in user_action['inform_slots'].keys():
			user_inform_slots_rep[0, self.slot_set[slot]] = 1.0

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
		agent_act_rep = np.zeros((1, self.act_cardinality))
		if agent_last:
			agent_act_rep[0, self.act_set[agent_last['diaact']]] = 1.0

		########################################################################
		#   Encode last agent inform slots
		########################################################################
		agent_inform_slots_rep = np.zeros((1, self.slot_cardinality))
		if agent_last:
			for slot in agent_last['inform_slots'].keys():
				agent_inform_slots_rep[0, self.slot_set[slot]] = 1.0

		########################################################################
		#   Encode last agent request slots
		########################################################################
		agent_request_slots_rep = np.zeros((1, self.slot_cardinality))
		if agent_last:
			for slot in agent_last['request_slots'].keys():
				agent_request_slots_rep[0, self.slot_set[slot]] = 1.0

		turn_rep = np.zeros((1, 1)) + state['turn'] / 10.

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
		kb_binary_rep = np.zeros((1, self.slot_cardinality + 1)) + np.sum(
			kb_results_dict['matching_all_constraints'] > 0.)
		for slot in kb_results_dict:
			if slot in self.slot_set:
				kb_binary_rep[0, self.slot_set[slot]] = np.sum(kb_results_dict[slot] > 0.)

		self.final_representation = np.hstack(
			[user_act_rep, user_inform_slots_rep, user_request_slots_rep, agent_act_rep, agent_inform_slots_rep,
			 agent_request_slots_rep, current_slots_rep, turn_rep, turn_onehot_rep, kb_binary_rep, kb_count_rep])

		return self.final_representation

	def run_policy(self, representation):
		""" epsilon-greedy policy """

		if random.random() < self.epsilon:
			return random.randint(0, self.num_actions - 1)
		else:
			if self.warm_start == 1:  # 热启动
				if len(self.experience_replay_pool) > self.experience_replay_pool_size:
					self.warm_start = 2
				return self.rule_request_inform_policy()
				# return self.rule_policy()
			else:
				action = self.DQN_policy(representation.squeeze()).item()
				return action

	def rule_policy(self):
		""" Rule Policy """

		act_slot_response = {}

		if self.current_slot_id < len(self.request_set):
			slot = self.request_set[self.current_slot_id]
			self.current_slot_id += 1

			act_slot_response = {}
			act_slot_response['diaact'] = "request"
			act_slot_response['inform_slots'] = {}
			act_slot_response['request_slots'] = {slot: "UNK"}
		elif self.phase == 0:
			act_slot_response = {'diaact': "inform", 'inform_slots': {'taskcomplete': "PLACEHOLDER"},
								 'request_slots': {}}
			self.phase += 1
		elif self.phase == 1:
			act_slot_response = {'diaact': "thanks", 'inform_slots': {}, 'request_slots': {}}

		return self.action_index(act_slot_response)

	def DQN_policy(self, state_representation):
		""" Return action from DQN"""

		with torch.no_grad():
			action = self.cur_net.clean_action(torch.FloatTensor(state_representation))
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
			act_slot_response = {'diaact': "inform", 'inform_slots': {'taskcomplete': "PLACEHOLDER"},
								 'request_slots': {}}
			self.phase += 1
		elif self.phase == 1:
			act_slot_response = {'diaact': "thanks", 'inform_slots': {}, 'request_slots': {}}
		# else:
		#    raise Exception("THIS SHOULD NOT BE POSSIBLE (AGENT CALLED IN UNANTICIPATED WAY)")

		return self.action_index(
			act_slot_response)  # {'act_slot_response': act_slot_response, 'act_slot_value_response': None}

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

		if self.predict_mode == False:  # Training Mode
			if self.warm_start == 1:
				self.experience_replay_pool.append(training_example)
		else:  # Prediction Mode
			self.experience_replay_pool.append(training_example)


	def train(self,episode):
		self.learner.update(episode,self.experience_replay_pool)
	def evolution(self,episode,all_fitness,rollout_fitness):
		print "进化种群适应度:", all_fitness
		print "原始种群适应度:", rollout_fitness
		print"本轮最大值适应度：", max(max(all_fitness), max(rollout_fitness)), "全局最大值适应度：", self.best_score
		if (max(all_fitness) > self.best_score or max(rollout_fitness) > self.best_score):
			if (max(all_fitness) >= max(rollout_fitness)):
				champ_index_pop = all_fitness.index(max(all_fitness))
				self.best_score = max(all_fitness)
				print"选取进化种群个体", champ_index_pop, "注入进化种群！"
				for id, actor in enumerate(self.population):
					self.population[id].load_state_dict(self.population[champ_index_pop].state_dict())
			else:
				champ_index_rollout = rollout_fitness.index(max(rollout_fitness))
				self.best_score = max(rollout_fitness)
				print"选取普通种群个体", champ_index_rollout, "注入进化种群！"
				for id, actor in enumerate(self.population):
					self.population[id].load_state_dict(self.rollout_bucket[champ_index_rollout].state_dict())
		else:
			print"执行进化算法！"
			self.evolver.epoch(episode, self.population, all_fitness, self.rollout_bucket)

	"""
	针对EA的EII
	"""
	def eii(self,episode,all_fitness):
		print "进化种群适应度:", all_fitness
		print"本轮最大值适应度：", max(all_fitness), "全局最大值适应度：", self.best_score
		if (max(all_fitness) > self.best_score):

			champ_index_pop = all_fitness.index(max(all_fitness))
			self.best_score = max(all_fitness)
			print"选取进化种群个体", champ_index_pop, "注入进化种群！"
			for id, actor in enumerate(self.population):
				self.population[id].load_state_dict(self.population[champ_index_pop].state_dict())

		else:
			print"执行进化算法！"
			self.evolver.epoch(episode, self.population, all_fitness, self.rollout_bucket)


	def save(self, domain, epoch,success_rate=None):
		directory = os.path.join(os.path.dirname(self.save_dir), "save/" + domain)
		if self.pop_size>0:
			if self.rollout_size >0:
				directory = os.path.join(directory, "eierl")
			else:
				directory = os.path.join(directory, "ea")
		else:
			directory = os.path.join(directory, "dqn")
		if not os.path.exists(directory):
			os.makedirs(directory)
		if self.pop_size>0:
			if self.rollout_size >0:
				if success_rate!=None:
					torch.save(self.learner.dqn.state_dict(), directory + '/' + 'best_'+'succ_'+str(success_rate)+'eierl.pol.mdl')
				else:
					torch.save(self.learner.dqn.state_dict(), directory + '/' + str(epoch) + '_eierl.pol.mdl')
			else:
				torch.save(self.best_policy.state_dict(), directory + '/' + str(epoch) + '_ea.pol.mdl')
		else:
			if success_rate != None:
				torch.save(self.learner.dqn.state_dict(),
						   directory + '/' + 'best_' + 'succ_' + str(success_rate) + 'dqn.pol.mdl')
			else:
				torch.save(self.learner.dqn.state_dict(), directory + '/' + str(epoch) + '_dqn.pol.mdl')
		print ('<<dialog policy>> epoch {}: saved network to mdl'.format(epoch))

	def load(self, filename):
		erl_mdl_candidates = [
			filename + '_erl_dqn.pol.mdl',
			os.path.join(os.path.dirname(os.path.abspath(__file__)), self.save_dir+'/'+filename + '_erl_dqn.pol.mdl'),
		]

		for erl_mdl in erl_mdl_candidates:
			if os.path.exists(erl_mdl):
				print("加载已训练模型：", erl_mdl)
				self.learner.dqn.load_state_dict(torch.load(erl_mdl, map_location=DEVICE))
				self.learner.target_dqn.load_state_dict(torch.load(erl_mdl, map_location=DEVICE))
				print('<<dialog policy>> loaded checkpoint from file: {}'.format(erl_mdl))
				break



	# def train(self, frame_limit):
	# 	for gen in range(1, 1000000000):  # Infinite generations
	# 		# Train one iteration
	# 		self.forward_generation(gen)
	#
	# 	###Kill all processes
	# 	try:
	# 		for p in self.task_pipes: p[0].send('TERMINATE')
	# 		for p in self.evo_task_pipes: p[0].send('TERMINATE')
	# 	except:
	# 		None
	#
	#


