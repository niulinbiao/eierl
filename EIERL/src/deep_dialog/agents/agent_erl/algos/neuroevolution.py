# coding=utf-8
import random
import numpy as np
import math
import math,sys,os
from deep_dialog.agents.agent_erl.core import utils
# sys.path.append('/home/py36/ConvLab-2-master/ConvLab-2-master/convlab2/policy/erl_dqn/core')
class SSNE:

	def __init__(self, args):
		self.gen = 0
		self.args = args
		self.population_size = args['pop_size']
		# self.writer = args.writer

		self.elite_fraction = 0.1
		self.crossover_prob = 0.15
		self.mutation_prob = 0.90
		self.weight_magnitude_limit = 10000000
		# self.elite_fraction = 0.1
		# self.crossover_prob = 0.05
		# self.mutation_prob = 0.5
		# self.weight_magnitude_limit = 1000
		#RL TRACKERS
		self.rl_policy = None
		self.selection_stats = {'elite': 0, 'selected': 0, 'discarded': 0, 'total': 0}

	def selection_tournament(self, index_rank, num_offsprings, tournament_size):
		"""Conduct tournament selection

			Parameters:
				  index_rank (list): Ranking encoded as net_indexes
				  num_offsprings (int): Number of offsprings to generate
				  tournament_size (int): Size of tournament

			Returns:
				  offsprings (list): List of offsprings returned as a list of net indices

		"""
		# 在竞标赛中，从排名列表中随机选择tournament_size个个体进行比较，
		# 选择其中排名最低的个体作为胜者。
		# 然后，将胜者的索引添加到后代列表中，重复这个过程num_offsprings次。
		total_choices = len(index_rank)
		offsprings = []
		for i in range(num_offsprings):
			winner = np.min(np.random.randint(total_choices, size=tournament_size))
			offsprings.append(index_rank[winner])

		# 把里面重复的去掉，只保留一个
		offsprings = list(set(offsprings))  # Find unique offsprings
		#  Number of offsprings should be even
		if len(offsprings) % 2 != 0:  # Number of offsprings should be even
			offsprings.append(index_rank[winner])
		return offsprings

	def list_argsort(self, seq):
		"""Sort the list

			Parameters:
				  seq (list): list

			Returns:
				  sorted list

		"""
		# 返回排序后的索引
		return sorted(range(len(seq)), key=seq.__getitem__)

	def regularize_weight(self, weight, mag):
		"""Clamps on the weight magnitude (reguralizer)
		将权重值限制在[-mag, mag]的范围内
		确保权重值不会超过指定的范围，以防止权重值过大或过小对模型造成不良影响

			Parameters:
				  weight (float): weight
				  mag (float): max/min value for weight

			Returns:
				  weight (float): clamped weight

		"""
		if weight > mag: weight = mag
		if weight < -mag: weight = -mag
		return weight

	def crossover_inplace(self, gene1, gene2):
		"""Conduct one point crossover in place

			Parameters:
				  gene1 (object): A pytorch model
				  gene2 (object): A pytorch model

			Returns:
				None

		"""

		keys1 = list(gene1.state_dict())
		keys2 = list(gene2.state_dict())

		for key in keys1:
			if key not in keys2: continue

			# References to the variable tensors
			W1 = gene1.state_dict()[key]
			W2 = gene2.state_dict()[key]

			if len(W1.shape) == 2:  # Weights no bias
				num_variables = W1.shape[0]
				# Crossover opertation [Indexed by row]
				try:
					num_cross_overs = random.randint(0, int(num_variables * 0.3))  # Number of Cross overs
				except:
					num_cross_overs = 1
				for i in range(num_cross_overs):
					receiver_choice = random.random()  # Choose which gene to receive the perturbation
					if receiver_choice < 0.5:
						ind_cr = random.randint(0, W1.shape[0] - 1)  #
						W1[ind_cr, :] = W2[ind_cr, :]
					else:
						ind_cr = random.randint(0, W1.shape[0] - 1)  #
						W2[ind_cr, :] = W1[ind_cr, :]

			elif len(W1.shape) == 1:  # Bias or LayerNorm
				if random.random() < 0.8: continue  # Crossover here with low frequency
				num_variables = W1.shape[0]
				# Crossover opertation [Indexed by row]
				# num_cross_overs = random.randint(0, int(num_variables * 0.05))  # Crossover number
				for i in range(1):
					receiver_choice = random.random()  # Choose which gene to receive the perturbation
					if receiver_choice < 0.5:
						ind_cr = random.randint(0, W1.shape[0] - 1)  #
						W1[ind_cr] = W2[ind_cr]
					else:
						ind_cr = random.randint(0, W1.shape[0] - 1)  #
						W2[ind_cr] = W1[ind_cr]

	def mutate_inplace(self, gene):
		"""就地进行突变

			Parameters:
				  gene (object): A pytorch model

			Returns:
				None

		"""
		mut_strength = 0.1
		num_mutation_frac = 0.05
		super_mut_strength = 10
		super_mut_prob = 0.05
		reset_prob = super_mut_prob + 0.02

		num_params = len(list(gene.parameters()))
		ssne_probabilities = np.random.uniform(0, 1, num_params) * 2

		for i, param in enumerate(gene.parameters()):  # Mutate each param

			# References to the variable keys
			W = param.data
			if len(W.shape) == 2:  # Weights, no bias

				num_weights = W.shape[0] * W.shape[1]
				ssne_prob = ssne_probabilities[i]

				if random.random() < ssne_prob:
					num_mutations = random.randint(0,
												   int(math.ceil(
													   num_mutation_frac * num_weights)))  # Number of mutation instances
					for _ in range(num_mutations):
						ind_dim1 = random.randint(0, W.shape[0] - 1)
						ind_dim2 = random.randint(0, W.shape[-1] - 1)
						random_num = random.random()

						if random_num < super_mut_prob:  # Super Mutation probability
							W[ind_dim1, ind_dim2] += random.gauss(0, super_mut_strength * W[ind_dim1, ind_dim2])
						elif random_num < reset_prob:  # Reset probability
							W[ind_dim1, ind_dim2] = random.gauss(0, 0.1)
						else:  # mutauion even normal
							W[ind_dim1, ind_dim2] += random.gauss(0, mut_strength * W[ind_dim1, ind_dim2])

						# Regularization hard limit
						W[ind_dim1, ind_dim2] = self.regularize_weight(W[ind_dim1, ind_dim2],
																	   self.weight_magnitude_limit)

			elif len(W.shape) == 1:  # Bias or layernorm
				num_weights = W.shape[0]
				ssne_prob = ssne_probabilities[i] * 0.04  # Low probability of mutation here

				if random.random() < ssne_prob:
					num_mutations = random.randint(0,
												   int(math.ceil(
													   num_mutation_frac * num_weights)))  # Number of mutation instances
					for _ in range(num_mutations):
						ind_dim = random.randint(0, W.shape[0] - 1)
						random_num = random.random()

						if random_num < super_mut_prob:  # Super Mutation probability
							W[ind_dim] += random.gauss(0, super_mut_strength * W[ind_dim])
						elif random_num < reset_prob:  # Reset probability
							W[ind_dim] = random.gauss(0, 1)
						else:  # mutauion even normal
							W[ind_dim] += random.gauss(0, mut_strength * W[ind_dim])

						# Regularization hard limit
						W[ind_dim] = self.regularize_weight(W[ind_dim], self.weight_magnitude_limit)

	def reset_genome(self, gene):
		"""Reset a model's weights in place

			Parameters:
				  gene (object): A pytorch model

			Returns:
				None

		"""
		for param in (gene.parameters()):
			param.data.copy_(param.data)

	def epoch(self, gen, pop, fitness_evals, migration):

		self.gen += 1
		# 精英个体数量
		num_elitists = int(self.elite_fraction * len(fitness_evals))
		if num_elitists < 2: num_elitists = 1

		# Entire epoch is handled with indices; Index rank nets by fitness evaluation (0 is the best after reversing)
		# 越靠前适应度越低
		index_rank = self.list_argsort(fitness_evals)
		# 越靠后适应度越低
		index_rank.reverse()
		# 获取精英个体的索引
		elitist_index = index_rank[:num_elitists]  # Elitist indexes safeguard

		# Selection step
		# 使用锦标赛选择算法从排序后的基因中选择出一组后代个体，用于进行后续的交叉和突变操作
		offsprings = self.selection_tournament(index_rank,
											   num_offsprings=len(index_rank) - len(elitist_index) - len(migration),
											   tournament_size=3)

		# Figure out unselected candidates
		# 选出那些既不在offsprings中也不在elitist_index的个体
		unselects = []

		new_elitists = []
		for net_i in range(len(pop)):
			if net_i in offsprings or net_i in elitist_index:
				continue
			else:
				unselects.append(net_i)

		# 打乱顺序，增加随机性
		random.shuffle(unselects)

		# Migration Tracker
		if self.rl_policy != None:  # RL Transfer happened
			self.selection_stats['total'] += 1.0
			if self.rl_policy in elitist_index:
				self.selection_stats['elite'] += 1.0
			elif self.rl_policy in offsprings:
				self.selection_stats['selected'] += 1.0
			elif self.rl_policy in unselects:
				self.selection_stats['discarded'] += 1.0
			self.rl_policy = None
			# self.writer.add_scalar('elite_rate', self.selection_stats['elite'] / self.selection_stats['total'], gen)
			# self.writer.add_scalar('selection_rate',
			# 					   (self.selection_stats['elite'] + self.selection_stats['selected']) /
			# 					   self.selection_stats['total'], gen)
			# self.writer.add_scalar('discard_rate', self.selection_stats['discarded'] / self.selection_stats['total'],
			# 					   gen)

		# Inheritance step (sync learners to population) --> Migration
		# 将迁移策略的参数值复制到种群中的目标个体中，并记录下被替换的个体的索引
		if len(pop) > 1:
			for policy in migration:
				replacee = unselects.pop(0)
				print "原始个体注入！"
				utils.hard_update(target=pop[replacee], source=policy)
				self.rl_policy = replacee

		# Elitism step, assigning elite candidates to some unselects
		if len(pop) > 1:
			for i in elitist_index:
				try:
					replacee = unselects.pop(0)
				except:
					replacee = offsprings.pop(0)
				new_elitists.append(replacee)
				utils.hard_update(target=pop[replacee], source=pop[i])

		# 对未被选择的个体进行交叉操作
		if len(unselects) % 2 != 0:  # Number of unselects left should be even
			unselects.append(unselects[random.randint(0, len(unselects) - 1)])
		for i, j in zip(unselects[0::2], unselects[1::2]):
			print "对未被选择的个体进行交叉操作"
			off_i = random.choice(new_elitists);
			off_j = random.choice(offsprings)
			utils.hard_update(target=pop[i], source=pop[off_i])
			utils.hard_update(target=pop[j], source=pop[off_j])
			self.crossover_inplace(pop[i], pop[j])

		# 对被选择的个体进行交叉操作
		for i, j in zip(offsprings[0::2], offsprings[1::2]):
			print "对被选择的个体进行交叉操作"
			if random.random() < self.crossover_prob:
				self.crossover_inplace(pop[i], pop[j])

		# Mutate all genes in the population except the new elitists
		# 对除了新精英个体的个体进行基因突变
		for net_i in range(len(pop)):
			if net_i not in new_elitists:  # Spare the new elitists
				print "对除了新精英个体的个体进行基因突变"
				if random.random() < self.mutation_prob:
					self.mutate_inplace(pop[net_i])












