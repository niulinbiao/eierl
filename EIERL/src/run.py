# coding=utf-8

import argparse, json, copy, os
import cPickle as pickle
import random

import numpy
import pandas as pd

from deep_dialog.agents.agent_erl.core import utils
from deep_dialog.dialog_system import DialogManager, text_to_dict
from deep_dialog.agents import AgentCmd, InformAgent, RequestAllAgent, RandomAgent, EchoAgent, RequestBasicsAgent, AgentDQN, RequestInformSlotAgent
from deep_dialog.agents.agent_erl.algos.erl_trainer import ERL
from deep_dialog.agents.agent_icm.agent_icm import AgentICM
from deep_dialog.agents.agent_noise.agent_noise import AgentNoiseDQN
from deep_dialog.usersims import UserSimulator,   RuleSimulator, RuleRestaurantSimulator, RuleTaxiSimulator
from deep_dialog import dialog_config
from deep_dialog.dialog_config import *

from deep_dialog.nlu import nlu
from deep_dialog.nlg import nlg

seed = 5
numpy.random.seed(seed)
random.seed(seed)

import torch

""" 
Launch a dialog simulation per the command line arguments
This function instantiates a user_simulator, an agent, and a dialog system.
Next, it triggers the simulator to run for the specified number of episodes.
"""

""" load action """
def load_actions(sys_req_slots, sys_inf_slots):
    dialog_config.feasible_actions = [
        {'diaact':"confirm_question", 'inform_slots':{}, 'request_slots':{}},
        {'diaact':"confirm_answer", 'inform_slots':{}, 'request_slots':{}},
        {'diaact':"thanks", 'inform_slots':{}, 'request_slots':{}},
        {'diaact':"deny", 'inform_slots':{}, 'request_slots':{}},
    ]

    for slot in sys_inf_slots:
        dialog_config.feasible_actions.append({'diaact':'inform', 'inform_slots':{slot:"PLACEHOLDER"}, 'request_slots':{}})

    for slot in sys_req_slots:
        dialog_config.feasible_actions.append({'diaact':'request', 'inform_slots':{}, 'request_slots': {slot: "UNK"}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--dict_path', dest='dict_path', type=str, default='./deep_dialog/data_restaurant/slot_dict.v2.p', help='path to the .json dictionary file')
    parser.add_argument('--kb_path', dest='kb_path', type=str, default='./deep_dialog/data_restaurant/restaurant.kb.1k.v1.p', help='path to the movie kb .json file')
    parser.add_argument('--act_set', dest='act_set', type=str, default='./deep_dialog/data_restaurant/dia_acts.txt', help='path to dia act set; none for loading from labeled file')
    parser.add_argument('--slot_set', dest='slot_set', type=str, default='./deep_dialog/data_restaurant/restaurant_slots.txt', help='path to slot set; none for loading from labeled file')
    parser.add_argument('--goal_file_path', dest='goal_file_path', type=str, default='./deep_dialog/data_restaurant/user_goals_first.v1.p', help='a list of user goals')
    parser.add_argument('--diaact_nl_pairs', dest='diaact_nl_pairs', type=str, default='./deep_dialog/data_restaurant/sim_dia_act_nl_pairs.v2.json', help='path to the pre-defined dia_act&NL pairs')

    parser.add_argument('--max_turn', dest='max_turn', default=20, type=int, help='maximum length of each dialog (default=20, 0=no maximum length)')
    parser.add_argument('--episodes', dest='episodes', default=1, type=int, help='Total number of episodes to run (default=1)')
    parser.add_argument('--slot_err_prob', dest='slot_err_prob', default=0.05, type=float, help='the slot err probability')
    parser.add_argument('--slot_err_mode', dest='slot_err_mode', default=0, type=int, help='slot_err_mode: 0 for slot_val only; 1 for three errs')
    parser.add_argument('--intent_err_prob', dest='intent_err_prob', default=0.05, type=float, help='the intent err probability')
    
    parser.add_argument('--agt', dest='agt', default=0, type=int, help='Select an agent: 0 for a command line input, 1-6 for rule based agents')
    parser.add_argument('--usr', dest='usr', default=1, type=int, help='Select a user simulator. 0 is a Frozen user simulator.')
    
    parser.add_argument('--epsilon', dest='epsilon', type=float, default=0, help='Epsilon to determine stochasticity of epsilon-greedy agent policies')

    parser.add_argument('--rollout_size', dest='rollout_size', type=float, default=1, help='number of individuals in the DQN population')
    parser.add_argument('--pop_size', dest='pop_size', type=float, default=3, help='number of individuals in evolved populations')
    
    # load NLG & NLU model
    parser.add_argument('--nlg_model_path', dest='nlg_model_path', type=str, default='./deep_dialog/models/nlg/restaurant/lstm_tanh_[1532068150.19]_98_99_294_0.983.p', help='path to model file')
    parser.add_argument('--nlu_model_path', dest='nlu_model_path', type=str, default='./deep_dialog/models/nlu/restaurant/lstm_[1532107808.26]_68_74_20_0.997.p', help='path to the NLU model file')
    
    parser.add_argument('--act_level', dest='act_level', type=int, default=0, help='0 for dia_act level; 1 for NL level')
    parser.add_argument('--run_mode', dest='run_mode', type=int, default=0, help='run_mode: 0 for default NL; 1 for dia_act; 2 for both')
    parser.add_argument('--auto_suggest', dest='auto_suggest', type=int, default=0, help='0 for no auto_suggest; 1 for auto_suggest')
    parser.add_argument('--cmd_input_mode', dest='cmd_input_mode', type=int, default=0, help='run_mode: 0 for NL; 1 for dia_act')
    
    # RL agent parameters
    parser.add_argument('--experience_replay_pool_size', dest='experience_replay_pool_size', type=int, default=1000, help='the size for experience replay')
    parser.add_argument('--dqn_hidden_size', dest='dqn_hidden_size', type=int, default=60, help='the hidden size for DQN')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--gamma', dest='gamma', type=float, default=0.9, help='gamma for DQN')
    parser.add_argument('--predict_mode', dest='predict_mode', type=bool, default=False, help='predict model for DQN')
    parser.add_argument('--simulation_epoch_size', dest='simulation_epoch_size', type=int, default=50, help='the size of validation set')
    parser.add_argument('--warm_start', dest='warm_start', type=int, default=1, help='0: no warm start; 1: warm start for training')
    parser.add_argument('--warm_start_epochs', dest='warm_start_epochs', type=int, default=100, help='the number of epochs for warm start')
    
    parser.add_argument('--trained_model_path', dest='trained_model_path', type=str, default=None, help='the path for trained model')
    parser.add_argument('-o', '--write_model_dir', dest='write_model_dir', type=str, default='./deep_dialog/save/', help='write model to disk')
    parser.add_argument('--save_check_point', dest='save_check_point', type=int, default=10, help='number of epochs for saving model')
     
    parser.add_argument('--success_rate_threshold', dest='success_rate_threshold', type=float, default=0.3, help='the threshold for success rate')
    
    parser.add_argument('--split_fold', dest='split_fold', default=5, type=int, help='the number of folders to split the user goal')
    parser.add_argument('--learning_phase', dest='learning_phase', default='all', type=str, help='train/test/all; default is all')

    parser.add_argument('--torch_seed', dest='torch_seed', type=int, default=100, help='random seed for troch')
    args = parser.parse_args()
    params = vars(args)

    print ('Dialog Parameters: ')
    print (json.dumps(params, indent=2))


max_turn = params['max_turn']
num_episodes = params['episodes']

agt = params['agt']
usr = params['usr']

dict_path = params['dict_path']
goal_file_path = params['goal_file_path']

# load the user goals from .p file
all_goal_set = pickle.load(open(goal_file_path, 'rb'))

# split goal set
split_fold = params.get('split_fold', 5)
goal_set = {'train':[], 'valid':[], 'test':[], 'all':[]}
for u_goal_id, u_goal in enumerate(all_goal_set):
    if u_goal_id % split_fold == 1: goal_set['test'].append(u_goal)
    else: goal_set['train'].append(u_goal)
    goal_set['all'].append(u_goal)
# end split goal set

kb_path = params['kb_path']
kb = pickle.load(open(kb_path, 'rb'))

act_set = text_to_dict(params['act_set'])
slot_set = text_to_dict(params['slot_set'])

################################################################################
# a movie dictionary for user simulator - slot:possible values
################################################################################
movie_dictionary = pickle.load(open(dict_path, 'rb'))

dialog_config.run_mode = params['run_mode']
dialog_config.auto_suggest = params['auto_suggest']

################################################################################
#   Parameters for Agents
################################################################################
agent_params = {}
agent_params['max_turn'] = max_turn
agent_params['epsilon'] = params['epsilon']
agent_params['rollout_size'] = params['rollout_size']
agent_params['pop_size'] = params['pop_size']

agent_params['agent_run_mode'] = params['run_mode']
agent_params['agent_act_level'] = params['act_level']

agent_params['experience_replay_pool_size'] = params['experience_replay_pool_size']
agent_params['dqn_hidden_size'] = params['dqn_hidden_size']
agent_params['batch_size'] = params['batch_size']
agent_params['gamma'] = params['gamma']
agent_params['predict_mode'] = params['predict_mode']
agent_params['trained_model_path'] = params['trained_model_path']
agent_params['warm_start'] = params['warm_start']
agent_params['cmd_input_mode'] = params['cmd_input_mode']


torch.manual_seed(params['torch_seed'])

if agt == 0:
    agent = AgentCmd(kb, act_set, slot_set, agent_params)
elif agt == 1:
    agent = InformAgent(kb, act_set, slot_set, agent_params)
elif agt == 2:
    agent = RequestAllAgent(kb, act_set, slot_set, agent_params)
elif agt == 3:
    agent = RandomAgent(kb, act_set, slot_set, agent_params)
elif agt == 4:
    #agent = EchoAgent(kb, act_set, slot_set, agent_params)
    agent = RequestInformSlotAgent(kb, act_set, slot_set, agent_params, movie_request_slots, movie_inform_slots)
elif agt == 5: # movie request rule agent
    agent = RequestBasicsAgent(kb, act_set, slot_set, agent_params, movie_request_slots)
elif agt == 6: # restaurant request rule agent
    agent = RequestBasicsAgent(kb, act_set, slot_set, agent_params, restaurant_request_slots)
elif agt == 7: # taxi request agent
    agent = RequestBasicsAgent(kb, act_set, slot_set, agent_params, taxi_request_slots)
elif agt == 8: # taxi request-inform rule agent
    agent = RequestInformSlotAgent(kb, act_set, slot_set, agent_params, taxi_request_slots, taxi_inform_slots)
elif agt == 9: # DQN agent for movie domain
    agent = AgentDQN(kb, act_set, slot_set, agent_params)
    agent.initialize_config(movie_request_slots, movie_inform_slots)
elif agt == 10: # restaurant request-inform rule agent
    agent = RequestInformSlotAgent(kb, act_set, slot_set, agent_params, restaurant_request_slots, restaurant_inform_slots)
elif agt == 11: # taxi request-inform-cost rule agent
    agent = RequestInformSlotAgent(kb, act_set, slot_set, agent_params, taxi_request_slots, taxi_inform_cost_slots)
elif agt == 12: # DQN agent for restaurant domain
    load_actions(dialog_config.restaurant_sys_request_slots, dialog_config.restaurant_sys_inform_slots)
    agent = AgentDQN(kb, act_set, slot_set, agent_params)
    agent.initialize_config(restaurant_request_slots, restaurant_inform_slots)
elif agt == 13: # DQN agent for taxi domain
    load_actions(dialog_config.taxi_sys_request_slots, dialog_config.taxi_sys_inform_slots)
    agent = AgentDQN(kb, act_set, slot_set, agent_params)
    agent.initialize_config(taxi_request_slots, taxi_inform_slots)
elif agt == 14: # ERL agent for movie domain
    agent = ERL(kb, act_set, slot_set, agent_params)
    agent.initialize_config(movie_request_slots, movie_inform_slots)
elif agt == 15: # ICM agent for movie domain
    agent = AgentICM(kb, act_set, slot_set, agent_params)
    agent.initialize_config(movie_request_slots, movie_inform_slots)
elif agt == 16: # noiseDQN agent for movie domain
    agent = AgentNoiseDQN(kb, act_set, slot_set, agent_params)
    agent.initialize_config(movie_request_slots, movie_inform_slots)
elif agt == 17: # EA agent for movie domain
    agent = ERL(kb, act_set, slot_set, agent_params)
    agent.initialize_config(movie_request_slots, movie_inform_slots)
################################################################################
#    Add your agent here
################################################################################
else:
    pass

################################################################################
#   Parameters for User Simulators
################################################################################
usersim_params = {}
usersim_params['max_turn'] = max_turn
usersim_params['slot_err_probability'] = params['slot_err_prob']
usersim_params['slot_err_mode'] = params['slot_err_mode']
usersim_params['intent_err_probability'] = params['intent_err_prob']
usersim_params['simulator_run_mode'] = params['run_mode']
usersim_params['simulator_act_level'] = params['act_level']
usersim_params['learning_phase'] = params['learning_phase']

if usr == 0:# real user
    user_sim = RealUser(movie_dictionary, act_set, slot_set, goal_set, usersim_params)
elif usr == 1: # movie simulator
    user_sim = RuleSimulator(movie_dictionary, act_set, slot_set, goal_set, usersim_params)
elif usr == 2: # restaurant simulator
    user_sim = RuleRestaurantSimulator(movie_dictionary, act_set, slot_set, goal_set, usersim_params)
elif usr == 3: # taxi simulator
    user_sim = RuleTaxiSimulator(movie_dictionary, act_set, slot_set, goal_set, usersim_params)

################################################################################
#    Add your user simulator here
################################################################################
else:
    pass


################################################################################
# load trained NLG model
################################################################################
nlg_model_path = params['nlg_model_path']
diaact_nl_pairs = params['diaact_nl_pairs']
nlg_model = nlg()
nlg_model.load_nlg_model(nlg_model_path)
nlg_model.load_predefine_act_nl_pairs(diaact_nl_pairs) # load nlg templates

agent.set_nlg_model(nlg_model)
user_sim.set_nlg_model(nlg_model)


################################################################################
# load trained NLU model
################################################################################
nlu_model_path = params['nlu_model_path']
nlu_model = nlu()
nlu_model.load_nlu_model(nlu_model_path)

agent.set_nlu_model(nlu_model)
user_sim.set_nlu_model(nlu_model)


################################################################################
# Dialog Manager
################################################################################
dialog_manager = DialogManager(agent, user_sim, act_set, slot_set, kb)
    
    
################################################################################
#   Run num_episodes Conversation Simulations
################################################################################
status = {'successes': 0, 'count': 0, 'cumulative_reward': 0}

simulation_epoch_size = params['simulation_epoch_size']
batch_size = params['batch_size'] # default = 16
warm_start = params['warm_start']
warm_start_epochs = params['warm_start_epochs']

success_rate_threshold = params['success_rate_threshold']
save_check_point = params['save_check_point']


""" Best Model and Performance Records """
best_model = {}
best_res = {'success_rate': 0, 'ave_reward':float('-inf'), 'ave_turns': float('inf'), 'epoch':0}
if agt==14:
    best_model['model'] = copy.deepcopy(agent.learner)
else:
    best_model['model'] = copy.deepcopy(agent)
best_res['success_rate'] = 0

performance_records = {}
performance_records['success_rate'] = {}
performance_records['ave_turns'] = {}
performance_records['ave_reward'] = {}


""" Save model """
def save_model(path, agt, success_rate, agent, best_epoch, cur_epoch):
    filename = 'agt_%s_%s_%s_%.5f.p' % (agt, best_epoch, cur_epoch, success_rate)
    filepath = os.path.join(path, filename)
    checkpoint = {}
    if (agt == 9 or agt == 12 or agt == 13): checkpoint['model'] = copy.deepcopy(agent.dqn)
    checkpoint['params'] = params
    try:
        pickle.dump(checkpoint, open(filepath, "wb"))
        print ('saved model in %s' % (filepath, ))
    except Exception as e:
        print ('Error: Writing model fails: %s' % (filepath, ))
        print (e)

""" save performance numbers """
def save_performance_records(path, agt, records,seed):
    filename = 'agt_%s_result_seed_%s.json' % (agt,seed)
    filepath = os.path.join(path, filename)

    # 检查并创建目录
    if not os.path.exists(path):
        os.makedirs(path)
        print "创建目录: path",path

    print records
    try:
        print "保存json文件"
        json.dump(records, open(filepath, "wb"))
        print ('saved model in %s' % (filepath, ))
    except Exception as e:
        print ('Error: Writing model fails: %s' % (filepath, ))
        print (e)

""" Run N simulation Dialogues """
def simulation_epoch(simulation_epoch_size,record_training_data=True):
    successes = 0
    cumulative_reward = 0
    cumulative_turns = 0
    
    res = {}
    for episode in xrange(simulation_epoch_size):
        dialog_manager.initialize_episode()
        episode_over = False
        while(not episode_over):
            episode_over, reward = dialog_manager.next_turn(record_training_data=record_training_data)
            cumulative_reward += reward
            if episode_over:
                if reward > 0: 
                    successes += 1
                    print ("simulation episode %s: Success" % (episode))
                else: print ("simulation episode %s: Fail" % (episode))
                cumulative_turns += dialog_manager.state_tracker.turn_count
    
    res['success_rate'] = float(successes)/simulation_epoch_size
    res['ave_reward'] = float(cumulative_reward)/simulation_epoch_size
    res['ave_turns'] = float(cumulative_turns)/simulation_epoch_size
    print ("simulation success rate %s, ave reward %s, ave turns %s" % (res['success_rate'], res['ave_reward'], res['ave_turns']))
    return res,cumulative_reward


def simulation_erl(simulation_epoch_size, record_training_data=True):
    successes = 0
    cumulative_reward = 0
    cumulative_turns = 0

    for episode in xrange(simulation_epoch_size):
        dialog_manager.initialize_episode()
        episode_over = False
        while (not episode_over):
            episode_over, reward = dialog_manager.next_turn(record_training_data=record_training_data)
            cumulative_reward += reward
            if episode_over:
                if reward > 0:
                    successes += 1
                    print ("simulation erl %s: Success" % (episode))
                else:
                    print ("simulation erl %s: Fail" % (episode))
                cumulative_turns += dialog_manager.state_tracker.turn_count


    return float(cumulative_reward)/simulation_epoch_size

""" Warm_Start Simulation (by Rule Policy) """
def warm_start_simulation():
    successes = 0
    cumulative_reward = 0
    cumulative_turns = 0
    
    res = {}
    warm_start_run_epochs = 0
    for episode in xrange(warm_start_epochs):
        dialog_manager.initialize_episode()
        episode_over = False
        while(not episode_over):
            episode_over, reward = dialog_manager.next_turn()   # 包括经验收集
            cumulative_reward += reward
            if episode_over:
                if reward > 0: 
                    successes += 1
                    print ("warm_start simulation episode %s: Success" % (episode))
                else: print ("warm_start simulation episode %s: Fail" % (episode))
                cumulative_turns += dialog_manager.state_tracker.turn_count
        
        warm_start_run_epochs += 1
        
        if len(agent.experience_replay_pool) >= agent.experience_replay_pool_size:
            break
        
    agent.warm_start = 2
    res['success_rate'] = float(successes)/warm_start_run_epochs
    res['ave_reward'] = float(cumulative_reward)/warm_start_run_epochs
    res['ave_turns'] = float(cumulative_turns)/warm_start_run_epochs
    print ("Warm_Start %s epochs, success rate %s, ave reward %s, ave turns %s" % (episode+1, res['success_rate'], res['ave_reward'], res['ave_turns']))
    print ("Current experience replay buffer size %s" % (len(agent.experience_replay_pool)))



def run_episodes(count, status):
    successes = 0
    cumulative_reward = 0
    cumulative_turns = 0
    
    if (agt == 9 or agt == 12 or agt == 13 or agt==15 or agt==16) and params['trained_model_path'] == None and warm_start == 1:
        print ('warm_start starting ...')
        warm_start_simulation()
        print ('warm_start finished, start RL training ...')

    for episode in xrange(count):
        print ("Episode: %s" % (episode))
        agent.predict_mode = False
        dialog_manager.initialize_episode()
        episode_over = False

        while(not episode_over):
            episode_over, reward = dialog_manager.next_turn(record_training_data=False)
            cumulative_reward += reward

            if episode_over:
                if reward > 0:
                    print ("Successful Dialog!")
                    successes += 1
                else: print ("Failed Dialog!")

                cumulative_turns += dialog_manager.state_tracker.turn_count
        # simulation
        if (agt == 9 or agt == 12 or agt == 13 or agt==15 or agt==16) and params['trained_model_path'] == None:
            agent.predict_mode = True    # 控制是否存经验池 存
            simulation_epoch(4)
            agent.predict_mode = False

            # 评估模型
            print "评估模型："
            simulation_res,_ = simulation_epoch(50,False)
            
            performance_records['success_rate'][episode] = simulation_res['success_rate']
            performance_records['ave_turns'][episode] = simulation_res['ave_turns']
            performance_records['ave_reward'][episode] = simulation_res['ave_reward']

            # 记录训练结果
            result_data = {}
            result_data['epoch'] = [episode]
            result_data['ave_turns'] = [simulation_res['ave_turns']]
            result_data['ave_reward'] = [simulation_res['ave_reward']]
            result_data['success_rate'] = [simulation_res['success_rate']]
            # 指定列的顺序
            columns = ['epoch', 'ave_turns', 'ave_reward', 'success_rate']
            df = pd.DataFrame(result_data, columns=columns)
            # 将DataFrame数据写入csv文件（如果文件不存在则创建，如果存在则追加写入）
            df.to_csv('acc.csv', mode='a', index=False, header=not os.path.exists('acc.csv'))
            
            if simulation_res['success_rate'] >= best_res['success_rate']:
                if simulation_res['success_rate'] >= success_rate_threshold: # threshold = 0.30
                    agent.predict_mode = True  # 控制是否存经验池 存
                    simulation_epoch(4)
            # 为后续保存最佳模型做准备
            if simulation_res['success_rate'] > best_res['success_rate']:
                agent.save("movie", "best", simulation_res['success_rate'])
                best_model['model'] = copy.deepcopy(agent)
                best_res['success_rate'] = simulation_res['success_rate']
                best_res['ave_reward'] = simulation_res['ave_reward']
                best_res['ave_turns'] = simulation_res['ave_turns']
                best_res['epoch'] = episode
                
            # agent.clone_dqn = copy.deepcopy(agent.dqn)
            print ("开始训练!")
            agent.train(episode,batch_size, 1)


            print ("Simulation success rate %s, Ave reward %s, Ave turns %s, Best success rate %s" % (performance_records['success_rate'][episode], performance_records['ave_reward'][episode], performance_records['ave_turns'][episode], best_res['success_rate']))
            if episode % save_check_point == 0 and params['trained_model_path'] == None: # save the model every 10 episodes
                agent.save("movie",episode)

                # save_model(params['write_model_dir'], agt, best_res['success_rate'], best_model['model'], best_res['epoch'], episode)
                save_performance_records(params['write_model_dir'], agt, performance_records,args.torch_seed)
        
        print("Progress: %s / %s, Success rate: %s / %s Avg reward: %.2f Avg turns: %.2f" % (episode+1, count, successes, episode+1, float(cumulative_reward)/(episode+1), float(cumulative_turns)/(episode+1)))
    print("Success rate: %s / %s Avg reward: %.2f Avg turns: %.2f" % (successes, count, float(cumulative_reward)/count, float(cumulative_turns)/count))
    status['successes'] += successes
    status['count'] += count
    
    if (agt == 9 or agt == 12 or agt == 13 or agt==15 or agt==16)  and params['trained_model_path'] == None:
        agent.save("movie", count-1)
        # save_model(params['write_model_dir'], agt, best_res['success_rate'], best_model['model'], best_res['epoch'], count)
        save_performance_records(params['write_model_dir'], agt, performance_records,args.torch_seed)


def run_erl_episodes(count, status):
    print "执行EIERL算法！"
    successes = 0
    cumulative_reward = 0
    cumulative_turns = 0
    # agent.cur_net.load_state_dict(agent.learner.dqn.state_dict())
    agent.cur_net = agent.learner.dqn
    if (agt == 14) and params['trained_model_path'] == None and warm_start == 1:
        print ('warm_start starting ...')
        warm_start_simulation()
        print ('warm_start finished, start RL training ...')
    for episode in xrange(count):
        # agent.cur_net.load_state_dict(agent.learner.dqn.state_dict())
        agent.cur_net = agent.learner.dqn
        print ("Episode: %s" % (episode))
        agent.predict_mode = False
        dialog_manager.initialize_episode()
        episode_over = False

        while (not episode_over):

            episode_over, reward = dialog_manager.next_turn(record_training_data=False)
            cumulative_reward += reward

            if episode_over:
                if reward > 0:
                    print ("Successful Dialog!")
                    successes += 1
                else:
                    print ("Failed Dialog!")

                cumulative_turns += dialog_manager.state_tracker.turn_count
        # simulation
        print "收集经验"
        if (agt == 14) and params['trained_model_path'] == None:
            agent.predict_mode = True  # 控制是否存经验池 存
            rollout_fitness = []
            for rollout_id in range(len(agent.rollout_bucket)):
                print "原始种群：",rollout_id
                # utils.hard_update(target=agent.rollout_bucket[rollout_id], source=agent.learner.dqn)
                agent.rollout_bucket[rollout_id].load_state_dict(agent.learner.dqn.state_dict())
                # agent.cur_net.load_state_dict(agent.learner.dqn.state_dict())
                agent.cur_net = agent.learner.dqn
                fitness = simulation_erl(1) 
                rollout_fitness.append(fitness)
            all_fitness = []
            for pop_id in range(len(agent.population)):
                print "进化种群：", pop_id
                # agent.cur_net.load_state_dict(agent.population[pop_id].state_dict())
                agent.cur_net = agent.population[pop_id]
                fitness = simulation_erl(1)
                all_fitness.append(fitness)

            agent.predict_mode = False
            # 评估
            print "评估模型"
            # agent.cur_net.load_state_dict(agent.learner.dqn.state_dict())
            agent.cur_net = agent.learner.dqn
            simulation_res,_ = simulation_epoch(50, False)

            performance_records['success_rate'][episode] = simulation_res['success_rate']
            performance_records['ave_turns'][episode] = simulation_res['ave_turns']
            performance_records['ave_reward'][episode] = simulation_res['ave_reward']

            # 记录训练结果
            result_data = {}
            result_data['epoch'] = [episode]
            result_data['ave_turns'] = [simulation_res['ave_turns']]
            result_data['ave_reward'] = [simulation_res['ave_reward']]
            result_data['success_rate'] = [simulation_res['success_rate']]
            # 指定列的顺序
            columns = ['epoch', 'ave_turns', 'ave_reward', 'success_rate']
            df = pd.DataFrame(result_data, columns=columns)
            # 将DataFrame数据写入csv文件（如果文件不存在则创建，如果存在则追加写入）
            df.to_csv('acc.csv', mode='a', index=False, header=not os.path.exists('acc.csv'))

            if simulation_res['success_rate'] >= best_res['success_rate']:
                if simulation_res['success_rate'] >= success_rate_threshold:  # threshold = 0.30
                    agent.predict_mode = True  # 控制是否存经验池 存
                    # agent.cur_net.load_state_dict(agent.learner.dqn.state_dict())
                    agent.cur_net = agent.learner.dqn
                    simulation_epoch(1)
            # 为后续保存最佳模型做准备
            if simulation_res['success_rate'] > best_res['success_rate']:
                # 保存最佳模型
                agent.save("movie", "best",simulation_res['success_rate'])
                best_model['model'] = copy.deepcopy(agent.learner)
                best_res['success_rate'] = simulation_res['success_rate']
                best_res['ave_reward'] = simulation_res['ave_reward']
                best_res['ave_turns'] = simulation_res['ave_turns']
                best_res['epoch'] = episode

            # agent.clone_dqn = copy.deepcopy(agent.dqn)
            if agent.rollout_size>0:
                print ("开始训练!")
                agent.train(episode)
            if agent.pop_size > 0:
                print ("开始进化!")
                agent.evolution(episode,all_fitness,rollout_fitness)
            # agent.experience_replay_pool.clear()
            print ("Simulation success rate %s, Ave reward %s, Ave turns %s, Best success rate %s" % (
            performance_records['success_rate'][episode], performance_records['ave_reward'][episode],
            performance_records['ave_turns'][episode], best_res['success_rate']))


            if episode % save_check_point == 0 and params['trained_model_path'] == None:  # save the model every 10 episodes
                agent.save("movie", episode)
                # save_model(params['write_model_dir'], agt, best_res['success_rate'], best_model['model'], best_res['epoch'], episode)
                save_performance_records(params['write_model_dir'], agt, performance_records,args.torch_seed)

        print("Progress: %s / %s, Success rate: %s / %s Avg reward: %.2f Avg turns: %.2f" % (
        episode + 1, count, successes, episode + 1, float(cumulative_reward) / (episode + 1),
        float(cumulative_turns) / (episode + 1)))
    print("Success rate: %s / %s Avg reward: %.2f Avg turns: %.2f" % (
    successes, count, float(cumulative_reward) / count, float(cumulative_turns) / count))
    status['successes'] += successes
    status['count'] += count

    if ( agt == 14) and params['trained_model_path'] == None:
        agent.save("movie", count-1)
        # save_model(params['write_model_dir'], agt, best_res['success_rate'], best_model['model'], best_res['epoch'],
        #            count)
        save_performance_records(params['write_model_dir'], agt, performance_records,args.torch_seed)



def run_ea_episodes(count, status):
    print "执行进化算法！"
    successes = 0
    cumulative_reward = 0
    cumulative_turns = 0
    # agent.cur_net.load_state_dict(agent.learner.dqn.state_dict())
    agent.cur_net = agent.learner.dqn
    if (agt == 17) and params['trained_model_path'] == None and warm_start == 1:
        print ('warm_start starting ...')
        warm_start_simulation()
        print ('warm_start finished, start RL training ...')

    print ("开始训练!")
    agent.train(-1)
    for pop_id in range(len(agent.population)):
        agent.population[pop_id].load_state_dict(agent.learner.dqn.state_dict())
    for episode in xrange(count):
        cumulative_turns_single = 0.0
        cumulative_reward_single = 0.0
        agent.predict_mode = False
        # agent.cur_net.load_state_dict(agent.learner.dqn.state_dict())
        for pop_id in range(len(agent.population)):
            agent.cur_net = agent.population[pop_id]
            print ("Episode: %s" % (episode))
            print '个体',pop_id,"进行测试"
            dialog_manager.initialize_episode()
            episode_over = False

            while (not episode_over):

                episode_over, reward = dialog_manager.next_turn(record_training_data=False)
                cumulative_reward_single += reward

                if episode_over:
                    if reward > 0:
                        print ("Successful Dialog!")
                        successes += 1
                    else:
                        print ("Failed Dialog!")

                    cumulative_turns_single += dialog_manager.state_tracker.turn_count
        cumulative_reward_single = float(cumulative_reward_single) / agent.pop_size
        cumulative_turns_single = float(cumulative_turns_single) / agent.pop_size
        cumulative_reward+=cumulative_reward_single
        cumulative_turns+=cumulative_turns_single
        # simulation
        # print "收集经验"
        # 进化算法不需要收集经验   只需要与环境交互的适应度即累计奖励
        if (agt == 17) and params['trained_model_path'] == None:
            all_fitness = []
            for pop_id in range(len(agent.population)):
                # agent.cur_net.load_state_dict(agent.population[pop_id].state_dict())
                agent.cur_net = agent.population[pop_id]
                fitness = simulation_erl(1)
                print "进化种群,个体：", pop_id,"适应度：",fitness
                all_fitness.append(fitness)
            # 评估 每一个个体  然后取最大值
            print "评估模型"
            simulation_res_all = {}
            simulation_res_all['success_rate'] = 0.0
            simulation_res_all['ave_turns'] = 0.0
            simulation_res_all['ave_reward'] = 0.0
            for pop_id in range(len(agent.population)):
                print '个体', pop_id, "进行测试"
                agent.cur_net = agent.population[pop_id]
                simulation_res, _ = simulation_epoch(50, False)
                if simulation_res['success_rate'] >= simulation_res_all['success_rate']:
                    simulation_res_all['success_rate'] = simulation_res['success_rate']
                    simulation_res_all['ave_turns'] = simulation_res['ave_turns']
                    simulation_res_all['ave_reward'] = simulation_res['ave_reward']
                    agent.best_policy = agent.population[pop_id]


            performance_records['success_rate'][episode] = simulation_res_all['success_rate']
            performance_records['ave_turns'][episode] = simulation_res_all['ave_turns']
            performance_records['ave_reward'][episode] = simulation_res_all['ave_reward']

            # 记录训练结果
            result_data = {}
            result_data['epoch'] = [episode]
            result_data['ave_turns'] = [simulation_res_all['ave_turns']]
            result_data['ave_reward'] = [simulation_res_all['ave_reward']]
            result_data['success_rate'] = [simulation_res_all['success_rate']]
            # 指定列的顺序
            columns = ['epoch', 'ave_turns', 'ave_reward', 'success_rate']
            df = pd.DataFrame(result_data, columns=columns)
            # 将DataFrame数据写入csv文件（如果文件不存在则创建，如果存在则追加写入）
            df.to_csv('acc.csv', mode='a', index=False, header=not os.path.exists('acc.csv'))


            if agent.pop_size > 0:
                print ("开始进化!")
                agent.eii(episode,all_fitness)
            print ("Simulation success rate %s, Ave reward %s, Ave turns %s, Best success rate %s" % (
            performance_records['success_rate'][episode], performance_records['ave_reward'][episode],
            performance_records['ave_turns'][episode], best_res['success_rate']))

            if episode % save_check_point == 0 and params['trained_model_path'] == None:  # save the model every 10 episodes
                agent.save("movie", episode)
                # save_model(params['write_model_dir'], agt, best_res['success_rate'], best_model['model'], best_res['epoch'], episode)
                save_performance_records(params['write_model_dir'], agt, performance_records,seed=args.torch_seed)


        print("Progress: %s / %s, Success rate: %s / %s Avg reward: %.2f Avg turns: %.2f" % (
        episode + 1, count, successes, episode + 1, float(cumulative_reward) / (episode + 1),
        float(cumulative_turns) / (episode + 1)))
    print("Success rate: %s / %s Avg reward: %.2f Avg turns: %.2f" % (
    successes, count, float(cumulative_reward) / count, float(cumulative_turns) / count))
    status['successes'] += successes
    status['count'] += count


    agent.save("movie", count-1)
    save_performance_records(params['write_model_dir'], agt, performance_records,args.torch_seed)


if agt==17:
    # 执行EA
    run_ea_episodes(num_episodes, status)
elif agt==14:
    # 执行EIERL
    run_erl_episodes(num_episodes, status)
else:
    # 执行其它
    run_episodes(num_episodes, status)




