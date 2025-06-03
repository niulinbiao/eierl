# coding=utf-8
import os,json
from torch.utils.tensorboard import SummaryWriter

class Parameters:
    def __init__(self, parser):
        """Parameter class stores all parameters for policy gradient

        Parameters:
            None

        Returns:
            None
        """
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../config.json'), 'r') as f:
            cfg = json.load(f)
        #Env args
        # self.frameskip = vars(parser.parse_args())['frameskip']
        self.total_steps = int(cfg['total_steps'] * 1000000)
        # self.gradperstep = vars(parser.parse_args())['gradperstep'] 配置文件
        self.savetag = cfg['savetag']
        self.seed =  (cfg['seed'])
        # self.batch_size = vars(parser.parse_args())['batchsize']配置文件
        # self.rollout_size = vars(parser.parse_args())['rollsize'] 配置文件

        # 附加
        self.log_dir_path = vars(parser.parse_args())['log_dir_path']
        self.log_path_suffix = vars(parser.parse_args())['log_path_suffix']
        self.dataset_name = vars(parser.parse_args())['dataset_name']
        self.model_name = vars(parser.parse_args())['model_name']
        self.load_path = vars(parser.parse_args())['load_path']
        self.batchsz = int(vars(parser.parse_args())['batchsz'])
        self.process_num = int(vars(parser.parse_args())['process_num'])

        # self.hidden_size = vars(parser.parse_args())['hidden_size']  配置文件
        # self.actor_lr = vars(parser.parse_args())['actor_lr'] 配置文件
        # self.tau = vars(parser.parse_args())['tau']
        # self.gamma = vars(parser.parse_args())['gamma']配置文件
        # self.reward_scaling = vars(parser.parse_args())['reward_scale']
        # self.buffer_size = int(vars(parser.parse_args())['buffer'] * 1000000)  配置文件
        # self.learning_start = vars(parser.parse_args())['learning_start'] 配置文件

        # self.pop_size = vars(parser.parse_args())['popsize'] 配置文件
        # self.num_test = vars(parser.parse_args())['num_test'] 配置文件
        self.test_frequency = 1
        self.asynch_frac = 1.0  # Aynchronosity of NeuroEvolution

        #Non-Args Params
        self.elite_fraction = 0.2
        self.crossover_prob = 0.15
        self.mutation_prob = 0.90
        self.extinction_prob = 0.005  # Probability of extinction event
        self.extinction_magnituide = 0.5  # Probabilty of extinction for each genome, given an extinction event
        self.weight_magnitude_limit = 10000000
        self.mut_distribution = 1  # 1-Gaussian, 2-Laplace, 3-Uniform


        # self.alpha = vars(parser.parse_args())['alpha']
        self.target_update_interval = 1
        # self.alpha_lr = 1e-3

        #Save Results
        self.savefolder = 'Results/Plots/'
        if not os.path.exists(self.savefolder): os.makedirs(self.savefolder)
        self.aux_folder = 'Results/Auxiliary/'
        if not os.path.exists(self.aux_folder): os.makedirs(self.aux_folder)

        self.save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), cfg['save_dir2'])
        self.gradperstep = cfg['gradperstep']
        self.num_test = cfg['num_test']
        self.rollout_size = cfg['rollout_size']
        self.pop_size = cfg['pop_size']


        self.savetag += str('erl_dqn')
        self.savetag += '_seed' + str(self.seed)
        self.savetag += '_roll' + str(self.rollout_size)
        self.savetag += '_pop' + str(self.pop_size)
        # self.savetag += '_alpha' + str(self.alpha)


        self.writer = SummaryWriter(log_dir='Results/tensorboard/' + self.savetag)




