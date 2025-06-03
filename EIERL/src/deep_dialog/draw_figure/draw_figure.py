# coding=utf-8
import argparse, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sns.set(style="darkgrid")
sns.set(font_scale=1.6)

width = 8
height = 5.8
plt.figure(figsize=(width, height))
linewidth=1.1

def read_performance(path, attribute):
    success_rate = []
    data = json.load(open(path, 'rb'))
    for key in sorted(data[attribute].keys(), key=lambda k: int(k)):
        if int(key) > -1:
            if attribute == 'discriminator_loss':
                success_rate.append(data[attribute][key]/50)
            else:
                success_rate.append(data[attribute][key])

    # 对数据进行平滑处理，此处不使用初始缓存值，直接从第一个实际数据点开始
    smooth_num = 1
    d = [success_rate[i*smooth_num:i*smooth_num + smooth_num] for i in range(int(len(success_rate)/smooth_num))]

    success_rate_new = []
    if d:
        cache = d[0][0]  # 初始化缓存值为第一个数据点的值，而非0
    else:
        cache = 0  # 如果没有数据，则默认为0（实际情况应避免）

    alpha = 0.85
    for i in d:
        cur = sum(i)/float(smooth_num)
        cache = cache * alpha + (1 - alpha) * cur
        success_rate_new.append(cache)

    return success_rate_new



# def show_model_performance(path, record_list=range(1, 4)):
#     attributes = ['success_rate', 'ave_reward', 'ave_turns']
#     records = {
#         'success_rate': {'100': 0, '200': 0, '300': 0},
#         'ave_reward': {'100': 0, '200': 0, '300': 0},
#         'ave_turns': {'100': 0, '200': 0, '300': 0}
#     }
#     for i in record_list:
#         data = json.load(open('{}_{}/agt_9_performance_records.json'.format(path, i), 'rb'))
#         for attribute in records.keys():
#             records[attribute]['100'] += data[attribute]['100'] / len(record_list)
#             records[attribute]['200'] += data[attribute]['200'] / len(record_list)
#             records[attribute]['300'] += data[attribute]['300'] / len(record_list)
#     print(path)
#     print(records)
#     return records


def draw(color, marker, linestyle, record_list=range(1,4), model_path="", attribute="success_rate"):
    datapoints = []
    for i in record_list:
        datapoints.append(
            read_performance('./acc/{}/{}_{}.json'.format(model_path, model_path, i), attribute))

    min_len = min(len(i) for i in datapoints)
    data = np.asarray([i[0:min_len] for i in datapoints])
    mean = np.mean(data,axis=0)

    var = np.std(data,axis=0)
    l, = plt.plot(range(mean.shape[0]), mean, color, marker=marker, markevery=30, markersize=11, label='Plan 1 step with Model', linewidth=linewidth)
    plt.fill_between(range(mean.shape[0]), mean+var/2, mean-var/2, facecolor=color, alpha=0.2)
    return l


def main(params):
    colors = ['#2f79c0', '#278b18', '#ff5186', '#8660a4', '#D49E0F', '#FF8800','#CD6090']
    markers = [',', 'o', '^', 's', 'p', 'd','*']
    linestyles = ['solid', 'dashed', 'dashdot', 'dotted', '-.', '--', '-', ':']
    global_idx = 1500

    # example
    model_path_list = [
        'dqn',
        'dqn_epsilon_0.05',
        'eierl_pop_3',
        'noise_dqn',
        'icm',
        'gpt',
        'gpt2'
    ]
    label_list = [ 'DQN_EPSILON_0.0','DQN_EPSILON_0.05','EIERL','NOISY_DQN','ICM_DQN','LLM_DP','LLM_DP_NLG']
    curve_list = []
    for i, model in enumerate(model_path_list):
        record_list = range(1,6)
        curve_list.append(draw(model_path=model, color=colors[i], marker=markers[i], linestyle=linestyles[i], record_list=record_list))

    plt.grid(True)
    plt.ylabel('Success rate')
    plt.xlabel('Epoch')
    plt.legend(curve_list, label_list, loc='upper left', frameon=False)
    plt.xlim([0, 500])
    plt.ylim([-0.01, 0.9])
    plt.savefig('./figure.png', format='png')  # Save as PNG


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_file', dest='result_file', type=str, default='agt_10_performance_records.json', help='path to the result file')

    args = parser.parse_args()
    params = vars(args)
    print(json.dumps(params, indent=2))

    main(params)
