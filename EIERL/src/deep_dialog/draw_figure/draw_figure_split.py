#!/usr/bin/env python 
# -*- coding:utf-8 -*-
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

linewidth = 1.1

def read_performance(path, attribute):
    success_rate = []
    data = json.load(open(path, 'rb'))
    for key in sorted(data[attribute].keys(), key=lambda k: int(k)):
        if int(key) > -1:
            if attribute == 'discriminator_loss':
                success_rate.append(data[attribute][key]/50)
            else:
                success_rate.append(data[attribute][key])

    smooth_num = 1
    d = [success_rate[i*smooth_num:i*smooth_num + smooth_num] for i in range(int(len(success_rate)/smooth_num))]

    success_rate_new = []
    if d:
        cache = d[0][0]
    else:
        cache = 0

    alpha = 0.85
    for i in d:
        cur = sum(i)/float(smooth_num)
        cache = cache * alpha + (1 - alpha) * cur
        success_rate_new.append(cache)

    return success_rate_new

def draw(color, marker, linestyle, record_list=range(1,4), model_path="", attribute="success_rate"):
    datapoints = []
    for i in record_list:
        datapoints.append(
            read_performance('./acc/{}/{}_{}.json'.format(model_path,model_path, i), attribute))

    min_len = min(len(i) for i in datapoints)
    data = np.asarray([i[0:min_len] for i in datapoints])
    mean = np.mean(data,axis=0)
    var = np.std(data,axis=0)
    l, = plt.plot(range(mean.shape[0]), mean, color, marker=marker, markevery=30, markersize=11,
                  label='Plan 1 step with Model', linewidth=linewidth)
    plt.fill_between(range(mean.shape[0]), mean + var / 2, mean - var / 2, facecolor=color, alpha=0.2)
    return l

def main(params):
    plt.figure(figsize=(width, height))  # Create a figure for the curve plot
    colors = ['#2f79c0', '#278b18', '#ff5186', '#8660a4', '#D49E0F', '#FF8800', '#CD6090']
    markers = [',', 'o', '^', 's', 'p', 'd', '*']
    linestyles = ['solid', 'dashed', 'dashdot', 'dotted', '-.', '--', '-', ':']

    model_path_list = [
        'dqn',
        'dqn_epsilon_0.05',
        'dqn_epsilon_0.1',
        'dqn_epsilon_0.15',
        'dqn_epsilon_0.2',
        'dqn_epsilon_0.25'
    ]
    label_list = ['DQN_EPSILON_0.0', 'DQN_EPSILON_0.05', 'DQN_EPSILON_0.1', 'DQN_EPSILON_0.15', 'DQN_EPSILON_0.2',
                  'DQN_EPSILON_0.25']

    curve_list = []
    for i, model in enumerate(model_path_list):
        record_list = range(1,6)
        curve_list.append(draw(model_path=model, color=colors[i], marker=markers[i], linestyle=linestyles[i], record_list=record_list))

    plt.grid(True)
    plt.ylabel('Success rate')
    plt.xlabel('Epoch')
    plt.xlim([0, 500])
    plt.ylim([-0.01, 0.9])
    plt.savefig('./curve_plot.png', format='png')  # Save curve plot as PNG

    # Create a new figure for the legend
    fig, ax = plt.subplots()
    ax.set_axis_off()
    legend = plt.legend(curve_list, label_list, loc='center', frameon=False)
    fig.canvas.draw()
    bbox = legend.get_frame().get_bbox().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig('./legend_plot.png', format='png', bbox_inches=bbox)  # Save legend plot with reduced whitespace as PNG

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_file', dest='result_file', type=str, default='agt_10_performance_records.json', help='path to the result file')
    args = parser.parse_args()
    params = vars(args)
    print(json.dumps(params, indent=2))

    main(params)
