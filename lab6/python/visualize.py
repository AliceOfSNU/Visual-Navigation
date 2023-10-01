import matplotlib
import matplotlib.pyplot as plt
import sys
sys.path.append(".")
import os
import time
import os.path as osp
import numpy as np
from collections import OrderedDict
import argparse
import seaborn as sns
import csv



def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('--path', type=str, nargs='+', default=(0,),
                        help='which csv files to read')
    parser.add_argument('--tags', type=str, nargs='+', default=[],
                        help='legend items')
    parser.add_argument('--item', type=str, nargs='+', default=('mean_success_rate'),
                        help='which column to read')
    parser.add_argument('--max_m', type=int, default=None,
                        help='maximum million')
    parser.add_argument('--smooth_coeff', type=int, default=25,
                        help='smooth coeff')
    parser.add_argument('--title', type=str, default='experiment',
                        help='title')
    parser.add_argument('--output_dir', type=str, default='../fig',
                        help='directory for plot output (default: ../fig)')
    args = parser.parse_args()
    return args

def post_process(array, smooth_para):
    new_array = []
    for i in range(len(array)):
        if i < len(array) - smooth_para:
            new_array.append(np.mean(array[i:i+smooth_para]))
        else:
            new_array.append(np.mean(array[i:None]))
    return new_array    

# driver code
args = get_args()
fig = plt.figure(figsize=(10,7))
ax1 = fig.add_subplot(111)
colors = ['b', 'g', 'r', 'c', 'y' ]
styles = ['solid', 'dotted', 'dashed']

'''
 @ param
 @ param
'''

for linecnt, (path, tag) in enumerate(zip(args.path, args.tags)):
    lc = colors[linecnt % len(colors)]
    step_number = []

    #assume no repetition of experiments(N=1 always, no average across seeds)
    temp_ys = []
    temp_xs = []
    with open(path,'r') as f:
        csv_reader = csv.DictReader(f)
        n = 0
        for row in csv_reader:
            temp_ys.append(float(row[args.item[0]])) #read item
            temp_xs.append(n)
            n+=1

    step_number = np.array(temp_xs) / 10 # time axis
    final_step = []
    last_step = len(step_number)
    for i in range(len(step_number)):
        if args.max_m is not None and step_number[i] >= args.max_m:
            last_step = i
            break
        final_step.append(step_number[i])

    temp_ys= temp_ys[:last_step]
    # main plot
    ys = post_process(temp_ys, args.smooth_coeff)
    ax1.plot(final_step, ys, label=tag, color=lc, linewidth=2, alpha = 1.0)
    # no smoothing
    ax1.plot(final_step, temp_ys, color=lc, linewidth=2, alpha=0.3)

ax1.set_xlabel('time', fontsize=30)
ax1.tick_params(labelsize=25)

box = ax1.get_position()

leg = ax1.legend(
           loc='upper center', bbox_to_anchor=(0.5, -0.05),
           ncol=5,
           fontsize=25)

for legobj in leg.legendHandles:
    legobj.set_linewidth(10.0)

plt.title(args.title, fontsize=40)





if not os.path.exists( args.output_dir ):
    os.mkdir( args.output_dir )
plt.savefig( os.path.join( args.output_dir, args.title ) )
plt.close()