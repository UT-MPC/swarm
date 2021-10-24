import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

cmap = get_cmap('Dark2')

LOCAL = 'local'
GREEDY = 'greedy'
OPPORTUNISTIC = 'opportunistic'
GR = 'gradient replay'

OPPORTUNISTIC_LOW_THRES = 'opportunistic (low thres.)'
GR_LOW_THRES = 'gradient replay (low thres.)'

OPPORTUNISTIC_WEIGHTED = 'opportunistic-weighted'
GR_NOT_WEIGHTED = 'gradient replay not weighted'

GR_WITHOUT_DECAY = 'gradient replay without decay'

plot_colors = {
    LOCAL: cmap(0),
    GREEDY: cmap(1),
    OPPORTUNISTIC: cmap(2),
    GR: cmap(3),

    OPPORTUNISTIC_LOW_THRES: cmap(4),
    GR_LOW_THRES: cmap(5)
}

plot_styles = {
    LOCAL: 'dotted',
    GREEDY: 'dashed',
    OPPORTUNISTIC: 'dashdot',
    GR: 'solid',

    OPPORTUNISTIC_LOW_THRES: (0, (3, 1, 1, 1, 1, 1)),
    GR_LOW_THRES: (0, (5, 1))
}

plot_legends = {
    LOCAL: 'local',
    GREEDY: 'greedy',
    OPPORTUNISTIC: 'opp',
    GR: 'GR',

    OPPORTUNISTIC_LOW_THRES: 'opp-low',
    GR_LOW_THRES: 'GR-low',

    OPPORTUNISTIC_WEIGHTED: 'opp-w',
    GR_NOT_WEIGHTED: 'GR-wo-w',

    GR_WITHOUT_DECAY: 'GR-wo-decay'
}

def draw_graph(logs, key, filename, lower_ylim=None):
    legends = []
    plt.figure(figsize=(8, 4))
    for k in logs.keys():
        plt.plot(np.arange(0, len(logs[k][key])), np.array(logs[k][key]), 
        lw=1.2, color=plot_colors[k], linestyle=plot_styles[k])
        legends.append(plot_legends[k])
    plt.legend(legends)
    key = key[0].upper() + key[1:]
    plt.ylabel(key)
    if lower_ylim != None:
        plt.ylim(bottom=lower_ylim)
    plt.xlabel("Encounters")
    plt.grid(linestyle=':')
    plt.savefig(filename)
    plt.close()

def main():
    # parse arguments
    parser = argparse.ArgumentParser(description='set params for controlled experiment')
    parser.add_argument('--hist', dest='log_file',
                        type=str, default=None, help='log file')
    parser.add_argument('--out', dest='graph_file',
                        type=str, default='figs/figure.pdf', help='output figure name')
    parser.add_argument('--metrics', dest='metrics',
                        type=str, default='loss-and-accuracy', help='metrics')
    parser.add_argument('--lower', dest='lower',
                        type=float, default=None, help='lower y limit')

    parsed = parser.parse_args()  

    with open(parsed.log_file, 'rb') as handle:
        logs = pickle.load(handle)

    if parsed.metrics == 'loss-and-accuracy':
        key = 'accuracy'
    elif parsed.metrics == 'f1-score-weighted':
        key = 'f1-score'
    else:
        ValueError('invalid metrics: {}'.format(parsed.metrics))
    
    if parsed.graph_file != None:
        draw_graph(logs, key, parsed.graph_file, parsed.lower)

if __name__ == '__main__':
    main()