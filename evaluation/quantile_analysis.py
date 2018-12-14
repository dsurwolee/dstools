import pandas as pd
import numpy as np

def quantile_analysis(score, label, percentiles=[0.1,1,0.1], cost=None):
    from collections import OrderedDict
    stats = OrderedDict({
        'top_ntiles': [],
        'size': [],
        'thresholds': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'f2': [],
        'tps': [],
        'fps': [],
        'fns': []
    })
    if isinstance(cost, (pd.Series)):
        stats[cost.name] = []

    for n in np.arange(*percentiles):
        t = score.quantile(1-n)
        size = (score >= t).sum()
        precision = label[score >= t].mean()
        recall = label[score >= t].sum() / label.sum()
        f1 = 2 * (precision * recall) / (precision + recall)
        f2 = 5 * (precision * recall) / (4 * precision + recall)
        tps = label[score >= t].sum()
        fps = size - tps
        fns = label.sum() - tps
        if isinstance(cost, (pd.Series)):
            total = cost[(score >= t) &  (label == 1)].sum()

        stats['top_ntiles'].append(n)
        stats['thresholds'].append(t)
        stats['size'].append(size)
        stats['precision'].append(precision)
        stats['recall'].append(recall)
        stats['f1'].append(f1)
        stats['f2'].append(f2)
        stats['tps'].append(tps)
        stats['fps'].append(fps)
        stats['fns'].append(fns)
        if isinstance(cost, (pd.Series)):
            stats[cost.name].append(total)
    return pd.DataFrame(stats)