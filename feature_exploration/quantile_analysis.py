import numpy as np
import pandas as pd

def quantile_analysis(score, label, percentiles=[0.1,1,0.1]): 
    from collections import OrderedDict
    stats = OrderedDict({
        'top_ntiles': [],
        'size': [],
        'thresholds': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'tps': [],
        'fps': [],
        'fns': []
    })
    
    for n in np.arange(*percentiles):
        t = score.quantile(1-n)
        size = (score >= t).sum()
        precision = label[score >= t].mean()
        recall = label[score >= t].sum() / label.sum()
        f1 = 2 * (precision * recall) / (precision + recall)
        tps = label[score >= t].sum()
        fps = size - tps
        fns = label.sum() - tps
        
        stats['top_ntiles'].append(n)
        stats['thresholds'].append(t)
        stats['size'].append(size)
        stats['precision'].append(precision)
        stats['recall'].append(recall)
        stats['f1'].append(f1)
        stats['tps'].append(tps)
        stats['fps'].append(fps)
        stats['fns'].append(fns)
    return pd.DataFrame(stats)
