import random
from itertools import combinations
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import multiprocessing as mp
import networkx as nx
import pandas as pd
import numpy as np

class MulticollinearityNetwork:
    
    def __init__(self, variables, processes, k=0.75, pair_sample_rate=None, row_sample=None):
        self.variables = variables
        self.row_sample = row_sample
        self.k = k
        self.processes = processes
        self.pair_sample_rate = pair_sample_rate
        
    def pairwise_spearman(self, X, pairs):
        
        pool = mp.Pool(self.processes)
        def spearman(func, x1, x2): 
            return pool.apply(func, args=(x1, x2))[0]

        results = [(x1, x2, spearman(spearmanr,X[x1],X[x2])) for x1, x2 in pairs]
        return results
        
    def compute_correlations(self, X):
        
        pairs = np.array(list(combinations(self.variables, 2)))
        if self.pair_sample_rate:
            pair_size = int(len(pairs) * self.pair_sample_rate)
            pairs_indices = np.random.choice(pairs.shape[0], size=pair_size, replace=False)
            pairs = pairs[pairs_indices].tolist()
            
        if self.row_sample:
            X = X.sample(frac=self.row_sample, replace=False)
            
        return self.pairwise_spearman(X, pairs)
    
    def correlation_network(self, correlations):
        self.G = nx.Graph()
        
        # Add nodes
        for v in self.variables:
            self.G.add_node(v)
        
        # Add connected variables
        for v1, v2, c in correlations:
            if c >= self.k:
                self.G.add_edge(v1, v2)
        return self.G
    
    def get_node_clusters(self, correlation_network):
        subgraphs = {}
        generator_correlation_network = nx.connected_component_subgraphs(correlation_network)
        for i, sg in enumerate(generator_correlation_network):
            subgraphs[i] = list(sg.nodes())
        self.subgraphs = subgraphs
        return subgraphs
    
    def select_nodes(self, sample=None, size_cutoff=None):
        top_variable_list = []
        for i in nx.connected_component_subgraphs(self.G):
            sg_betweenness = nx.betweenness_centrality(i)
            sorted_variable = sorted(sg_betweenness, key=sg_betweenness.__getitem__, reverse=True)
            top_variable_list.append(sorted_variable[0])
        return top_variable_list