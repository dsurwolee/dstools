# Multicollinearity Analysis for Event #1 
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class MulticollinearityNetwork:
    
    def __init__(self, data, variables, col_sample=1, row_sample=1, stat='spearman'):

        if col_sample != 1:
            variables = np.random.choice(variables, replace=False, size=int(len(variables) * col_sample))
            data = data[variables]

        if row_sample != 1:
            data = data.sample(frac=row_sample, replace=False)

        self.data = data
        self.variables = variables
        self.stat = stat
        
    def compute(self, na='None', k=0.75):

        if isinstance(na, (int)):
            corr_df = (
                self.data[self.variables]
                    .fillna(na)
                    .corr(self.stat)
                    .stack()
                    .reset_index()
                    .rename(columns={'level_0': 'x1','level_1': 'x2', 0: self.stat})
            )
        elif na == 'median':
            corr_df = (
                self.data[self.variables]
                    .fillna(np.median)
                    .corr(self.stat)
                    .stack()
                    .reset_index()
                    .rename(columns={'level_0': 'x1','level_1': 'x2', 0: self.stat})
            )
        else: 
            corr_df = (
                self.data[self.variables]
                    .corr(self.stat)
                    .stack()
                    .reset_index()
                    .rename(columns={'level_0': 'x1','level_1': 'x2', 0: self.stat})
            )
        corr_df = corr_df[(corr_df.x1 != corr_df.x2)]
        corr_df['greater_than_or_equal_to_k'] = False
        corr_df.loc[corr_df[self.stat] >= k, 'greater_than_or_equal_to_k'] = True        
        count_of_corr = (
             corr_df.groupby('x1')['greater_than_or_equal_to_k']
                .sum()
                .sort_values()
        )
        return corr_df[corr_df['greater_than_or_equal_to_k'] == True], count_of_corr 
    
    def network(self, corr_df, figsize=(10,10)):
        
        G = nx.Graph()
        for v1, v2 in zip(corr_df['x1'],corr_df['x2']):
            G.add_edge(v1, v2)
        self.G = G
        
        f, ax = plt.subplots(figsize=figsize)
        nx.draw(G, with_labels=True, font_size=8, node_size=13)
    
    def subgraphs(self):
        sub_graphs = list(nx.connected_component_subgraphs(self.G))

        #n gives the number of sub graphs
        n = len(sub_graphs)

        # you can now loop through all nodesdata
        graphs = {}
        for i in range(n):
            sub_group = list(sub_graphs)[i].nodes()
            graphs[i] = sub_group
        
        return graphs
