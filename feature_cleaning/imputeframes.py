import pandas as pd
import numpy as np

class ImputeFrames:
    """ Imputes frames with choice to fill
          
        Impute Methods: ['median', 'mean'] or any value
     
        Use fit to compute statistics from single column.
        Use the impute method to fill-in the column with 
        the fillna method selection.
    """
    def __init__(self, method='median', fill_value=None):  
        self.method = method
        self.fill_value = fill_value
     
    def fit(self, series):
        if self.method == 'mean':
            self.fill_value = series.mean()
        if self.method == 'median':
            self.fill_value =  series.median()
    
    def impute(self, series):
        return series.fillna(self.fill_value) 