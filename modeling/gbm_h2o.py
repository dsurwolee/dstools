"""
This module 

"""

# Third-Party Libraries
import pandas as pd
import numpy as np
import h2o
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.grid.grid_search import H2OGridSearch

# Custom Module
from .quantile_analysis import quantile_analysis

# =============================================================================
# GBM H2O models training and performance evaluation 
# =============================================================================

class H2OGBMGrid:
    """GBM grid search and performance evaluation
    
       
    """


    def __init__(self,
                 grid_id, 
                 model_params,
                 hyper_params, 
                 search_params={'strategy':'Cartesian'}):
        self.grid_id = grid_id
        self.model_params = model_params
        self.hyper_params = hyper_params
        self.search_params = search_params

    def create_grid(self):
        """Returns an H2O grid search object 
        """
        gbm_model = H2OGradientBoostingEstimator(**self.model_params)
        gbm_grid = H2OGridSearch(model=gbm_model, 
                                 hyper_params=self.hyper_params, 
                                 grid_id=self.grid_id, 
                                 search_criteria=self.search_params)
        return gbm_grid
    
    def train(self, grid, train, valid, X, y):
        """Performs grid search on GBM
        """
        self.gbm_grid = grid
        self.gbm_grid.train(x=X, y=y, training_frame=train, 
                            validation_frame=valid)
        
    def predict(self, model, X):
        return model.predict(X)
    
    def save_models(self, fpath):
        """Save trained models to path
        """
        for name in self.gbm_grid.model_ids:
            model = h2o.get_model(name)
            h2o.save_model(model, fpath+model.model_id)
    
    def performance(self, X, y, percentiles=[0.1,1,0.1], cost=None, 
                    filterby=[0.1], _round=3, sort_by=None):
        """
        
        Parameters
        ----------
        X : H2O DataFrame containing feature vectors
            
        y : Array of binary values
        
        percentiles : list, (default=[0.1,1,0.1])
            abc
        cost : Array, (default=None)
            abc
        filterby :
        _round :
        
        sort_by : 
        
        Returns
        -------        
        self.model_comparison : Pandas DataFrame    
        """
        # Perform quantile analysis
        self.quantile_analysis = {}
        self.h2o_model_objects = {}
        for model_id in self.gbm_grid.model_ids:
            model = h2o.get_model(model_id)
            self.h2o_model_objects[model_id] = model
            pred_y = self.predict(model, X).as_data_frame()['p1']
            table = quantile_analysis(pred_y, y, percentiles=percentiles, cost=cost)
            self.quantile_analysis[model_id] = table
        
        # Create dataframe with a header
        header = ['model_id', 'train_AUC', 'valid_AUC', 'test_AUC']
        for param_name in self.hyper_params:        # Add hyper-parameter names
            header.append(param_name)
        measures = ['thres', 'precision', 'recall'] # Add quantile measures
        if isinstance(cost, (np.ndarray, pd.Series, list)):
            measures.append('cost')
        for k in filterby:
            for m in measures:
                header.append(str(k) + '_' + m)
        self.model_comparison = pd.DataFrame(columns=header)
        
        # Append rows
        for model_id in self.h2o_model_objects:
            row = {}
            model = self.h2o_model_objects[model_id]
            # Add model id values
            row['model_id'] = model_id
            # Add model AUC
            row['train_AUC'] = model.model_performance(train=True).auc()
            row['valid_AUC'] = model.model_performance(valid=True).auc()
            row['test_AUC'] = model.model_performance(test_data=X).auc()
            # Add hyper parameter values
            for param in self.hyper_params:
                value = model.params[param]['actual']
                row[param] = value
            # Add threshold, precision, recall (and cost)
            table = self.quantile_analysis[model_id]
            for k in filterby:
                table_row = table.loc[table['top_ntiles'] == k]
                for m in measures:
                    m_header = str(k) + '_' + m
                    row[m_header] = table_row[m].iloc[0]
            self.model_comparison = self.model_comparison.append(row, ignore_index=True)
        self.model_comparison = self.model_comparison.round(3)
        if sort_by:
            self.model_comparison = self.model_comparison.sort_values(sort_by, ascending=False)
        return self.model_comparison