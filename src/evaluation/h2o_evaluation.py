import pandas as pd
import numpy as np
import h2o
from .quantile_analysis import *

class TrainH2OGBMModel:
    def __init__(self, model_params={}, hyper_params={}, grid_id='', search_params={'strategy':'Cartesian'}):
        """ Instantiate H2O GBM model with H2O grid object """
        if grid_id == '':
            raise Exception('must provide grid id.')
        
        from h2o.estimators.gbm import H2OGradientBoostingEstimator
        from h2o.grid.grid_search import H2OGridSearch
        
        gbm_model = H2OGradientBoostingEstimator(**model_params)
        gbm_grid = H2OGridSearch(model=gbm_model, hyper_params=hyper_params, grid_id=grid_id, search_criteria=search_params)
        self.hyper_params = hyper_params
        self.gbm_grid = gbm_grid
        
    def save_models(self, filepath):
        _id = self.gbm_grid.grid_id
        for name in self.gbm_grid.model_ids:
            model = h2o.get_model(name)
            h2o.save_model(model, filepath+_id+'_'+name)
        
    def train(self, train, valid, features='', target=''):
        self.gbm_grid.train(x=features, y=target, training_frame=train, validation_frame=valid)
        return self.gbm_grid
    
    def recall_and_precision(self, test_scores, test_labels, quantile_cutoff):
        max_cutoff = quantile_cutoff * 2
        quantile_df = quantile_analysis(test_scores, test_labels, percentiles=[quantile_cutoff,max_cutoff,quantile_cutoff])
        threshold = round(float(quantile_df['thresholds']),4)
        precision = round(float(quantile_df['precision']),4)
        recall = round(float(quantile_df['recall']),4)
        return threshold, precision, recall
    
    def evaluate(self, testhex, labels, quantiles=[0.0001,0.001,0.005,0.01,0.1]):
        # Create Dictionary
        model_metrics_dict = {
            'model_id': [],
            'train': [],
            'valid': [],
            'test_data': [], 
        }
        for param in self.hyper_params:
            model_metrics_dict[param] = []
        
        for cutoff in quantiles:
            cutoff = str(cutoff).replace('0.','_pt_')
            for stat in ['threshold','precision', 'recall']:
                name = stat+cutoff
                model_metrics_dict[name] = []
        
        AUC_metric_setting = [ # Create AUC keys
                {'train': True}, 
                {'valid': True},  
                {'test_data': testhex}
        ] 
        
        # Populate Dictionary
        models_ids = self.gbm_grid.model_ids
        for model_id in models_ids:
            # Get Model
            model_metrics_dict['model_id'].append(model_id)
            model = h2o.get_model(model_id) 
            # Grab parameter values of a trained model
            model_parameters = model.params 
            for param in self.hyper_params: 
                param_value = model_parameters[param]['actual']
                model_metrics_dict[param].append(param_value)
            for setting in AUC_metric_setting: # Get AUC
                key = list(setting.keys())[0]
                auc = model.model_performance(**setting).auc()
                model_metrics_dict[key].append(auc)
            scores = model.predict(testhex).as_data_frame()['p1']
            for cutoff in quantiles:
                t, prec, recall = self.recall_and_precision(scores, labels, cutoff)
                cutoff = str(cutoff).replace('0.','_pt_')
                threshold_name = 'threshold'+cutoff
                precision_name = 'precision'+cutoff
                recall_name = 'recall'+cutoff
                model_metrics_dict[threshold_name].append(t)
                model_metrics_dict[precision_name].append(prec)
                model_metrics_dict[recall_name].append(recall)
        return model_metrics_dict # pd.DataFrame(model_metrics_dict).set_index('model_id')