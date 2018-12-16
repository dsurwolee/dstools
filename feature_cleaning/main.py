import pandas as pd
import numpy as np
from .feature_cleaning import * 
from .feature_engineering import *

def cleaner(x, y, method, args, method_mapping):
    
    func = method_mapping[method]
    if 'ImputeMissing' == method:
        ImputeMissing(method=args['method']).fit(x).impute(x)

def run(train, valid, test, methods, var_procedure):

    for c in var_procedure:
        cleaning_steps = var_procedures['cleaning']
        feature_engineering_steps = var_procedures['feature_engineering']
        for method, args for cleaning_steps.items(): # Loop through cleaning procedures
            func = cleaner[method]
                  
if __name__ == '__main__':
    
    methods = {'ImputeMissing': ImputeMissing}

    """
    {'train': df_train, 'valid': df_valid, 'test': df_test, 
     'procedure': {'variable': 
                     {'cleaning': 
                         {"ImputeMissing": {Arguments for ImputeMissing Object}}
                      'feature_engineering': {}}
                     }
                }
    """ 
    
    
    
    