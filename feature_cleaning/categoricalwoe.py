import pandas as pd
import numpy as np

# Clean categorical variables
class CategoricalWoe:
    
    def __init__(self, WOE_LIMIT=10):
        self.WOE_UPPER = WOE_LIMIT
        self.WOE_LOWER = -WOE_LIMIT
        self.WOE = {}
        self.rare_values = {}
        self.iv = None
        
    def replace_null_values(self, column):
        if column.dtypes == 'object':
            return column.fillna('missing')
        return column.cat.add_categories("missing").fillna("missing")
    
    def replace_rare_values(self, column, rare_values):
        def replacer(x, rare_value):
            try:
                rare_value[x]
            except:
                return x
            return 'rare'
        return column.map(lambda x: replacer(x, rare_values))

    def replace_unseen_values(self, column, column_woe_dict): 
        return column.map(lambda x: 'rare' if x not in column_woe_dict else x)
    
    def replace_with_woe(self, column, column_woe_dict):
        return column.map(lambda x: column_woe_dict[x] if x in column_woe_dict else 0)
    
    def get_rare_values(self, X, thres):
        dist = X.value_counts(normalize=True)
        dict_dist = dist.to_dict()
        return {k: v for k, v in dict_dist.items() if v < thres}
            
    def get_woe(self, x, y):
        dist = pd.crosstab(x, y, normalize='columns')
        not_fraud = dist.iloc[:,0] 
        fraud = dist.iloc[:,1] 
        value_woe = np.log(not_fraud/fraud).clip(self.WOE_LOWER, self.WOE_UPPER)
        self.iv = ((not_fraud - fraud) * value_woe).sum()
        return value_woe.to_dict()
    
    def fit(self, X, y, thres=0.02):
        X_imputed = self.replace_null_values(X)
        self.rare_values = self.get_rare_values(X, thres)
        X_replaced_with_rare = self.replace_rare_values(X_imputed, self.rare_values)
        self.WOE = self.get_woe(X_replaced_with_rare, y)
    
    def replace(self, X):
        X = self.replace_null_values(X)
        X = self.replace_rare_values(X, self.rare_values)
        X = self.replace_unseen_values(X, column_woe_dict=self.WOE)
        X = self.replace_with_woe(X, column_woe_dict=self.WOE)
        return X