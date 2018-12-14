import pandas as pd
import numpy as np

class PreprocessText:
    
    def __init__(self, config={}):
        
        # Store values
        self.config = config
        self.info = {}
        self.mapping = {}
        
    def check_regex(self, pattern):
        if isinstance(pattern, (str)):
            return True
        return False
        
    def replace_with_value(self, x, regex, value='', na_value='missing'):
        if x.isnull().any(): # Check if null exists. If so, replace. 
            x = x.fillna(na_value)
        return x.str.replace(regex, value)
    
    def replace_empty(self, x, na_value='missing'):
        return x.map(lambda x: na_value if len(x) == 0 else x)
        
    def lower_case(self, x):
        return x.str.lower()

    def preprocess(self, x):
        """ 
            x (Series): a single Pandas Series 
        """
        
        methods = {
            'replace_value': self.replace_with_value,
            'lower_case': self.lower_case,
            'remove_special_chars': self.replace_with_value,
            'remove_stopwords': self.replace_with_value,
            'remove_extra_spaces': self.replace_with_value,
            'replace_empty': self.replace_empty
        }
        
        raw = x.copy().reset_index().drop('index',axis=1)
        self.info['raw'] = x.nunique()
        for c in self.config:
            cleaner = methods[c]
            config = self.config[c]
            if self.check_regex(config): # Check if the method requires regex pattern 
                x = cleaner(x=x, regex=config)
            else:
                x = cleaner(x=x)
            self.info[c] = x.nunique()
            
        # Create mapping between raw value and cleaned version
        raw.columns = ['raw']
        raw['cleaned'] = x
        self.mapping = raw.drop_duplicates().set_index('raw').to_dict()['cleaned']
        return x