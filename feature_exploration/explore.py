import glob
import pickle
import time
import json
import pandas as pd

def variable_exploration_print(series):
    """ Prints variable information """

    def contains_string(x):
        try: 
            [int(v) for v in x.unique()]
        except:
            return True
        return False
    
    string_details = ("""Variable name: {}\n""".format(series.name) +
                      """Variable dtypes: {}\n""".format(series.dtypes) +
                      """Variable null rate: {}\n""".format(series.isnull().mean()) + 
                      """Variable value counts: {}\n""".format(series.value_counts().iloc[:10].to_dict()) + 
                      """Variable number of unique values: {}\n""".format(series.nunique()) + 
                      """Variable contains string: {}\n\n""".format(contains_string(series)) 
                     )
    return string_details

def variable_exploration_dict(series):
    """ Prints variable information """

    def contains_string(x):
        try: 
            [int(v) for v in x.unique()]
        except:
            return True
        return False
    
    dict_variable_details = ({'variable': series.name,
                              'dtypes': series.dtypes, 
                              'null_rate': series.isnull().mean(),
                              'value_counts': series.value_counts().iloc[:10].to_dict(),
                              'nunique': series.nunique(),
                              'contains_string': contains_string(series)})
    return dict_variable_details