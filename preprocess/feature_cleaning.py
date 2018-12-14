import pandas as pd
import numpy as np
            
def replace_string_with_na(x):
    """Apply this function on continuous columns with object values
       This replaces string with nan value.
       
       :x (np.series): a single series of Pandas
       :returns: a cleaned Pandas Series
    """
    try:
        return float(x)
    except:
        return np.nan

