import pandas as pd
import numpy as np
import psutil

def pd_reduce_size(df, dtypes_mapping={}):
    """Reduces the memory size of dataframe"""
    for col, dtype in df.dtypes.items():
        if col in dtypes_mapping:
            dtype = dtypes_mapping[col]
            df[col] = df[col].astype(dtype)
            print(col, df[col].dtypes)
        else: 
            if dtype == 'int':
                df[col] = df[col].astype('int32')
            elif dtype == 'float':
                df[col] = df[col].astype('float32')
    return df
