import glob
import pickle
import time
import json
import pandas as pd

def sets_intersection(l):
    from functools import reduce
    return list(reduce(set.intersection, l))

def concatenate_tables(dfs, common_vars=[]):
    if not common_vars:
        col_sets = [set(df.columns.tolist()) for df in dfs]
        common_vars = sets_intersection(col_sets)
    df_ids = dfs[0][common_vars]
    return pd.concat([df_ids]+[df.drop(df_ids, axis=1) for df in dfs], axis=1) 

def read_multi_csv(regex):
    from glob import glob
    filepaths = glob(regex)
    return list(map(pd.read_csv, filepaths))

def split_data(df, train_frac, valid_frac):
    train = df.sample(frac=train_frac, replace=False) 
    valid = df[~df.index.isin(train.index)].sample(frac=valid_frac / (1-train_frac), replace=False)
    test = df[~df.index.isin(train.index) & ~df.index.isin(valid.index)]
    return train, valid, test

def read_json(filepath):
    with open(filepath, 'r') as file:
        return json.load(file)
    
def dump_json(filepath, obj):
    """ Dumps a dictionary object into a designated json filepath """
    with open(filepath, 'w') as file:
        return json.dump(obj, file)

def pd_read_csv(filepath, args={}, report=True):
    """ Shows time and shape of data loaded """
    if report:
        filename = filepath.split('/')[-1]
        print('Reading-in {0}...'.format(filename))
        start_time = time.time()
        file = pd.read_csv(filepath, **args)
        print('File reading completed. Took %s seconds' % round(time.time() - start_time, 4))
        print('File shape', file.shape)
        return file
    return pd.read_csv(filepath, **args)
    
def pd_write_csv():
    pass

def chunk_list(splits, l):
    """ Splits list into a list of lists. """
    size = int(len(l) / splits) + 1
    return [l[i*size:size*i+size] for i in range(splits)]

def pickle_save(filepath, obj):
    """ Save Python object in a Pickle file. """
    with open(filepath, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
def pickle_load(filepath): 
    """ Load Python object from a Pickle file. """
    with open(filepath, 'rb') as handle:
        return pickle.load(handle)

def read_multi_csv(regex):
    """ Read-in multiple files and returns a list of Pandas Dataframe """
    filepaths = glob.glob(regex)
    return list(map(read_csv, filepaths))

def write_multi_csv(splits, df, keys=[], path='', compression='gzip'):
    """ Splits dataframe column-wise and outputs each chunk into csv """
    fields = [c for c in df.columns if c not in keys]
    size = int(len(fields) / splits) + 1
    for i in range(splits):
        sets = fields[i*size:size*i+size]
        filepath = path+'-{0}.csv.gz'.format(i)
        df[keys+sets].to_csv(filepath, compression=compression, index=False)

def pd_isin(df, l, c):
    """ Selects rows in dataframe based on values in list """
    return df[df[c].isin(l)]

def pd_notin(df, l, c):
    """ De-selects rows in dataframe based on values in list """
    return df[~df[c].isin(l)]

def date_range(df, date):
    """ Returns a tuple of date range in dataframe """
    df[date] = pd.to_datetime(df[date])
    return df[date].min(), df[date].max()

def intersection(x,y):
    """ Returns a list of common elements between two lists """
    return list(set(x).intersection(y))

def get_dummies_from_list(series, sep=None, fillna=True):
    """ Splits a column of list and creates dummy variables 
        for each value in list. If series is string, specify
        a separator to create a list form.
        
        E.G.:
        
        s1 = pd.Series(['a b', 'c', 'd'])
        ssplit = get_dummies_from_list(s1, sep=' ')
        
        s1 output:
            a | b | c | d 
            1   1   0   0
            0   0   1   0
            0   0   0   1
    """
    series_is_string = True if isinstance(series[0], (str)) else False
    if series_is_string and sep == None:
        raise ValueError('Series must be list. Specify separator to create list first.')
    
    # Handle missing values
    if fillna:
        series = series.fillna('missing')
    else:
        series = series.fillna([])
    
    # Apply separator if the column is string
    if sep:
        series = series.str.split(sep)
    return pd.get_dummies(series.apply(pd.Series).stack()).sum(level=0)