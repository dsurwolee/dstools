# Parallel Pipeline Processing

from multiprocessing import Pool
from pandasql import sqldf
import pandas as pd
import numpy as np

num_cores = 30 #number of cores on your machine

def parallelize_aggregation(df, func, aggparam={}):
    key = aggparam['key']
    unique_keys = df[key].unique()
    pool = Pool(num_cores)
    print('[INFO] Starting parallel Panda SQL process')
    results = []
    for i, k in enumerate(unique_keys):
        payload = {'pid': i,
                   'data': df[df[key] == k],
                   'windows': aggparam['windows'],
                   'key': key,
                   'time_key': aggparam['time_key'],
                   'fields': aggparam['fields']}
        results.append(pool.apply(func, args=(payload,)))
    print('[INFO] Concatenating chunks.')
    df = pd.concat(results)
    print('[INFO] Concatenation complete.')
    pool.close()
    pool.join()
    return df
          
def generate_query(key='', time_key='', w="", fields=[], w_dict={'1hr': '3600', '1day': '86400', '7day': '604800'}):
    agg_fields = ", ".join('count(distinct b.{1}) as num_distinct_{1}_per_{0}_last_{2}'
                            .format(key, f, w) for f in fields)

    query = """select a.{0}, a.{1},
            {2}
            from data a left join data b on a.{0} = b.{0} and a.{1} >= b.{1}
            where (a.{1} - b.{1}) <= {3}
            group by a.{0}, a.{1}
            order by a.{1} asc""".format(key, time_key, agg_fields, w_dict[w])
    
    return query
        
def aggregate_per_key(payload):
    
    print('[INFO] Process ID {} started...'.format(payload['pid']))
    aggs = []
    data = payload['data']
    for window in payload['windows']:
        query = generate_query(key=payload['key'],
                               time_key=payload['time_key'],
                               fields=payload['fields'],
                               w=window)
        aggs.append(sqldf(query, locals()))
    print('[INFO] Process ID {} Ended...'.format(payload['pid']))
    merged = pd.concat(aggs, axis=1)
    _, i = np.unique(merged.columns, return_index=True)
    return merged.iloc[:,i]


# Aggregations at ip level
agg_params = {'windows': ['1hr','1day','7day'],
              'key': 'ip_address',
              'time_key':  '__time',
              'fields': ['device_id','local_ip_address','customer_session_id','customer_user_id']}
columns_ips = ['ip_address','__time','device_id','local_ip_address','customer_session_id','customer_user_id']
sorted_df_ip = merged_df[columns_ips].sort_values(by=['ip_address','__time'], ascending=[1,1])
df_ip_mp = parallelize_aggregation(sorted_df_ip, aggregate_per_key, agg_params)

# Aggregations at device level
agg_params = {'windows': ['1hr','1day','7day'],
              'key': 'device_id',
              'time_key':  '__time',
              'fields': ['ip_address','local_ip_address','customer_session_id','customer_user_id']}

columns_devices = ['device_id','__time','ip_address','local_ip_address','customer_session_id','customer_user_id']
sorted_df_device = merged_df[columns_devices].sort_values(by=['device_id','__time'], ascending=[1,1])
df_device_mp = parallelize_aggregation(sorted_df_device, aggregate_per_key, agg_params)
df_device_mp.to_csv('/home/daniel/device_risk_score/data/poc_data_pipeline_device.csv',index=False)