import pandas as pd
import numpy as np
import psutil

def monitor_memory(df, before=True, description='', round_at=6):
    """Displays statistics on dataframe shape and size and server memory status"""
    from psutil import virtual_memory
    step = 'Before' if before else 'After'
    description = '-' if not description else description+' -'
    
    # Convert values to gigs
    convert_to_gig = lambda x: round(x / 1024**3, round_at)
    
    # Get statistics
    df_shape = df.shape
    df_memory = convert_to_gig(df.memory_usage(deep=True).sum())
    df_server_memory_available = convert_to_gig(virtual_memory().available)
    df_server_memory_used = convert_to_gig(virtual_memory().used)
    
    print('{0} {1} dataframe shape: {2}'.format(step, description, df_shape))
    print('{0} {1} dataframe memory size: {2} GBytes'.format(step, description, df_memory))
    print('{0} {1} server available memory: {2} GBytes'.format(step, description, df_server_memory_available))
    print('{0} {1} server used memory: {2} GBytes'.format(step, description, df_server_memory_used))