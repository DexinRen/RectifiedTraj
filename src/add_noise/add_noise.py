"""
Example input:
                  timestamp  agent  longitude   latitude
0 2019-07-10 21:15:00+00:00  11429 -84.404480  33.729288
1 2019-07-10 21:15:00+00:00  11430 -84.378287  33.738387
2 2019-07-10 21:15:00+00:00  11431 -84.374524  33.743824
3 2019-07-10 21:15:00+00:00  11432 -84.375505  33.736158
4 2019-07-10 21:15:00+00:00  11433 -84.399542  33.736723
"""
import numpy as np
import pandas as pd
import multiprocessing as mp

def add_correlated_noise(latitudes, longitudes, alpha=0.1, sigma=0.0001):
    """
    Adds realistic correlated (OU process) noise to GPS traces.
    
    alpha: Controls how fast the error corrects (drift factor).
    sigma: Magnitude of the noise (approx degrees).
    """
    n = len(latitudes)
    # Initialize noise vectors
    lat_noise = np.zeros(n)
    lon_noise = np.zeros(n)
    
    # Generate white noise components
    white_lat = np.random.normal(0, sigma, n)
    white_lon = np.random.normal(0, sigma, n)
    
    # Apply Ornstein-Uhlenbeck process (Drift)
    for t in range(1, n):
        lat_noise[t] = lat_noise[t-1] * (1 - alpha) + white_lat[t]
        lon_noise[t] = lon_noise[t-1] * (1 - alpha) + white_lon[t]
        
    return latitudes + lat_noise, longitudes + lon_noise

if __name__ == "__main__":
    num_workers = mp.cpu_count() // 4 * 3
    data_root = "/storage/datasets_public/patterns_of_life/"
    
    # Read df; sorted by agents already
    df = pd.read_parquet(data_root + 'agent_traces_sorted.parquet')
    
    # Split job by agent
    agent_dfs = [group for name, group in df.groupby('agent')]
    with mp.Pool(processes=num_workers) as pool:
        results = pool.starmap(add_correlated_noise, [(agent_df['latitude'], agent_df['longitude']) for agent_df in agent_dfs])
    for i, (agent_df, (lat_noisy, lon_noisy)) in enumerate(zip(agent_dfs, results)):
        agent_dfs[i]['latitude_n'] = lat_noisy
        agent_dfs[i]['longitude_n'] = lon_noisy
    df = pd.concat(agent_dfs)
    # Save the noisy data
    df.to_parquet(data_root + 'agent_traces_with_noise.parquet', index=False)
    