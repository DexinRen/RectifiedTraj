import os
from concurrent.futures import ProcessPoolExecutor  # Changed from ThreadPoolExecutor
from tqdm import tqdm
import pandas as pd
# import geopandas as gpd  <-- Removed to prevent GDAL threading conflicts
from pyproj import Transformer

# Ensure this function is at the top level so it can be pickled
def write_agent_data(args):
    """Worker function to write a single agent's data to CSV file."""
    agent_id, agent_df, output_dir = args
    agent_file = os.path.join(output_dir, f"agent_{agent_id}.csv")
    
    # Check if file exists to determine write mode
    file_exists = os.path.exists(agent_file)
    write_header = not file_exists
    mode = 'a' if file_exists else 'w'
    
    agent_df.to_csv(agent_file, mode=mode, index=False, header=write_header)
    return agent_id

def main():
    # Configuration
    input_dir = "/storage/datasets_public/patterns_of_life/logs/2026-02-05"
    output_dir = "/storage/datasets_public/patterns_of_life/by_agent/"
    
    # Use fewer workers to avoid Disk I/O thrashing since we are writing many small files
    num_workers = min(4, os.cpu_count() or 1)
    print(f"{num_workers} worker processes will be used.")

    os.makedirs(output_dir, exist_ok=True)

    col_names = {
        1: 'timestamp', 
        2: 'location', 
        3: 'agent',
    }
    usecols, names = zip(*col_names.items())

    # Initialize Transformer ONCE outside the loop (optimization)
    # always_xy=True ensures output is (Lon, Lat)
    utm_to_wgs84 = Transformer.from_crs("EPSG:32616", "EPSG:4326", always_xy=True)

    sorted_filenames = sorted(os.listdir(input_dir))
    
    for filename in tqdm(sorted_filenames, ncols=100, desc="Processing files"):
        file_path = os.path.join(input_dir, filename)
        
        # 1. Load Data
        df = pd.read_csv(file_path, sep="\t", header=None, usecols=usecols, names=names)

        # 2. Vectorized Regex Extraction (Faster)
        coords = df['location'].str.extract(r'POINT \((?P<x>[^ ]+) (?P<y>[^)]+)\)')
        
        # 3. Transform Coordinates
        # Note: Do this on the main process. Passing raw floats to workers is faster/safer.
        lons, lats = utm_to_wgs84.transform(coords['x'].values, coords['y'].values)
        df['longitude'] = lons
        df['latitude'] = lats
        df.drop(columns='location', inplace=True)

        # 4. Parse Dates
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True).astype('datetime64[ms, UTC]')

        # 5. Prepare Groups
        # Note: We convert groupby to a list here. If memory is tight, this is the bottleneck.
        agent_groups = [
            (agent_id, group, output_dir) 
            for agent_id, group in df.groupby('agent')
        ]
        
        # 6. Parallel Write using ProcessPoolExecutor
        # ProcessPoolExecutor is safer for segfault avoidance than ThreadPoolExecutor
        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            # We use list() to force execution and show the progress bar
            list(tqdm(pool.map(write_agent_data, agent_groups),
                      total=len(agent_groups),
                      ncols=100,
                      desc=f"  Writing agents from {filename}",
                      leave=False))

if __name__ == '__main__':
    main()