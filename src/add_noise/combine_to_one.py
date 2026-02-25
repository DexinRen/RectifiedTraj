import pandas as pd
import glob
import os

# Configuration
input_dir = '/storage/datasets_public/patterns_of_life/with_noise'
output_file = '/storage/datasets_public/patterns_of_life/agent_traces_with_noise_large.parquet'

# Get list of all .csv files
csv_files = glob.glob(os.path.join(input_dir, "*.csv"))

if not csv_files:
    print("No CSV files found in the directory.")
else:
    print(f"Found {len(csv_files)} files. Starting conversion...")

    # Initialize the parquet file with the first CSV to establish schema
    for i, file in enumerate(csv_files):
        print(f"Processing ({i+1}/{len(csv_files)}): {os.path.basename(file)}")
        
        # Read the CSV
        df = pd.read_csv(file)
        
        # On first iteration, create the file. On subsequent, append.
        # Note: 'fastparquet' or 'pyarrow' engines handle appending differently.
        if i == 0:
            df.to_parquet(output_file, engine='pyarrow', index=False)
        else:
            # For a single file, we append to the existing parquet
            df.to_parquet(output_file, engine='pyarrow', index=False, append=True)

    print(f"Successfully created {output_file}")
