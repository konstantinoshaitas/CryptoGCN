import os
import pandas as pd

def preprocess_data():
    # Get the current script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Define paths using relative paths
    data_dir = os.path.join(script_dir, '..', 'data')
    correlation_data_path = os.path.join(data_dir, 'correlation_data', 'aggregated_asset_data.csv')
    sequential_data_dir = os.path.join(data_dir, 'sequential_data')
    processed_data_dir = os.path.join(data_dir, 'processed_sequential_data')

    # Ensure the processed data directory exists
    os.makedirs(processed_data_dir, exist_ok=True)

    # Load the aggregated asset data CSV
    correlation_data = pd.read_csv(correlation_data_path, index_col=0)
    correlation_data.index = pd.to_datetime(correlation_data.index, utc=True)

    # Get the start and end dates from the correlation data
    start_date = correlation_data.index.min()
    end_date = correlation_data.index.max()

    print(f"Aligning data from {start_date} to {end_date} (UTC)")

    # Process each CSV in the sequential_data directory
    for filename in os.listdir(sequential_data_dir):
        if filename.endswith('.csv'):
            file_path = os.path.join(sequential_data_dir, filename)
            print(f"Processing {filename}...")

            # Load the CSV
            df = pd.read_csv(file_path, index_col=0)
            df.index = pd.to_datetime(df.index, utc=True)

            # Align the data to the correlation data date range
            aligned_df = df.loc[start_date:end_date]

            # Ensure the index is in UTC
            aligned_df.index = aligned_df.index.tz_convert('UTC')

            # Forward fill any missing values
            aligned_df = aligned_df.ffill()

            # Backward fill any remaining missing values at the start
            aligned_df = aligned_df.bfill()

            # Save the processed CSV
            output_path = os.path.join(processed_data_dir, f"processed_{filename}")
            aligned_df.to_csv(output_path)
            print(f"Saved processed file to {output_path}")

    print("Preprocessing complete!")

if __name__ == "__main__":
    preprocess_data()