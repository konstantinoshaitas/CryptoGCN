import pandas as pd
import os

def aggregate_asset_data(csv_files, output_file):
    """
    Aggregates closing prices from multiple asset CSV files into a single CSV file,
    only including the overlapping date ranges.

    :param csv_files: List of file paths to the individual asset CSV files.
    :param output_file: Path to the output CSV file.
    """
    data_frames = []
    date_ranges = []

    for file in csv_files:
        # Read each CSV file
        df = pd.read_csv(file)
        df.columns = df.columns.str.lower()  # Ensure all column names are lowercase
        df['time'] = pd.to_datetime(df['time'], utc=True)  # Ensure the 'time' column is in datetime format with UTC

        # Extract the asset name from the file name (or set a unique name for the asset)
        asset_name = os.path.basename(file).split('.')[0]

        # Use the 'time' column as the index
        df.set_index('time', inplace=True)

        # Extract the 'close' price and rename it to the asset name
        df = df[['close']].rename(columns={'close': asset_name})

        data_frames.append(df)
        date_ranges.append((df.index.min(), df.index.max()))

    # Find the common date range
    common_start_date = max([start for start, end in date_ranges])
    common_end_date = min([end for start, end in date_ranges])

    # Filter data frames to the common date range
    filtered_data_frames = [df.loc[common_start_date:common_end_date] for df in data_frames]

    # Merge the filtered data frames
    aggregated_data = pd.concat(filtered_data_frames, axis=1, join='inner')

    # Reset the index to make 'time' a column again
    aggregated_data.reset_index(inplace=True)

    # Save the aggregated data to the output file
    aggregated_data.to_csv(output_file, index=False)

# Example usage
csv_files = [
    r'C:\Users\koko\Desktop\THESIS\CryptoGCN\data\INDEX_BTCUSD, 480_8318d.csv',
    r'C:\Users\koko\Desktop\THESIS\CryptoGCN\data\CRYPTO_SOLUSD, 480_d942c.csv',
    r'C:\Users\koko\Desktop\THESIS\CryptoGCN\data\INDEX_ETHUSD, 480_ca04d.csv',
    r'C:\Users\koko\Desktop\THESIS\CryptoGCN\data\CRYPTO_AAVEUSD, 480_a7ede.csv',
    r'C:\Users\koko\Desktop\THESIS\CryptoGCN\data\CRYPTO_AVAXUSD, 480_978b1.csv'
    # Add paths to other crypto asset CSV files here
]
output_file = r'C:\Users\koko\Desktop\THESIS\CryptoGCN\data\aggregated_asset_data.csv'
aggregate_asset_data(csv_files, output_file)
