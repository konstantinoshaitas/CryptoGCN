import pandas as pd
import os


def merge_csv_files(csv_files, output_file):
    """
    Merges multiple CSV files for the same ticker in chronological order, avoiding duplicate times.

    :param csv_files: List of file paths to the individual CSV files for the same ticker.
    :param output_file: Path to the output CSV file.
    """
    combined_df = pd.DataFrame()

    for file in csv_files:
        # Read each CSV file
        df = pd.read_csv(file)
        df.columns = df.columns.str.lower()  # Ensure all column names are lowercase
        df['time'] = pd.to_datetime(df['time'])  # Ensure the 'time' column is in datetime format

        # Append to the combined DataFrame
        combined_df = pd.concat([combined_df, df])

    # Drop duplicate rows based on 'time' column, keeping the first occurrence
    combined_df.drop_duplicates(subset='time', keep='first', inplace=True)

    # Sort the DataFrame by 'time' column
    combined_df.sort_values(by='time', inplace=True)

    # Reset the index
    combined_df.reset_index(drop=True, inplace=True)

    # Save the combined data to the output file
    combined_df.to_csv(output_file, index=False)


# Example usage
csv_files = [
    r'C:\Users\koko\Desktop\THESIS\CryptoGCN\data\ticker_data_part1.csv',
    r'C:\Users\koko\Desktop\THESIS\CryptoGCN\data\ticker_data_part2.csv',
    # Add paths to other CSV files for the same ticker here
]
output_file = r'C:\Users\koko\Desktop\THESIS\CryptoGCN\data\merged_ticker_data.csv'
merge_csv_files(csv_files, output_file)
