import pandas as pd

def extract_first_week_data(file_path, output_file='first_week_data.csv'):
    """
    Load dataset, rename 'Unnamed: 0' to 'Date-Time' if necessary,
    and extract data for the first week of October 2024.
    
    Parameters:
        file_path (str): Path to the input CSV file.
        output_file (str): Path to save the filtered data.
    
    Returns:
        None
    """
    # Load dataset with 'Date-Time' as strings
    df = pd.read_csv(file_path, dtype={'Unnamed: 0': str})
    
    # Rename 'Unnamed: 0' to 'Date-Time' if not done yet
    if 'Unnamed: 0' in df.columns:
        df.rename(columns={'Unnamed: 0': 'Date-Time'}, inplace=True)

    # Attempt to convert 'Date-Time' to datetime format
    # Checking the format of the first non-empty entry
    sample_entry = df['Date-Time'].dropna().iloc[0]
    if '/' in sample_entry:  # This suggests the format is likely %d/%m/%Y
        df['Date-Time'] = pd.to_datetime(df['Date-Time'].str.strip(), format='%d/%m/%Y %H:%M', errors='coerce')
    else:  # Assume the format is likely %Y-%m-%d
        df['Date-Time'] = pd.to_datetime(df['Date-Time'].str.strip(), format='%Y-%m-%d %H:%M:%S', errors='coerce')
    
    # Check for any conversion errors
    if df['Date-Time'].isnull().any():
        print("Conversion issues with 'Date-Time' detected. Rows with NaT:")
        # print(df[df['Date-Time'].isnull()])
        # # Additionally, print the original 'Date-Time' entries that caused NaT
        # print("Original 'Date-Time' entries causing issues:")
        # problematic_rows = df[df['Date-Time'].isnull()]['Date-Time']
        # print(problematic_rows)

    # Filter for the first week of October 2024 (October 1 to October 7)

    # for column in df.columns:
    #     if df[column].isnull().any():
    #         print("Nan's detected in aggregations of column ", column, " of csv", file_path)

    first_week_data_list = []

    # drop no numerical columns as they cannoit be aggregated
    for column in df.columns:
        if df[column].dtype == 'object':
            df = df.drop(column, axis=1)

    # for each aggregate the numerical columns over half an hour length
    # for ii in range(7):
    #     first_week_data_day = df[(df['Date-Time'] >= f'2024-10-0{ii+1}') & (df['Date-Time'] < f'2024-10-0{ii+2}')]
    #     first_week_data_day = first_week_data_day.set_index('Date-Time').resample('30Min',closed = 'right',label ='right').mean()
    #     first_week_data_list.append(first_week_data_day)
    for ii in range(7):
        first_week_data = df[(df['Date-Time'] >= f'2024-10-0{ii+1}') & (df['Date-Time'] < f'2024-10-0{ii+2}')]
        first_week_data = first_week_data.set_index('Date-Time').resample('30Min', closed='right', label='right').mean()
        
        # Forward fill to fill missing values with the last valid observation
        first_week_data = first_week_data.ffill()  # Fill forward
    
        first_week_data_list.append(first_week_data)
        

    first_week_data = pd.concat(first_week_data_list)

    # for column in first_week_data.columns:
    #     if first_week_data[column].isnull().any():
    #         print("Nan's detected in aggregations of column ", column, " of csv", file_path)

    # Check if any data was found
    if not first_week_data.empty:
        # Save to output file
        first_week_data.to_csv(output_file, index=True)
        print(f"Data for the first week of October 2024 saved to '{output_file}'")
    else:
        print("No data found for the first week of October 2024.")

extract_first_week_data('data/channel_A_schedule.csv', 'data/AGGREGATE_FIRST_WEEK_channel_A_schedule.csv')
extract_first_week_data('data/channel_0_schedule.csv', 'data/AGGREGATE_FIRST_WEEK_channel_0_schedule.csv')
extract_first_week_data('data/channel_1_schedule.csv', 'data/AGGREGATE_FIRST_WEEK_channel_1_schedule.csv')
extract_first_week_data('data/channel_2_schedule.csv', 'data/AGGREGATE_FIRST_WEEK_channel_2_schedule.csv')
extract_first_week_data('data/channel_0_conversion_rates.csv', 'data/AGGREGATE_FIRST_WEEK_channel_0_conversion_rates.csv')
extract_first_week_data('data/channel_1_conversion_rates.csv', 'data/AGGREGATE_FIRST_WEEK_channel_1_conversion_rates.csv')
extract_first_week_data('data/channel_2_conversion_rates.csv', 'data/AGGREGATE_FIRST_WEEK_channel_2_conversion_rates.csv')