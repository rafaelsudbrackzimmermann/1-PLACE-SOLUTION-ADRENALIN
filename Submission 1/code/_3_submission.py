import os
import pandas as pd
from _1_pre_process import plot_energy_data

def prepare_and_submit(data, data_id, real_id, min_limit, submission_path, plot_=False):
    """
    Prepares and submits the data subset for a specific real ID.
    Updates 'Temperature_dependent(kW)' and saves the submission file.

    Args:
    data (DataFrame): The main data frame.
    data_id (DataFrame): The DataFrame with ID and temperature dependent kW values.
    real_id (str): The real ID to filter the data.
    min_limit (float): Minimum limit multiplier for 'Temperature_dependent(kW)'.
    submission_path (str): The path to save the submission file.
    plot_ (bool): If True, plots the data.
    """
    # Prepare data subset
    subset = data[data['id_real'] == real_id].copy()
    subset['timestamp'] = pd.to_datetime(subset['timestamp'])
    data_id['timestamp'] = pd.to_datetime(data_id['timestamp'])
    subset['hour'] = subset['timestamp'].dt.hour
    subset['minute'] = subset['timestamp'].dt.minute
    subset['weekday'] = subset['timestamp'].dt.weekday
    subset['Temperature_dependent(kW)'] = subset['kw_total'] * min_limit
    
    # Merge with data_id to update 'Temperature_dependent(kW)' values
    temp_dependent_dict = data_id.set_index('id_real_total')['Temperature_dependent(kW)'].to_dict()
    subset['Temperature_dependent(kW)'] = subset['id_real_total'].map(temp_dependent_dict)

    subset = subset[['timestamp', 'Temperature_dependent(kW)']]

    # Set negative 'Temperature_dependent(kW)' to 0
    subset.loc[subset['Temperature_dependent(kW)'] < 0, 'Temperature_dependent(kW)'] = 0
    subset['Temperature_dependent(kW)'] = subset['Temperature_dependent(kW)'].fillna(0)
    
    # Prepare and save the submission file
    if plot_:
        plot_energy_data(subset, ['kw_total', 'Temperature_dependent(kW)'],
                         title=f'{real_id} {subset["Temperature_dependent(kW)"].mean()} - Temperature_dependent(kW) - Forecast Disaggregation - Submission')
            
    data_id = data_id.reset_index()
    data_group = data_id[['timestamp', 'Temperature_dependent(kW)', 'folder']]
    folder = data_group['folder'].values[0]
    file_name = real_id + '.csv'
    submission_file_path = os.path.join(submission_path, folder, file_name)
    
    # Create folder if it does not exist
    if not os.path.exists(os.path.join(submission_path, folder)):
        os.makedirs(os.path.join(submission_path, folder))
    subset.to_csv(submission_file_path, index=False)
    
    print(f'File saved to: {submission_file_path}')
    print(subset.shape, subset['Temperature_dependent(kW)'].isnull().sum(), subset['Temperature_dependent(kW)'].mean())
    print(data_id.shape, data_id['Temperature_dependent(kW)'].isnull().sum(), data_id['Temperature_dependent(kW)'].mean())


