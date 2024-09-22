import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hmmlearn import hmm
from _1_pre_process import plot_energy_data, show_plots
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from bokeh.plotting import figure, show

import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
from bokeh.util.warnings import BokehDeprecationWarning
warnings.filterwarnings("ignore", category=BokehDeprecationWarning)

# model 1 - Searching for multiplier values using the percentiles of energy consumption highly correlated with air temperature
def run_markov(df_, n_components=2):
    np.random.seed(42)

    energy_data = np.array(df_['kw_total']).reshape(-1, 1)
    # print('data shape', energy_data.shape)

    # Defining an HMM
    # The number of components corresponds to the number of power states you expect
    model = hmm.GaussianHMM(n_components=n_components, covariance_type="full", n_iter=100)

    # Training the model
    model.fit(energy_data)

    # Predicting states
    hidden_states = model.predict(energy_data)

    # Initialize columns for the masks
    for state in np.unique(hidden_states):
        df_[f'mask{state}'] = np.nan  # Initialize with NaN

    # Applying 'kw_total' values based on states
    for state in np.unique(hidden_states):
        mask = hidden_states == state
        df_.loc[mask, f'mask{state}'] = df_.loc[mask, 'kw_total']

    # Plotting the data
    plot_energy_data(df_, ['mask0', 'mask1'], title='Markov', plot_type='scatter')

    return hidden_states
def return_markov_dfs(file_path):
    # Load data from CSV
    data = pd.read_csv(file_path)
    # Convert 'timestamp' column to datetime
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    # Extract date, hour, weekday, and season from the timestamp
    data['date'] = data['timestamp'].dt.date
    data['hour'] = data['timestamp'].dt.hour
    data['weekday'] = data['timestamp'].dt.weekday
    data['season'] = data['timestamp'].dt.month % 12 // 3 + 1

    # Run the Markov model function
    hidden_states = run_markov(data, n_components=2)

    # Masks for occupied and non-occupied states
    mask_occupied = hidden_states == 1

    # DataFrames for weekdays (assuming 'occupied' implies weekdays)
    df_weekdays = data[data.index.isin(data.index[mask_occupied])]
    df_weekends = data[~data.index.isin(data.index[mask_occupied])]

    # Columns to plot
    columns_to_plot = ['kw_total']
    # Plotting the data
    plot_energy_data(df_weekdays, columns_to_plot, title='df_weekdays')
    plot_energy_data(df_weekends, columns_to_plot, title='df_weekends')

    return df_weekdays, df_weekends
def calculate_multiplier(path_09, upper_quantile=0.95, lower_quantile=0.05, correlation_threshold=0.7, name='L09'):
    weekend_df, weekday_df = return_markov_dfs(path_09)
    weekend_df_09 = weekend_df if weekend_df['kw_total'].mean() > weekday_df['kw_total'].mean() else weekday_df
    multipliers = []
    for hour in range(24):
        for season in range(1, 5):
            for year in weekend_df_09['year'].unique():
                filtered_data = weekend_df_09[(weekend_df_09['hour'] == hour) & (weekend_df_09['season'] == season) & (weekend_df_09['year'] == year)].copy()
                # Correlation with temperature
                correlation = filtered_data['kw_total'].corr(filtered_data['air_temperature_at_2m(deg_C)'])
                if len(filtered_data) < 20 or abs(correlation) < correlation_threshold:
                    continue

                upper_quantile_value = filtered_data['kw_total'].quantile(upper_quantile)
                filtered_data['upper_quantile'] = upper_quantile_value
                
                lower_quantile_value = filtered_data['kw_total'].quantile(lower_quantile)
                filtered_data['lower_quantile'] = lower_quantile_value
                
                multiplier = 1 - ((upper_quantile_value - lower_quantile_value) / upper_quantile_value)
                multipliers.append(multiplier)
                
                # Calculating mean and standard deviation of 'kw_total'
                mean_kw_total = filtered_data['kw_total'].mean()
                std_kw_total = filtered_data['kw_total'].std()

                # Normalizing 'air_temperature_at_2m(deg_C)' to have the same mean and standard deviation as 'kw_total'
                filtered_data['air_temperature_normalized'] = (
                    (filtered_data['air_temperature_at_2m(deg_C)'] - filtered_data['air_temperature_at_2m(deg_C)'].mean()) / filtered_data['air_temperature_at_2m(deg_C)'].std()
                ) * std_kw_total + mean_kw_total
                
                plot_energy_data(filtered_data, ['kw_total', 'upper_quantile', 'lower_quantile', 'air_temperature_normalized'], title=f'h:{hour} s:{season} corr:{correlation} amp:{(upper_quantile_value - lower_quantile_value)/upper_quantile_value}')

    print(f'Plug Global Power {name}', np.mean(multipliers), np.std(multipliers), np.min(multipliers), np.max(multipliers))
    show_plots(ncols=4)
    return np.mean(multipliers), np.std(multipliers)

path_09 = 'D:/repositorios/REP_ADRENALIN/Submission 1/data/L09.B01_1H_cleaned.csv'
calculate_multiplier(path_09, upper_quantile=0.9, lower_quantile=0.1, correlation_threshold=0.7, name='L09')
path_10 = 'D:/repositorios/REP_ADRENALIN/Submission 1/data/L10.B01_1H_cleaned.csv'
calculate_multiplier(path_10, upper_quantile=0.99, lower_quantile=0.01, correlation_threshold=0.85, name='L10')

# model 2 searches for peaks and troughs to try to determine the values of plugs and lighting
def find_plugs_in_valleys_l9():
    path_09 = 'D:/repositorios/REP_ADRENALIN/Submission 1/data/L09.B01_5min_cleaned.csv'
    data = pd.read_csv(path_09)
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    # Filter data between '01-03-2021' and '01-09-2021'
    start_date = '2021-03-01'
    end_date = '2021-09-01'
    data = data.loc[(data['timestamp'] > start_date) & (data['timestamp'] < end_date)].reset_index(drop=True)

    hidden_states = run_markov(data, n_components=2)

    mask_non_occupied = hidden_states == 0
    mask_occupied = hidden_states == 1

    data['weekday_cluster'] = 0
    data.loc[mask_occupied, 'weekday_cluster'] = 1

    # Step 3: Filter the original DataFrame using these dates
    weekday_df = data[data.index.isin(data.index[mask_occupied])]
    weekend_df = data[~data.index.isin(data.index[mask_occupied])]

    data = weekend_df.copy()
    data = data.reset_index(drop=True)
    x = data['kw_total']
    x = -x  # Inverting the signal
    peaks, properties = find_peaks(x, height=(-80, -10), prominence=(60, 150), width=(0, 4))
    data['is_peak'] = 0
    data.loc[peaks, 'is_peak'] = 1

    # Creating the figure
    p = figure(title="Detection of Valleys with Specific Prominence and Width", x_axis_label='Index', y_axis_label='kw_total', width=800, height=400)

    # Adding the signal line
    p.line(x.index, x, legend_label="kw_total", line_width=2)

    p.circle(peaks, x[peaks], size=10, color="blue", legend_label="peaks", fill_alpha=0.6)

    # Adding a gray horizontal line at y=0
    p.line(x.index, np.zeros_like(x), line_dash="dashed", color="gray")

    # Adjusting the legend
    p.legend.location = "top_left"

    # Showing the graph
    show(p)

    # Analyze features of the peaks
    data.loc[peaks, 'peak_height'] = data['kw_total'][peaks]
    data.loc[peaks, 'peak_prominence'] = properties['prominences']
    data.loc[peaks, 'peak_width'] = properties['widths']
    data.loc[peaks, 'peak_base'] = properties['peak_heights'] - properties['prominences']
    properties['plugs'] = properties['prominences'] / ((properties['peak_heights'] - properties['prominences']) * -1)
    data.loc[peaks, 'plugs'] = properties['plugs']
    # Print average and median of plugs
    # print(f'average of plugs l09: {np.mean(properties["plugs"])}') 
    print(f'median of plugs l09: {np.median(properties["plugs"])}')
def find_plugs_in_peaks_l10():
    path10 = 'D:/repositorios/REP_ADRENALIN/Submission 1/data/L10.B01_15min_cleaned.csv'
    data = pd.read_csv(path10)
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    # Filter data between '01-03-2021' and '01-09-2021'
    start_date = '2019-03-01'
    end_date = '2023-09-01'
    data = data.loc[(data['timestamp'] > start_date) & (data['timestamp'] < end_date)].reset_index(drop=True)
    # Suppose x is your time series
    x = data['kw_total']

    # Finding peaks with relative prominence
    peaks, properties = find_peaks(x, height=(50, 200), prominence=(55, 100), width=(0, 7))

    # Add information about peaks to DataFrame
    data['is_peak'] = 0
    data.loc[peaks, 'is_peak'] = 1

    # Analyze features of the peaks
    data.loc[peaks, 'peak_height'] = data['kw_total'][peaks]
    data.loc[peaks, 'peak_prominence'] = properties['prominences']
    data.loc[peaks, 'peak_width'] = properties['widths']
    data.loc[peaks, 'peak_base'] = properties['peak_heights'] - properties['prominences']
    properties['plugs'] = properties['prominences'] / (properties['peak_heights'] - properties['prominences'])
    data.loc[peaks, 'plugs'] = properties['plugs']
    # Print average and median of plugs
    # print(f'average of plugs l10: {np.mean(properties["plugs"])}') 
    print(f'median of plugs l10: {np.median(properties["plugs"])}') 

    # Creating the figure
    p = figure(title="Detection of Peaks with Specific Prominence and Width", x_axis_label='Index', y_axis_label='kw_total', width=800, height=400)

    # Adding the signal line
    p.line(x.index, x, legend_label="kw_total", line_width=2)

    # Adding the detected peaks
    p.circle(peaks, x[peaks], size=10, color="blue", legend_label="peaks", fill_alpha=0.6)

    # Adding a gray horizontal line at y=0
    p.line(x.index, np.zeros_like(x), line_dash="dashed", color="gray")

    # Adjusting the legend
    p.legend.location = "top_left"

    # Showing the graph
    show(p)

find_plugs_in_valleys_l9()
find_plugs_in_peaks_l10()