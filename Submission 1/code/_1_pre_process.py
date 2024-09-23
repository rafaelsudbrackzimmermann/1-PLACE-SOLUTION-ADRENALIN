import numpy as np
import pandas as pd
import os
import random
import mplcursors
import matplotlib.pyplot as plt
from tabulate import tabulate
from bokeh.layouts import gridplot, column
from bokeh.plotting import figure, show
from bokeh.models import DatetimeTickFormatter, Div
from bokeh.palettes import Category10
from _0_parameters_test import parameters_dict
from metpy.units import units
from metpy.calc import apparent_temperature, heat_index, windchill, dewpoint_from_relative_humidity
import holidays

import warnings
warnings.filterwarnings('ignore', message='no explicit representation of timezones available for np.datetime64')
warnings.filterwarnings('ignore', message="Series.__getitem__ treating keys as positions is deprecated")
warnings.filterwarnings('ignore', message="'H' is deprecated and will be removed in a future version, please use 'h' instead")

# print datafram in tabular format
def print_t(df, num_rows=5, view_type=0, max_columns=10):
    """
    Prints a summary of the DataFrame with options for different display formats,
    adjusting the number of displayed columns if necessary.

    Args:
    df (DataFrame): The DataFrame to print.
    num_rows (int): Number of rows to display.
    view_type (int): Type of view, 0 for tabulate view, otherwise simple head view.
    max_columns (int): Maximum number of columns to display.
    """
    if isinstance(df, pd.DataFrame):
        # Adjust the number of columns to be displayed
        displayed_columns = df.columns[:max_columns]  # Selects the first 'max_columns' columns
        if view_type == 0:
            # Use tabulate to print the DataFrame in tabular form
            print(tabulate(df.iloc[:num_rows][displayed_columns], headers='keys', tablefmt='psql'))
        else:
            # Use the head method to view the first 'num_rows' rows
            print(df[displayed_columns].head(num_rows))
        print("Shape of DataFrame:", df.shape)  # Print the shape of the DataFrame
    else:
        # Simply print the scalar value if not a DataFrame
        print(df)

# Plots multiple time series of energy data.
list_of_plots = []
def plot_energy_data(data, columns, title="", width=450, height=350, show_plot=False, plot_type = 'line'):
    """
    Plots multiple time series of energy data.

    Args:
    data (DataFrame): DataFrame containing the data to be plotted.
    columns (list): List containing the names of the columns to plot.
    title (str, optional): Title of the plot.
    width (int, optional): Width of the plot.
    height (int, optional): Height of the plot.
    show_plot (bool, optional): If True, show the plot immediately. If False, store for later display.
    """
    # Initialize figure
    p = figure(title=title, x_axis_type='datetime', x_axis_label='Date and Time', y_axis_label='kW', width=width, height=height)
    # Set colors for each series
    colors = Category10[10] * (len(columns) // 10 + 1)

    # Plot each column
    for index, column in enumerate(columns):
        if column in data:
            # Customize appearance for specific series
            alpha, line_width = 0.5, 2
            if column == 'kw_total' or index == 0:
                alpha = 1  # More opaque for total kW
            elif column == 'Temperature_dependent(kW)':
                alpha = 0.7  # Slightly less opaque for temperature-dependent kW

            # Plot line
            # p.line(data['timestamp'], data[column], legend_label=column, color=colors[index], line_width=line_width, alpha=alpha)
            if plot_type == 'line':
                # Plot line
                p.line(data['timestamp'], data[column], legend_label=column, color=colors[index], line_width=line_width, alpha=alpha)
            elif plot_type == 'scatter':
                # Plot scatter
                p.scatter(data['timestamp'], data[column], legend_label=column, color=colors[index], size=2, alpha=alpha)

    # Customize legend and axis formatting
    p.legend.location = "top_left"
    p.legend.background_fill_alpha = 0.1
    p.legend.label_text_alpha = 0.7
    p.legend.border_line_alpha = 0.7
    p.xaxis.formatter = DatetimeTickFormatter(
        hours="%H:%M %d-%m-%Y",
        days="%d-%m-%Y",
        months="%m-%Y",
        years="%Y"
    )

    # Display or store the plot
    if show_plot:
        show(p)
    else:
        list_of_plots.append((p, title))
def show_plots(ncols=2):
    """
    Display all stored plots in a grid format.

    Args:
    ncols (int, optional): Number of columns in the grid layout.
    """
    if not list_of_plots:
        print("No plots to display.")
        return

    # Create a list of columns, each containing a plot and a title
    plot_columns = [column(Div(text=f"<h3 style='text-align:center'>{title}</h3>"), plot) for plot, title in list_of_plots]
    grid = gridplot(plot_columns, ncols=ncols)  # Organize columns in a grid
    show(grid)
    list_of_plots.clear()  # Clear the list of plots after displaying

# Function to create meteological indices
def calculate_meteorological_indices(df):
    # Convert the temperature, humidity, and wind speed columns to the correct units
    temperature = df['air_temperature_at_2m(deg_C)'].values * units.degC
    humidity = df['relative_humidity_at_2m(%)'].values * units.percent
    wind_speed = df['wind_speed_at_10m(km/h)'].values * units.km / units.hr
    
    # Calculate the heat index, wind chill, dew point, and apparent temperature
    df['heat_index'] = heat_index(temperature=temperature, relative_humidity=humidity)
    df['wind_chill'] = windchill(temperature=temperature, speed=wind_speed)
    df['dew_point'] = dewpoint_from_relative_humidity(temperature, humidity)
    df['apparent_temperature'] = apparent_temperature(temperature=temperature, speed=wind_speed,relative_humidity=humidity)

    created_columns = ['heat_index', 'wind_chill', 'dew_point', 'apparent_temperature']
    return df, created_columns

# Function to aggregate all data into a single dataframe and merge energy and weather data
def aggregate_data(train_data_path,
                  output_path):

    folders_kw = ['1h', '15min', '30min', '5min']
    folders_weather = 'weather'

    # load the energy data
    data_energy = pd.DataFrame()
    for folder in folders_kw:
        print(f'Processing Folder: {folder} -------------------------------------------------------------')
        folder_path = os.path.join(train_data_path, folder)
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            data = pd.read_csv(file_path)
            data = data.fillna(0)
            # concatenate the energy data into a df, using the file name as id
            print(f'Folder: {folder}, File: {file}')
            if folder == '1h':
                data['folder'] = '1H'
            else:
                data['folder'] = folder
           # get the string before the dot in file to use as id
            data['id'] = file.split('.')[0]
            data['building_id'] = file.split('.')[1].split('_')[0]
            data_energy = pd.concat([data_energy, data], axis=0)
            # create a real timestamp column, which takes the timestamp and normalizes it to always be full hours
            data_energy['timestamp_real'] = pd.to_datetime(data_energy['timestamp'])
            data_energy['timestamp_real'] = data_energy['timestamp_real'].dt.floor('h')
            

    data_energy = data_energy.fillna(0).reset_index(drop=True)
    data_energy['kw_total'] = data_energy['main_meter(kW)'] + data_energy['PV_battery_system(kW)']
    print_t(data_energy)    

    # load the weather data
    data_weather = pd.DataFrame()
    folder_path = os.path.join(train_data_path, folders_weather)
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        data = pd.read_csv(file_path)
        data = data.fillna(0)
        data['id'] = file.split('_')[0]
        
        data['timestamp_real'] = pd.to_datetime(data['timestamp'])
        data_weather = pd.concat([data_weather, data], axis=0)
        
    print_t(data_weather)

    # merge the energy and weather data using timestamp_real and id
    data_merged = pd.merge(data_energy, data_weather, on=['timestamp_real', 'id'], how='inner')
    print_t(data_merged)
    # check if any data was not merged
    print_t(data_merged['timestamp_real'].isnull().sum())
    print_t(data_merged['id'].isnull().sum())
    # save the file to a csv
    colunas = ['id_real', 'id', 'folder', 'building_id', 'id_real_total', 'timestamp', 'timestamp_number',
            'timestamp_real', 'timestamp_number_real', 'main_meter(kW)',	'PV_battery_system(kW)', 
            'kw_total', 'air_temperature_at_2m(deg_C)', 'relative_humidity_at_2m(%)', 
            'direct_solar_radiation(W/m^2)',  'diffuse_solar_radiation(W/m^2)', 'wind_speed_at_10m(km/h)', 'wind_direction_at_10m(deg)']
    data_merged = data_merged.rename(columns={'timestamp_x': 'timestamp'})
    # create a real id
    data_merged['id_real'] = data_merged['id'] + '.' + data_merged['building_id'] + '_' + data_merged['folder']
    data_merged['timestamp'] = pd.to_datetime(data_merged['timestamp'])
    data_merged['timestamp_number'] = data_merged['timestamp'].astype('int64') // 10**9
    data_merged['timestamp_number_real'] = data_merged['timestamp_real'].astype('int64') // 10**9
    data_merged['id_real_total'] = data_merged['id_real'] + '_' + data_merged['timestamp_number'].astype(str)
    data_merged = data_merged[colunas]
    data_merged['Temperature_dependent(kW)'] = np.nan
    data_merged.to_csv(f'{output_path}/data_merged.csv', index=False)

    print(f'File saved to: {output_path}/data_merged.csv ----------------------------------------------')
    print('DATA MERGED -------------------------------------------------------------------')
    print(data_merged['id_real'].value_counts())
    print_t(data_merged)

    return data_merged

# Verifies synchronization of time series data and fills gaps if any.
def check_sync_and_fill_gaps(data, date_column):
    """
    Verifies synchronization of time series data and fills gaps if any.
    """
    # Convert date column to datetime if not already
    data[date_column] = pd.to_datetime(data[date_column])
    initial_data_shape = data.shape
    
    # Set date column as index
    data.set_index(date_column, inplace=True)
    
    # Check and print duplicated values
    duplicated_indices = data.index[data.index.duplicated()]
    if not duplicated_indices.empty:
        print(f"Duplicated values found for the following timestamps:\n{duplicated_indices}")
    
    timeframe = data['folder'][0]
    print(f'Timeframe: {timeframe}')
    timeframe_to_freq = {
        '1h': 'H',
        '30min': '30T',
        '15min': '15T',
        '5min': '5T'
    }
    freq = timeframe_to_freq.get(timeframe, 'H')

    # Create a complete index of dates based on the total period of the data
    full_index = pd.date_range(start=data.index.min(), end=data.index.max(), freq=freq)
    
    # Reindex the DataFrame to ensure every time point is represented
    data = data.reindex(full_index)
    
    # Restore date index to original column
    data.reset_index(inplace=True)
    data.rename(columns={'index': date_column}, inplace=True)
    
    print('DATA REORDERING -------------------------------------------------------------------')
    print(f'Start: {initial_data_shape}, After Reordering: {data.shape}')
    
    return data
def load_dataframe(id_real, data_merged, week_mean_, day_mean_, plot=False):
    """
    Loads and preprocesses a DataFrame for a specific building by adding time-related features
    and meteorological indices, and filling gaps if necessary.
    """
    data = data_merged.copy()
    data = data[data['id_real'] == id_real].copy()

    # Convert the date column to datetime
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    if id_real == 'L09.B01_1H' or id_real == 'L09.B01_30min' or id_real == 'L09.B01_15min' or id_real == 'L09.B01_5min':
        # usar delta de 11 horas para ajustar o fuso horario
        data['timestamp'] = data['timestamp'] - pd.Timedelta(hours=11)
    data_original = data.copy().reset_index(drop=True)
    data = check_sync_and_fill_gaps(data, 'timestamp')
    
    # Derive time-based features
    data['hour'] = data['timestamp'].dt.hour
    data['minute'] = data['timestamp'].dt.minute
    data['weekday'] = data['timestamp'].dt.weekday
    data['season'] = data['timestamp'].dt.month % 12 // 3 + 1
    data['month'] = data['timestamp'].dt.month
    data['year'] = data['timestamp'].dt.year  
    data['week'] = data['timestamp'].dt.isocalendar().week
    data['date'] = data['timestamp'].dt.date
    data['day'] = data['timestamp'].dt.day
    data['day_of_year'] = data['timestamp'].dt.dayofyear

    # Create combined time-based features for aggregation purposes
    data['season_weekday_hour'] = data['hour']  + data['weekday']*24 + (data['season'] - 1) * 168    
    data['season_hour'] = data['hour'] + (data['season'] - 1) * 24
    data['month_hour'] = data['hour'] + (data['month'] - 1) * 24
    data['season_weekday'] = data['weekday'] + (data['season'] - 1) * 7
    data['weekday_hour'] = data['hour'] + data['weekday']*24
    
    # Calculate daily and weekly aggregation metrics
    data['day_mean'] = data.groupby(['date', 'id_real'])['kw_total'].transform('mean')
    data['week_mean'] = data.groupby(['year','week', 'id_real'])['kw_total'].transform('mean')
    data['week_min'] = data.groupby(['year', 'week', 'id_real'])['kw_total'].transform('min')
    data['week_max'] = data.groupby(['year', 'week', 'id_real'])['kw_total'].transform('max')
    data['week_mean_total'] = np.mean(data['week_mean']) * week_mean_
    data['day_mean_total'] = np.mean(data['day_mean']) * day_mean_
    data['day_min_rolling'] = data['week_mean'].rolling(window=168, min_periods=1, center=False).mean() *0.3
    
    # Calculate trigonometric time features
    data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
    data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
    data['day_sin'] = np.sin(2 * np.pi * data['weekday'] / 7)
    data['day_cos'] = np.cos(2 * np.pi * data['weekday'] / 7)
    data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
    data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
    data['season_sin'] = np.sin(2 * np.pi * data['season'] / 4)
    data['season_cos'] = np.cos(2 * np.pi * data['season'] / 4)

    # Calculate hourly means per month and season
    data['month_hour_mean'] = data.groupby(['year', 'month', 'hour', 'id_real'])['kw_total'].transform('mean')
    data['season_hour_mean'] = data.groupby(['year', 'season', 'hour', 'id_real'])['kw_total'].transform('mean')
    
    # Add meteorological indices and aggregate by day
    data, new_cols_temp = calculate_meteorological_indices(data)
    for col in new_cols_temp:
        data[f'{col}_day_mean'] = data.groupby(['date', 'id_real'])[col].transform('mean')
    data["Temperature_dependent(kW)"] = -1
    
    data_id = data.copy()
    
    # Check for null or infinite values in the dataset
    if data_id.isnull().values.any():
        print("Nulos no dataset")
        print(data_id.isnull().sum())
    if data_id.isin([np.nan, np.inf, -np.inf]).values.any():
        print("Infinitos no dataset")
        print(data_id.isin([np.nan, np.inf, -np.inf]).sum())
        
    # Optionally plot the processed data 
    if plot:
        plot_energy_data(data_id, ['kw_total', 'month_hour_mean', 'season_hour_mean'],
                         title=f"{id_real} Load Data Original com sincronização de gaps original: {data_original.shape} sinc_gaps: {data_id.shape}")
    
        print(data_id.columns)
        print_t(data_id, num_rows=3, view_type=0, max_columns=10)

    return data_id, data_original
def remove_intervals(data_id, intervalos, plot=False):
    """
    Sets the 'kw_total' column to NaN for specified intervals in the data.
    """
    for intervalo in intervalos:
        mes_inicio, dia_inicio = intervalo[0]
        mes_fim, dia_fim = intervalo[1]
        nome_intervalo = intervalo[2]
        
        if mes_inicio <= mes_fim:
            # Interval within the same year
            mask = ((data_id['month'] > mes_inicio) | 
                    ((data_id['month'] == mes_inicio) & (data_id['day'] >= dia_inicio))) & \
                   ((data_id['month'] < mes_fim) | 
                    ((data_id['month'] == mes_fim) & (data_id['day'] <= dia_fim)))
        else:
            # Interval that crosses the year boundary
            mask = ((data_id['month'] > mes_inicio) | 
                    ((data_id['month'] == mes_inicio) & (data_id['day'] >= dia_inicio))) | \
                   ((data_id['month'] < mes_fim) | 
                    ((data_id['month'] == mes_fim) & (data_id['day'] <= dia_fim)))
        
        # Instead of removing intervals, set 'kw_total' to NaN
        data_id.loc[mask, 'kw_total'] = np.nan
        # Optionally print information about the removal
        if plot:
            print(f"Removendo intervalo: {nome_intervalo} de {mes_inicio}-{dia_inicio} a {mes_fim}-{dia_fim}")
    
    return data_id
def remove_holidays(data, holidays, years, plot=False):
    """
    Sets 'kw_total' column to NaN for specified holidays in the data.
    """
    cleaned_data = data.copy()
    
    # Ensure the 'date' column is in datetime format
    cleaned_data['date'] = pd.to_datetime(cleaned_data['date'])
    
    # Create a list of all holiday dates for each year
    holiday_dates = []
    for year in years:
        for month, day, name in holidays:
            holiday_dates.append(pd.Timestamp(year=year, month=month, day=day))
            if plot:
                print(f"Removing {name}: Start date {cleaned_data['date'].iloc[0]}, Holiday: {holiday_dates[-1]}")
    
    # Display the number of dates to be removed
    print('Dates to be removed:', cleaned_data['date'].isin(holiday_dates).sum() / 24)
    print('Number of holidays:', len(holidays))

    # Instead of removing the holidays, set 'kw_total' to NaN
    cleaned_data.loc[cleaned_data['date'].isin(holiday_dates), 'kw_total'] = np.nan
    
    return cleaned_data
def remove_holidays_using_holidays(data, country_pt):
    """
    Sets 'kw_total' to NaN for official holidays in the specified country using the 'holidays' library.
    """
    # Mapping of Portuguese country names to country codes used by the 'holidays' library
    country_map = {
        'Noruega': 'Norway',
        'Dinamarca': 'Denmark',
        'Australia': 'Australia',
        '_': 'dont_do_anything',  # Placeholder for unsupported countries
    }

    # Retrieve the country code from the Portuguese name
    country_code = country_map.get(country_pt)
    if country_code == 'dont_do_anything':
        return data
    if not country_code:
        raise ValueError(f"Country '{country_pt}' not supported. Please use one of the following: {list(country_map.keys())}")

    # Create a holidays instance for the specified country covering the years in the data
    years = data['timestamp'].dt.year.unique()
    country_holidays = holidays.CountryHoliday(country_code, years=years)

    # Initialize an 'is_holiday' column as False
    data['is_holiday'] = False

    # Update 'is_holiday' to True on days that are official holidays
    data['is_holiday'] = data['timestamp'].dt.date.isin(country_holidays)

    print('Holidays found:', data['is_holiday'].sum())

    # Set 'kw_total' to NaN on holidays
    data.loc[data['is_holiday'], 'kw_total'] = np.nan

    return data
def remove_intervals_and_holidays(data, intervals, holidays, years, country_pt, clear_holidays, plot=False):
    """
    Removes specified holiday and interval periods from the data, setting 'kw_total' to NaN during those periods.
    """
    cleaned_data = data.copy()
    initial_null_count = cleaned_data['kw_total'].isnull().sum()
    
    # Remove holidays using the holidays library if specified
    if clear_holidays:
        cleaned_data = remove_holidays_using_holidays(cleaned_data, country_pt)
        print('HOLIDAYS CLEANED ---------------------------------------------------------------------')
        print(f'Initial shape: {initial_null_count}, Shape after cleaning: {cleaned_data["kw_total"].isnull().sum()}')
    else:
        print('HOLIDAYS NOT CLEANED ---------------------------------------------------------------------')
        print(f'Initial shape: {initial_null_count}, Shape after cleaning: {cleaned_data["kw_total"].isnull().sum()}')
    
    # Save the null count before removing additional holidays for comparison
    before_additional_holidays_null_count = cleaned_data['kw_total'].isnull().sum()
    cleaned_data = remove_holidays(cleaned_data, holidays, years, plot)
    
    # Display post-cleaning information
    print('HOLIDAYS ---------------------------------------------------------------------')
    print(f'Initial shape: {before_additional_holidays_null_count}, Shape after cleaning: {cleaned_data["kw_total"].isnull().sum()}')
    
    before_intervals_null_count = cleaned_data['kw_total'].isnull().sum()
    cleaned_data = remove_intervals(cleaned_data, intervals, plot)
    
    print('INTERVALS ---------------------------------------------------------------------')
    print(f'Initial count: {before_intervals_null_count}, After cleaning: {cleaned_data["kw_total"].isnull().sum()}')
    
    print('Initial shape', before_additional_holidays_null_count, 'Final shape:', cleaned_data['kw_total'].isnull().sum())

    if plot:
        plot_energy_data(cleaned_data, ['kw_total'],
                         title=f"Remove intervals and holidays: start: {data['kw_total'].isnull().sum()} removed: {cleaned_data['kw_total'].isnull().sum()}")

    # Print null values count for 'kw_total'
    print('Null values for kw_total')
    print(cleaned_data['kw_total'].isnull().sum())
    
    return cleaned_data
def fill_na(cleaned_df, seasonality_type=['hour', 'weekday'], plot=False):
    """
    Fills NaN values in the DataFrame by interpolating based on specified seasonality types.
    """
    df_filled_na = cleaned_df.copy()
    group_list = []
    plot_list_fill_na = False

    # Number of null values
    cols_list = []
    numeric_columns = df_filled_na.select_dtypes(include=['number']).columns
    for col in numeric_columns:
        if df_filled_na[col].isnull().sum() > 0:
            cols_list.append(col)
            # Group by specified seasonality types
            df_grouped = df_filled_na.groupby(seasonality_type)[col]
            for name, group in df_grouped:
                group_copy = group.copy()
                # Interpolate missing values using linear method looking forward and backward
                group.interpolate(method='linear', limit_direction='both', inplace=True)
                group_list.append((group_copy, group, str(name)))
                
                # Populate the DataFrame with interpolated values
                df_filled_na.loc[group.index, col] = group
    
    if plot:
        # Plot the final DataFrame
        plot_energy_data(df_filled_na, ['kw_total'],
                         title=f"FillNa for null values in kw_total: After fill na: {df_filled_na['kw_total'].isnull().sum()}")
        
        # Print null values in kw_total
        print(f'Null values in {cols_list}')
        print(df_filled_na[cols_list].isnull().sum())
    
    if plot_list_fill_na:
        if len(group_list) >= 6:
            sample_list = random.sample(group_list, 6)
            fig, axs = plt.subplots(1, 6, figsize=(25, 5))
            for (group_orig, group_filled, name), ax in zip(sample_list, axs.flatten()):
                ax.plot(group_filled, label='Interpolated Line', color='red', alpha=0.7) 
                ax.plot(group_orig, label='Original Line', color='green', alpha=0.7)  
                ax.set_title(f'{name}')
                ax.legend() 
            plt.tight_layout()
            plt.show()
            
    return df_filled_na
def preprocessing_and_load(real_id, data_merged, plot=False, outliers=False, 
                           output_path='D:\\repositorios\\ADRENALIN-2\\Load Desagregation\\projeto_final\\OtimizaçãoSTL\\Submissao Final\\data'):
    week_mean_factor = 0.35
    day_mean_factor = 0.03
    cleaned_df, original_df = load_dataframe(real_id, data_merged, week_mean_factor, day_mean_factor, plot)
    
    # Remove holidays as specified
    intervals = parameters_dict[real_id]['clean_data_parameters']['interval']
    holidays = parameters_dict[real_id]['clean_data_parameters']['holidays']
    years = parameters_dict[real_id]['clean_data_parameters']['year']
    country_pt = parameters_dict[real_id]['clean_data_parameters']['country']
    clear_holidays = parameters_dict[real_id]['clean_data_parameters']['clear_holidays']

    # Remove outliers and errors
    cleaned_intervals = remove_intervals_and_holidays(cleaned_df, intervals, holidays, years, country_pt, clear_holidays, plot)
    
    # Fill NaN values
    filled_cleaned_df = fill_na(cleaned_intervals, plot=plot)

    # Final result of NaN filling
    print('FINAL RESULT OF FILL NA -------------------------------------------------------------------')
    print(f'Null values in kw_total: {filled_cleaned_df["kw_total"].isnull().sum()}')

    plot_energy_data(filled_cleaned_df, ['kw_total', 'week_min'],
                     title=f"{real_id} Cleaned Load Data: cleaned_df: {filled_cleaned_df['kw_total'].shape} original_df: {original_df.shape}")
    
    # Save the files to CSV
    filled_cleaned_df.to_csv(f'{output_path}/{real_id}_cleaned.csv', index=False)
    original_df.to_csv(f'{output_path}/{real_id}_original.csv', index=False)
    print(f'File saved to: {output_path}/{real_id}_cleaned.csv ----------------------------------------------')
    print(f'File saved to: {output_path}/{real_id}_original.csv ----------------------------------------------')

    return filled_cleaned_df, original_df

