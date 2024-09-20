import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from _1_pre_process import plot_energy_data
from itertools import combinations

def execute_decomposition(cleaned_df, decomposition_type, seasonality, stl_trend, stl_seasonal, stl_robust):
    if decomposition_type == 'classica':
        result = seasonal_decompose(cleaned_df['kw_total'], model='additive', period=seasonality)
    elif decomposition_type in ['stl', 'stl_adjusted']:
        stl = sm.tsa.STL(cleaned_df['kw_total'], seasonal=stl_seasonal, trend=stl_trend, period=seasonality,
                         seasonal_deg=1, trend_deg=1, low_pass_deg=1, robust=stl_robust,
                         seasonal_jump=1, trend_jump=1, low_pass_jump=1)
        result = stl.fit()

    # Extracting components and ensuring they align with the original DataFrame
    cleaned_df['trend'] = result.trend.reindex(cleaned_df.index)
    cleaned_df['seasonal'] = result.seasonal.reindex(cleaned_df.index)
    cleaned_df['residual'] = result.resid.reindex(cleaned_df.index)
    return cleaned_df, result
def select_weekly_data(cleaned_df, selection_adjustment, seasonality):
    # Calculate the quantiles based on selection adjustment for kw_total
    cleaned_df['week_upper_quantile'] = cleaned_df.groupby(['year', 'week', 'id_real'])['kw_total'].transform(
        lambda x: x.quantile(selection_adjustment[1]))
    cleaned_df['week_lower_quantile'] = cleaned_df.groupby(['year', 'week', 'id_real'])['kw_total'].transform(
        lambda x: x.quantile(selection_adjustment[0]))
    
    # Calculate the count of records per group
    cleaned_df['week_count'] = cleaned_df.groupby(['year', 'week', 'id_real'])['kw_total'].transform('count')
    
    # Calculate median based on upper and lower quantiles
    cleaned_df['week_median'] = (cleaned_df['week_upper_quantile'] + cleaned_df['week_lower_quantile']) / 2
    
    # Filter dataframe based on calculated medians and minimum count as per seasonality
    filtered_df = cleaned_df[
        (cleaned_df['week_median'] < cleaned_df['week_median'].quantile(selection_adjustment[3])) &
        (cleaned_df['week_median'] > cleaned_df['week_median'].quantile(selection_adjustment[2])) &
        (cleaned_df['week_count'] >= seasonality)
    ]

    return cleaned_df, filtered_df
def adjust_base_and_height(cleaned_df, filtered_df, multiplier):
    # Adjust the base value for all data points in the cleaned dataframe
    cleaned_df['base_value'] = cleaned_df['kw_total'] * multiplier
    
    # Calculate the base value, median upper quantile (height), and median lower quantile (base) from filtered dataframe
    calculated_base_value = filtered_df['kw_total'] * multiplier
    median_height = filtered_df['week_upper_quantile'].median() * multiplier
    median_base = filtered_df['week_lower_quantile'].median() * multiplier
    
    # Add the median height and base to the cleaned dataframe for reference

    cleaned_df['median_height'] = median_height
    cleaned_df['median_base'] = median_base

    return cleaned_df, calculated_base_value, median_height, median_base
def optimize_parameters(decomposition, cleaned_df, filtered_df, base_value, result):
    best_nrmse = float('inf')
    best_params = {}
    for trend_adjustment in np.arange(1.2, 0.2, -0.05):
        for quantile_adjustment in np.arange(0.0001, 0.8, 0.01):
            if decomposition == 'classica':
                cleaned_df['predict2'] = trend_adjustment * (result.seasonal + result.trend.quantile(quantile_adjustment))
            elif decomposition in ['stl', 'stl_adjusted']:
                cleaned_df['predict2'] = trend_adjustment * (result.seasonal + result.trend)
            
            prediction = cleaned_df['predict2'].loc[filtered_df.index]
            
            error_exponent = 10  
            nrmse = np.mean(np.abs(base_value - prediction) ** (1 / error_exponent))
            if nrmse < best_nrmse:
                print(f'Trend Adjustment: {trend_adjustment}, Quantile Adjustment: {quantile_adjustment}, NRMSE: {nrmse:.2%}')
                best_nrmse = nrmse
                best_params = {'trend_adjustment': trend_adjustment, 'quantile_adjustment': quantile_adjustment}

    print(f'Best NRMSE: {best_nrmse:.2%}, Best Parameters:', best_params)
    return best_params
def calculate_prediction(cleaned_df, result, best_params, decomposition, selection_adjustment, height_base, base_base, multiplier, seasonal_trend_adjustment):
    if decomposition == 'classica':
        cleaned_df['predict2'] = best_params['trend_adjustment'] * (
            result.seasonal + result.trend.quantile(best_params['quantile_adjustment'])
        )

    elif decomposition == 'stl_adjusted':
        cleaned_df['week_year'] = cleaned_df['week'].astype(str) + '_' + cleaned_df['year'].astype(str)
        for day in cleaned_df['week_year'].unique():
            # Select data for specific day
            daily_data = cleaned_df[cleaned_df['week_year'] == day]
            # Calculate combination of seasonal and trend components
            adjusted_data = result.seasonal.loc[daily_data.index] + result.trend.loc[daily_data.index]
            # Normalize the data to range [0, 1]
            norm_max = adjusted_data.quantile(selection_adjustment[5])
            norm_min = adjusted_data.quantile(selection_adjustment[4])
            norm_amplitude = norm_max - norm_min
            normalized_data = (adjusted_data - norm_min) / norm_amplitude if norm_max != norm_min else np.zeros(len(adjusted_data))
            amplitude = height_base - base_base
            
            trend_seasonal_adjust = seasonal_trend_adjustment[0] * (norm_amplitude - amplitude)
            if seasonal_trend_adjustment[1] == 'norm_min':
                scaled_data = (norm_min + normalized_data * (amplitude + trend_seasonal_adjust)) * multiplier
            else:
                scaled_data = (base_base + normalized_data * (amplitude + trend_seasonal_adjust))

            # Assign scaled data back to DataFrame
            cleaned_df.loc[cleaned_df['week_year'] == day, 'predict2'] = scaled_data

    return cleaned_df
def update_and_plot_energy_data(cleaned_df, filtered_df, original_df):
    # Adjust 'heating_percentage' to be between 10% and 90% of 'kw_total'
    cleaned_df['heating_kw'] = (cleaned_df['kw_total'] - cleaned_df['predict2']).clip(lower=0)
    cleaned_df['Temperature_dependent(kW)'] = cleaned_df['heating_kw'].clip(lower=cleaned_df['kw_total'] * 0.1, upper=cleaned_df['kw_total'] * 0.9)
    cleaned_df['heating_percentage'] = cleaned_df['Temperature_dependent(kW)'] / cleaned_df['kw_total']

    # Plot results for filtered data
    plot_energy_data(cleaned_df.loc[filtered_df.index], ['base_value', 'predict2', 'trend', 'week_median',
                                   'week_upper_quantile', 'week_lower_quantile', 'median_height', 'median_base'], 
                     title=f'Reference Week: Max:{round(cleaned_df["median_height"].iloc[0])} Base: {round(cleaned_df["median_base"].iloc[0])}')

    # Plot results for the whole dataset
    plot_energy_data(cleaned_df, ['kw_total', 'predict2', 'trend', 'week_median',
                                  'week_upper_quantile', 'week_lower_quantile', 'median_height', 'median_base'], 
                     title='Prediction from Decomposition')

    # Map 'heating_percentage' to 'Temperature_dependent(kW)' in the original dataset
    heating_map = cleaned_df.set_index('id_real_total')['heating_percentage'].to_dict()
    original_df['heating_percentage'] = original_df['id_real_total'].map(heating_map)
    original_df['Temperature_dependent(kW)'] = original_df['kw_total'] * original_df['heating_percentage']
    
    # Plot disaggregated 'Temperature_dependent(kW)'
    plot_energy_data(original_df, ['kw_total', 'Temperature_dependent(kW)'], title='Disaggregation Forecast of Temperature_dependent(kW)')

    return cleaned_df, original_df
def model(cleaned_df, original_df, seasonality, multiplier, decomposition, stl_trend, stl_seasonal, selection_adjustment, stl_robust, seasonal_trend_adjustment):
    # Execute time series decomposition to separate the seasonal and trend components of the data
    cleaned_df, result = execute_decomposition(cleaned_df, decomposition, seasonality, stl_trend, stl_seasonal, stl_robust)
    
    # Uses quantile thresholds to select weekly data that represents minimum consumption or specific plug usage
    cleaned_df, filtered_df = select_weekly_data(cleaned_df, selection_adjustment, seasonality)
    
    # Adjusts the base and height of the selected data to minimize the error between actual data and the forecast, better aligning the model with observed behaviors
    cleaned_df, base_value, height_base, base_base = adjust_base_and_height(cleaned_df, filtered_df, multiplier)
    
    # Optimize parameters to minimize the error between the model predictions and actual data
    best_params = optimize_parameters(decomposition, cleaned_df, filtered_df, base_value, result)
    
    # Calculate predictions using the optimized parameters and adjust for seasonal and trend fluctuations
    cleaned_df = calculate_prediction(cleaned_df, result, best_params, decomposition, selection_adjustment, height_base, base_base, multiplier, seasonal_trend_adjustment)
    
    # Update the heating values based on the predictions and plot the results for visual analysis
    cleaned_df, original_df = update_and_plot_energy_data(cleaned_df, filtered_df, original_df)
    
    return cleaned_df, original_df
