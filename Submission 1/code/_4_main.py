from _0_parameters import parameters_dict
from _1_pre_process import preprocessing_and_load, aggregate_data, show_plots
from _2_model import model
from _3_submission import prepare_and_submit
import os
import pandas as pd
import shutil

PATH = 'D:/repositorios/REP_ADRENALIN/Submission 1/'
submission_path = PATH + 'submission'
train_data_path= PATH + 'data\\train_public'
output_path= PATH + 'data'
name_zip_file_path = PATH + 'submission'

NEW_DATA_MERGED = False
NEW_CLEANED_DF = True

# If data merge has already been done, no need to redo it
if os.path.exists(f'{output_path}/data_merged.csv') and not NEW_DATA_MERGED:
    data_merged = pd.read_csv(f'{output_path}/data_merged.csv')
else:
    data_merged = aggregate_data(train_data_path=train_data_path, output_path=output_path)

for real_id in parameters_dict.keys():
    print(f'Processing {real_id} -----------------------------------------------------')

    if 'L14.B04_1H' not in real_id:
        continue

    # If preprocessing has already been done, no need to redo it
    if os.path.exists(f'{output_path}/{real_id}_cleaned.csv') and os.path.exists(f'{output_path}/{real_id}_original.csv') and not NEW_CLEANED_DF:
        cleaned_df = pd.read_csv(f'{output_path}/{real_id}_cleaned.csv')
        cleaned_df['timestamp'] = pd.to_datetime(cleaned_df['timestamp'])
        original_df = pd.read_csv(f'{output_path}/{real_id}_original.csv')
        original_df['timestamp'] = pd.to_datetime(original_df['timestamp'])
    else:
        cleaned_df, original_df = preprocessing_and_load(real_id, data_merged, plot=True, outliers=False, output_path=output_path)

    seasonality = parameters_dict[real_id]['decomposition_parameter']['seasonality']
    multiplier = parameters_dict[real_id]['decomposition_parameter']['multiplier']
    decomposition = parameters_dict[real_id]['decomposition_parameter']['stl']
    stl_trend = parameters_dict[real_id]['decomposition_parameter']['stl_trend']
    stl_seasonal = parameters_dict[real_id]['decomposition_parameter']['stl_seasonal']
    stl_robust = parameters_dict[real_id]['decomposition_parameter']['stl_robust']
    base_selection_adjustment = parameters_dict[real_id]['decomposition_parameter']['week_plug_selection']
    seasonal_trend_adjustment = parameters_dict[real_id]['decomposition_parameter']['seasonal_trend_power']

    cleaned_df, original_df = model(cleaned_df, original_df, seasonality, multiplier, decomposition, stl_trend, stl_seasonal, base_selection_adjustment, stl_robust, seasonal_trend_adjustment)

    prepare_and_submit(data_merged, original_df, real_id, 0.2, submission_path, plot_=True)
    show_plots(ncols=4)
    print(f'Finished processing {real_id} -----------------------------------------------------')
    print()
    print()

    # Uncomment to pause and see plots
    # input('Press enter to continue: ')




# Zip the submission directory
shutil.make_archive(name_zip_file_path, 'zip', submission_path)
print('File saved to: submission.zip')
