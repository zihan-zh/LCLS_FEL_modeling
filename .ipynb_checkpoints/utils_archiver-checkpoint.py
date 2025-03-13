import datetime
import os
from copy import deepcopy
from typing import Union

import numpy as np
import pandas as pd

def convert_to_dataframe_no_filling_gap(archive_data, pv_name=None):

    data_frames = {}
    for pv_data in archive_data:
        df = pd.DataFrame({
            'datetime': pd.to_datetime(pv_data['value']['value']['secondsPastEpoch'] * 1e9 + pv_data['value']['value']['nanoseconds'], utc=True),
            pv_data['pvName']: pv_data['value']['value']['values']
        }).set_index('datetime')
        data_frames[pv_data['pvName']] = df.sort_index()  # Ensure it's sorted

    # Determine the base DataFrame
    if pv_name and pv_name in data_frames:
        base_df = data_frames[pv_name].reset_index()
    else:
        # Find the DataFrame with the most timestamps
        base_df = max(data_frames.values(), key=lambda df: df.index.size).reset_index()

    # Sort the base_df by datetime to ensure asof merge works correctly
    base_df = base_df.sort_values(by='datetime')

    # Merge other dataframes to the base_df using merge_asof
    for name, df in data_frames.items():
        if name != pv_name:
            df = df.reset_index().sort_values(by='datetime')  # Ensure df is sorted
            base_df = pd.merge_asof(base_df, df, on='datetime', direction='backward').set_index('datetime')

    # Drop the duplicated column if exists and rename appropriately
    for name in data_frames:
        if f'{name}_x' in base_df.columns:
            base_df.drop(columns=[f'{name}_x'], inplace=True)
        if f'{name}_y' in base_df.columns:
            base_df.rename(columns={f'{name}_y': name}, inplace=True)

    # Remove rows with NaN values
    base_df.dropna(inplace=True)

    # Reorder columns to match the order of PVs in archive_data
    column_order = [pv_data['pvName'] for pv_data in archive_data if pv_data['pvName'] in base_df.columns]
    base_df = base_df[column_order]

    # Convert timezone from UTC to 'US/Pacific'
    base_df = base_df.tz_convert('US/Pacific')
    
    return base_df

def group_sample(sample_set, pv_signal):

    # Step 1: Create a Boolean mask indicating where rows change
    changes = sample_set[pv_signal].ne(sample_set[pv_signal].shift()).any(axis=1)
    
    # Step 2: Calculate the cumulative sum of changes to create group identifiers
    sample_set['group'] = changes.cumsum()
    
    # Step 3: Count the number of occurrences in each group
    group_counts = sample_set.groupby('group').size()
    
    # Optional Step 4: Extract indices for each group
    group_indices = sample_set.groupby('group').apply(lambda x: x.index.tolist())
    
    columns_to_average = [col for col in sample_set.columns if col != 'group']
    # Convert datetime to Unix timestamp for averaging (keeping timezone)
    sample_set['datetime_numeric'] = sample_set.index.tz_localize(None).astype('int64') 
    
    # Group by 'group' and calculate the mean for datetime and other columns
    grouped_datetime = sample_set.groupby('group')['datetime_numeric'].mean()
    grouped_other_cols = sample_set.groupby('group')[columns_to_average].mean()
    
    # Convert the averaged Unix timestamp back to datetime (reapply timezone)
    grouped_datetime = pd.to_datetime(grouped_datetime).dt.tz_localize('UTC').dt.tz_convert(local_time_zone)
    
    # Manually adjust the datetime by adding 7 hours
    grouped_datetime += pd.Timedelta(hours=7)
    
    # Combine the averaged datetime with other columns
    grouped_df = pd.concat([grouped_datetime, grouped_other_cols], axis=1)
    
    # Resetting index to make 'group' a column instead of an index
    grouped_df.reset_index(inplace=True)
    
    # Optional: reorder columns to put datetime first
    grouped_df = grouped_df[['group', 'datetime_numeric'] + [col for col in grouped_df.columns if col not in ['group', 'datetime_numeric']]]
    
    # Rename the datetime column
    grouped_df.rename(columns={'datetime_numeric': 'average_datetime'}, inplace=True)
    
    # Display the new DataFrame
    return grouped_df
    