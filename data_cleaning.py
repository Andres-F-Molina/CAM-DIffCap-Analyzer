from mpr_file_convertor import convert_file_to_cloud
import numpy as np
from scipy.ndimage import median_filter
import pandas as pd
from scipy.signal import savgol_filter, medfilt
import logging


def read_data(file_path: str) -> pd.DataFrame:
    """
    Reads the data from the specified file path and returns a DataFrame.

    Parameters:
        file_path (str): The path to the data file.

    Returns:
        pd.DataFrame: A DataFrame containing the imported data.
    """
    logging.debug(f'DATA_CLEANING. Reading data from {file_path}')
    df = convert_file_to_cloud(file_path)
    logging.debug(f'DATA_CLEANING. Data from {file_path} converted to dataframe')
    return df


def remove_nan(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes NaN values from the voltage and charge columns.

    Parameters:
        df (pd.DataFrame): DataFrame containing 'voltage' and 'step_amp_hours'.

    Returns:
        pd.DataFrame: DataFrame with NaN values removed from 'voltage' and 'step_amp_hours'.
    """
    logging.debug('DATA_CLEANING. Removing NaN values from dataframe')
    voltage = df['voltage']
    charge = df['step_amp_hours']

    # Remove NaN values from the data (if any)
    valid_indices = ~np.isnan(voltage) & ~np.isnan(charge)

    # Create a new DataFrame with only the 'voltage' and 'step_amp_hours' columns
    df_clean = df.loc[valid_indices, ['voltage', 'step_amp_hours','step_type', 'current']].reset_index(drop=True)
    logging.debug('DATA_CLEANING. NaN values removed from dataframe')
    return df_clean


def calculate_dqdv(df: pd.DataFrame, smoothing_window1: int = 40, smoothing_window2: int = 3,
                   polyorder: int = 2, step_size: int = 1) -> pd.DataFrame:
    """
    Calculates dQ/dV with smoothing and adaptive step size to reduce noise.

    Parameters:
        df (pd.DataFrame): DataFrame containing 'voltage' and 'step_amp_hours'.
        smoothing_window1 (int): Window size for the Savitzky-Golay filter for the less noisy region.
        smoothing_window2 (int): Window size for the Savitzky-Golay filter for the more noisy region.
        polyorder (int): Polynomial order for the Savitzky-Golay filter.
        step_size (int): Step size for calculating dQ/dV to reduce noise.

    Returns:
        pd.DataFrame: DataFrame with 'dQ/dV' column added.
    """
    logging.debug(f'DATA_CLEANING. Calculating dQdV with window size for the less noisy'
                  f'region = {smoothing_window1}, and the more noisy region = {smoothing_window2}.'
                  f'polynomial order = {polyorder} and step size = {step_size}')
    voltage = df['voltage'].to_numpy()
    charge = df['step_amp_hours'].to_numpy()
    #charge_mA_h = charge * 1000

    # Calculate dQ and determine regions based on its absolute value
    dQ_raw = np.diff(charge)

    # Apply a median filter to dQ_raw to remove spikes
    dQ_raw = medfilt(dQ_raw, kernel_size=5)  # It is possible to adjust kernel_size to control smoothing

    # Initialize smoothed voltage and charge arrays
    voltage_smooth = np.copy(voltage)
    charge_smooth = np.copy(charge)
    #charge_smooth = np.copy(charge_mA_h)

    # Define masks for different regions
    mask_low_dQ = np.abs(dQ_raw) <= 0.0025
    mask_high_dQ = np.abs(dQ_raw) > 0.0025

    # Define windows for each region
    windows = {
        'low_dQ': smoothing_window1,
        'high_dQ': smoothing_window2
    }

    # Apply smoothing based on dQ absolute values
    for mask, window_key in [(mask_low_dQ, 'low_dQ'), (mask_high_dQ, 'high_dQ')]:
        if np.any(mask):
            indices = np.where(mask)[0]
            if len(indices) > polyorder:
                # Ensure window length does not exceed data length
                adjusted_window = min(windows[window_key], len(indices))
                adjusted_window = adjusted_window if adjusted_window % 2 != 0 else adjusted_window - 1  # Window length must be odd
                voltage_smooth[indices] = savgol_filter(voltage[indices], adjusted_window, polyorder)
                charge_smooth[indices] = savgol_filter(charge[indices], adjusted_window, polyorder)

    # Calculating dQ and dV with adjustable step size after smoothing
    dV = np.diff(voltage_smooth)
    dQ = np.diff(charge_smooth)

    # Initialize dQdV with NaN values
    dQdV = np.full_like(dV, np.nan)

    # Mask where dV is not zero to avoid division by zero
    non_zero_dV = dV != 0
    dQdV[non_zero_dV] = dQ[non_zero_dV] / dV[non_zero_dV]

    # Adjust the length of df and add new columns
    df = df.iloc[:-step_size].copy()  # Align the length of df with dQdV
    df['dQ/dV'] = dQdV
    df['voltage_smooth'] = voltage_smooth[:-step_size]  # Align with the adjusted df length
    df['charge_smooth'] = charge_smooth[:-step_size]  # Align with the adjusted df length

    logging.debug('DATA_CLEANING. dQdV calculated successfully')
    return df

def apply_median_filter(df: pd.DataFrame, size: int = 2) -> pd.DataFrame:
    """
    Applies a median filter to the dQ/dV data.

    Parameters:
        df (pd.DataFrame): DataFrame containing 'dQ/dV'.
        size (int): Size of the filter window. Default is 2.

    Returns:
        pd.DataFrame: DataFrame with 'filtered_dQ/dV' column added.
    """
    logging.debug(f'DATA_CLEANING. Applying median filter to dataframe with filter window size = {size}')
    df = df.copy()
    df['filtered_dQ/dV'] = median_filter(df['dQ/dV'], size=size)

    logging.debug('DATA_CLEANING. Median filter applied successfully to dataframe')
    return df


def remove_outliers(df: pd.DataFrame, percentile: float = 10.0) -> pd.DataFrame:
    """
    Removes outliers from the filtered dQ/dV data based on a threshold.

    Parameters:
        df (pd.DataFrame): DataFrame containing 'filtered_dQ/dV'.
        percentile (float): Percentile for calculating outlier thresholds. Default is 10.0 (10th and 90th percentiles).

    Returns:
        pd.DataFrame: DataFrame with outliers removed from 'filtered_dQ/dV'.
    """
    logging.debug('DATA_CLEANING. Removing outliers')
    filtered_dQdV = df['filtered_dQ/dV']

    # Calculate percentiles for outlier removal
    lower_percentile = np.percentile(filtered_dQdV[~np.isnan(filtered_dQdV)], percentile)
    upper_percentile = np.percentile(filtered_dQdV[~np.isnan(filtered_dQdV)], 100 - percentile)
    IQR = upper_percentile - lower_percentile
    lower_bound = lower_percentile - 1.5 * IQR
    upper_bound = upper_percentile + 1.5 * IQR

    # Identify and replace outliers with NaN
    outliers = (filtered_dQdV < lower_bound) | (filtered_dQdV > upper_bound)
    df.loc[outliers, 'filtered_dQ/dV'] = np.nan

    logging.debug('DATA_CLEANING. Outliers removed successfully')
    return df


def interpolate_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Interpolates the NaN values in the filtered dQ/dV data.

    Parameters:
        df (pd.DataFrame): DataFrame containing 'filtered_dQ/dV' with NaN values.

    Returns:
        pd.DataFrame: DataFrame with interpolated 'filtered_dQ/dV'.
    """
    logging.debug('DATA_CLEANING. Interpolating dQdV data')
    filtered_dQdV = df['filtered_dQ/dV']
    df = df.copy()

    # Interpolate to fill in NaN values
    interpolated_dQdV = np.interp(
        np.arange(len(filtered_dQdV)),
        np.arange(len(filtered_dQdV))[~np.isnan(filtered_dQdV)],
        filtered_dQdV[~np.isnan(filtered_dQdV)]
    )

    df['filtered_dQ/dV'] = interpolated_dQdV
    logging.debug('DATA_CLEANING. dQdV data interpolated successfully')
    return df


def calculate_mid_voltage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates midpoints for the voltage data and adds it to the DataFrame.

    Parameters:
        df (pd.DataFrame): DataFrame containing 'voltage'.

    Returns:
        pd.DataFrame: DataFrame with 'mid_voltage' column added.
    """
    logging.debug('DATA_CLEANING. Calculating mid voltage')
    voltage = df['voltage']
    mid_voltage = voltage[:-1] + np.diff(voltage) / 2

    df = df.iloc[:-1].copy()  # Align the length of df with mid_voltage
    df['mid_voltage'] = mid_voltage

    logging.debug('DATA_CLEANING. Mid voltage calculated successfully')
    return df