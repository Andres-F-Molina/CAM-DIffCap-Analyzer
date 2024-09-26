from data_cleaning import *
import pandas as pd
import logging


def clean_and_separate_data(df: pd.DataFrame, voltage_threshold: float = 4.3, current_epsilon: float = 1.0e-7) -> tuple:
    """
    Cleans the input dataframe by removing NaN values, filtering based on voltage, and separating charging and discharging data.

    Parameters:
        df (pd.DataFrame): Original dataframe containing the data.
        voltage_threshold (float): Voltage threshold to filter the data.
        current_epsilon (float): Threshold for current to distinguish between charging and discharging.

    Returns:
        tuple: Two dataframes, one for charging data and one for discharging data.
    """
    logging.debug(f"dQdV CALCULATOR. Cleaning and separating data")
    # Remove NaN values
    df_clean = remove_nan(df).reset_index(drop=True)

    # Filter based on voltage
    df_clean = df_clean[df_clean['voltage'] < voltage_threshold]

    # Separate charging and discharging data
    df_clean_dch = df_clean[df_clean['current'] < -current_epsilon]
    df_clean_ch = df_clean[df_clean['current'] > current_epsilon]

    logging.debug(f"dQdV CALCULATOR. Data cleaned and separated")
    return df_clean_ch, df_clean_dch


def dqdv_calculator(df: pd.DataFrame, voltage_threshold: float = 4.3, current_epsilon: float = 1.0e-7) -> pd.DataFrame:
    """
    Processes the input dataframe by cleaning it, separating it into charging and discharging data,
    calculating dQ/dV, combining the results, and applying further data processing.

    Parameters:
        df (pd.DataFrame): Original dataframe containing the data.
        voltage_threshold (float): Voltage threshold to filter the data.
        current_epsilon (float): Threshold for current to distinguish between charging and discharging.

    Returns:
        pd.DataFrame: Final processed dataframe.
    """
    logging.debug(f"dQdV CALCULATOR. Calculating dQdV.")
    # Clean and separate the data
    df_clean_ch, df_clean_dch = clean_and_separate_data(df, voltage_threshold, current_epsilon)

    # Calculate dQ/dV for charging and discharging data
    dqdv_df_ch = calculate_dqdv(df_clean_ch)
    dqdv_df_dch = calculate_dqdv(df_clean_dch)

    # Combine charging and discharging data
    dqdv_df_combined = pd.concat([dqdv_df_ch, dqdv_df_dch], axis=0, ignore_index=True)

    # Further data processing
    df_filtered = apply_median_filter(dqdv_df_combined, size=3)
    df_no_outliers = remove_outliers(df_filtered, percentile=20)
    df_interpolated = interpolate_data(df_no_outliers)
    df_final = calculate_mid_voltage(df_interpolated)

    logging.debug(f"dQdV CALCULATOR. dQdV calculated successfully.")
    return df_final


def extract_high_voltage_peaks(dQdV_curve, config):
    """
    Extracts peaks at high voltages based on config parameters.
    """
    logging.debug(f"dQdV CALCULATOR. Extracting high voltage peaks.")
    dQdV_charge = dQdV_curve[
        (dQdV_curve['filtered_dQ/dV'] > config['amplitude_bound_value']) &
        (dQdV_curve['voltage'] > config['lower_potential_bound']) &
        (dQdV_curve['voltage'] < config['upper_potential_bound'])
    ].copy()

    logging.debug(f"dQdV CALCULATOR. Calculating dQdV.")
    return dQdV_charge