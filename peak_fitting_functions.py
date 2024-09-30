import numpy as np
from scipy.optimize import minimize
import pandas as pd
import logging
from scipy.special import voigt_profile


# Define Voigt function for fitting
def voigt(x, amplitude, center, sigma, gamma):
    """
    Voigt profile function.
    amplitude: peak amplitude
    center: center of the peak
    sigma: standard deviation of the Gaussian part
    gamma: gamma parameter of the Lorentzian part
    """
    return amplitude * voigt_profile(x - center, sigma, gamma)

# Modify the objective function to include weights for Voigt
def weighted_objective_function(params, x, y, weights):
    # Unpack params for the Voigt profile
    amplitude, center, sigma, gamma = params
    return np.sum(weights * (voigt(x, amplitude, center, sigma, gamma) - y) ** 2)

# Function to normalize SSE
def normalize_sse(sse, y):
    return sse / np.sum(y ** 2)

# Update the fitting process accordingly
def fit_peak(dQdV_charge, peak_value, peak_mid_voltage, config):
    """
    Fits a Voigt profile to the detected peak.
    Returns fitted parameters, FWHM, area, and normalized SSE.
    """
    logging.debug(f"PEAK FITTING. Fitting peak.")
    # Initial guess, include gamma (Lorentzian width) in guess
    initial_guess = np.array([
        peak_value * config['amplitude_initial_guess_percentage'],
        peak_mid_voltage,
        config['width_initial_guess'],  # sigma (Gaussian width)
        config['gamma_initial_guess']   # gamma (Lorentzian width)
    ])

    # Bounds, include bounds for gamma
    lower_bound_center = peak_mid_voltage * config['peak_center_lower_bound_percentage']
    upper_bound_center = peak_mid_voltage * config['peak_center_upper_bound_percentage']

    bounds = [
        (None, np.max(dQdV_charge['filtered_dQ/dV'])),  # Amplitude
        (lower_bound_center, upper_bound_center),      # Center
        (config['width_lower_bound'], config['width_upper_bound']),  # Sigma (Gaussian width)
        (config['gamma_lower_bound'], config['gamma_upper_bound'])   # Gamma (Lorentzian width)
    ]

    # Weights remain unchanged
    min_value = dQdV_charge['filtered_dQ/dV'].min()
    peak_max_value = dQdV_charge['filtered_dQ/dV'].max()
    cutoff_value = min_value + config['weight_cutoff_value'] * (peak_max_value - min_value)
    weights = np.ones_like(dQdV_charge['filtered_dQ/dV'].values)
    weights[dQdV_charge['filtered_dQ/dV'] < cutoff_value] = config['lower_weight_value']

    # Fit using the Voigt profile
    result = minimize(
        weighted_objective_function,
        initial_guess,
        args=(
            dQdV_charge['mid_voltage'],
            dQdV_charge['filtered_dQ/dV'] - dQdV_charge['filtered_dQ/dV'].iloc[-1],
            weights
        ),
        method='Powell',
        bounds=bounds,
        options={
            'maxiter': config['max_iter'],
            'maxfev': config['max_eval'],
            'xtol': config['xtol'],
            'ftol': config['ftol'],
            'disp': False
        }
    )

    if not result.success:
        logging.error(f"PEAK FITTING. Fit did not converge: {result.message}")
        raise RuntimeError(f"Fit did not converge: {result.message}")

    current_SSE = result.fun
    normalized_SSE = normalize_sse(current_SSE, dQdV_charge['filtered_dQ/dV'].values)
    if normalized_SSE > config['normalized_SSE_threshold']:
        logging.error(f"PEAK FITTING. SSE too large {normalized_SSE}")
        raise ValueError(f"SSE too large: {normalized_SSE}")

    parameters = result.x
    x_fit = np.linspace(
        np.min(dQdV_charge['mid_voltage']),
        np.max(dQdV_charge['mid_voltage']),
        500
    )
    y_fit_optimized = voigt(x_fit, *parameters)

    # Calculate FWHM and area for the Voigt profile
    fwhm = 0.5346 * 2 * parameters[3] + np.sqrt(0.2166 * (2 * parameters[3]) ** 2 + (2.355 * parameters[2]) ** 2)
    area = parameters[0] * (parameters[2] * np.sqrt(2 * np.pi) + 2 * parameters[3])

    logging.debug(f"PEAK FITTING. Peak fitted successfully.")
    return parameters, fwhm, area, normalized_SSE, x_fit, y_fit_optimized


def find_peaks(dQdV_charge):
    """
    Detects peaks based on the first derivative of the dQ/dV curve.
    Returns peak indices and peak values.
    """
    logging.debug(f"PEAK FITTING. Finding peaks.")
    dQdV_charge = dQdV_charge.copy()
    dQdV_charge['dQ/dV_diff'] = dQdV_charge['filtered_dQ/dV'].diff().fillna(0)

    sign_change = np.sign(dQdV_charge['dQ/dV_diff'].values)

    # Handle zero sign values by carrying forward the last non-zero sign
    for i in range(1, len(sign_change)):
        if sign_change[i] == 0:
            sign_change[i] = sign_change[i - 1]

    # Detect where the sign goes from +1 to -1 (indicating a peak)
    peaks = np.where((sign_change[:-1] > 0) & (sign_change[1:] < 0))[0]
    #peak_values = dQdV_charge['filtered_dQ/dV'].iloc[peaks].values

    # Find the maximum value of 'filtered_dQ/dV' and its index
    max_peak_index = dQdV_charge['filtered_dQ/dV'].idxmax()
    peak_value = dQdV_charge['filtered_dQ/dV'].loc[max_peak_index]
    peak_mid_voltage = dQdV_charge['mid_voltage'].loc[max_peak_index]

    logging.debug(f"PEAK FITTING. Peak found at {peak_value}.")
    return peaks, peak_value, peak_mid_voltage

def process_fitting_results(fitting_results_dict, export_dir, sample_id):
    try:
        logging.debug(f"PEAK FITTING. Processing fitting results.")


        summary_results = []

        for doe, results in fitting_results_dict.items():
            df_results = pd.DataFrame(results)

            # Calculate averages and standard deviations for each column
            avg_amplitude = df_results['amplitude'].mean()
            std_amplitude = df_results['amplitude'].std()
            avg_center = df_results['center'].mean()
            std_center = df_results['center'].std()
            avg_fwhm = df_results['fwhm'].mean()
            std_fwhm = df_results['fwhm'].std()
            avg_area = df_results['area'].mean()
            std_area = df_results['area'].std()
            avg_sse = df_results['SSE'].mean()
            std_sse = df_results['SSE'].std()

            # Count the number of cells included in the calculations
            num_cells = len(df_results)

            # Append results to summary list
            summary_results.append({
                'DOE': doe,
                'avg_amplitude': avg_amplitude,
                'std_amplitude': std_amplitude,
                'avg_center': avg_center,
                'std_center': std_center,
                'avg_fwhm': avg_fwhm,
                'std_fwhm': std_fwhm,
                'avg_area': avg_area,
                'std_area': std_area,
                'avg_SSE': avg_sse,
                'std_SSE': std_sse,
                'num_cells': num_cells,
            })

        # Convert summary results to DataFrame and save to CSV
        summary_df = pd.DataFrame(summary_results)

        # Define the output file path
        output_file = f'{export_dir}/Summary_Peak_Fitting_{sample_id}.csv'

        # Save the DataFrame to CSV
        summary_df.to_csv(output_file, index=False)

        # Notify the user about the successful export
        print(f"Summary successfully exported to {output_file}")

        logging.debug(f"PEAK FITTING. Results processed successfully.")
        return summary_df

    except Exception as e:
        print(f"An error occurred: {e}")
        raise