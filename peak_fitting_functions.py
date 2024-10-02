import numpy as np
from scipy.optimize import minimize
import pandas as pd
import logging
from scipy.special import voigt_profile


# Define Voigt function for fitting
def voigt(x, amplitude, center, sigma, gamma):
    """
        Computes the Voigt profile function for given parameters.

        The Voigt profile is a convolution of a Gaussian and a Lorentzian profile.

        Parameters
        ----------
        x : array_like
            The independent variable where the Voigt profile is evaluated.
        amplitude : float
            The amplitude (peak height) of the Voigt profile.
        center : float
            The center position of the Voigt profile along the x-axis.
        sigma : float
            The standard deviation (width) of the Gaussian component.
        gamma : float
            The half-width at half-maximum (HWHM) of the Lorentzian component.

        Returns
        -------
        array_like
            The Voigt profile evaluated at each point in `x`.

        Notes
        -----
        The function normalizes the Voigt profile so that its peak value equals the specified amplitude.
        """
    vp = voigt_profile(x - center, sigma, gamma)
    vp0 = voigt_profile(0, sigma, gamma)  # Peak value at center
    return amplitude * vp / vp0  # Normalize so that peak value equals amplitude

# Define Gaussian function for fitting
def gaussian(x, amplitude, center, width):
    """
        Computes the Gaussian function for given parameters.

        Parameters
        ----------
        x : array_like
            The independent variable where the Gaussian function is evaluated.
        amplitude : float
            The amplitude (peak height) of the Gaussian function.
        center : float
            The center position of the Gaussian peak along the x-axis.
        width : float
            The standard deviation (width) of the Gaussian distribution.

        Returns
        -------
        array_like
            The Gaussian function evaluated at each point in `x`.
        """
    return amplitude * np.exp(-((x - center) ** 2) / (2 * width ** 2))

# Modify the objective function to include weights
def weighted_gaussian_objective_function(params, x, y, weights):
    """
        Objective function for fitting a Gaussian model to data with weighting.

        Calculates the weighted sum of squared errors between the Gaussian model
        and the observed data.

        Parameters
        ----------
        params : array_like
            The parameters [amplitude, center, width] of the Gaussian function.
        x : array_like
            The independent variable data.
        y : array_like
            The dependent variable data to fit.
        weights : array_like
            The weights applied to each data point in the fitting process.

        Returns
        -------
        float
            The weighted sum of squared errors between the Gaussian model and the data.
        """
    return np.sum(weights * (gaussian(x, *params) - y) ** 2)

# Modify the objective function to include weights for Voigt
def weighted_voigt_objective_function(params, x, y, weights):
    """
        Objective function for fitting a Voigt profile to data with weighting.

        Calculates the weighted sum of squared errors between the Voigt profile
        and the observed data.

        Parameters
        ----------
        params : array_like
            The parameters [amplitude, center, sigma, gamma] of the Voigt function.
        x : array_like
            The independent variable data.
        y : array_like
            The dependent variable data to fit.
        weights : array_like
            The weights applied to each data point in the fitting process.

        Returns
        -------
        float
            The weighted sum of squared errors between the Voigt profile and the data.
        """
    # Unpack params for the Voigt profile
    amplitude, center, sigma, gamma = params
    return np.sum(weights * (voigt(x, amplitude, center, sigma, gamma) - y) ** 2)

# Function to normalize SSE
def normalize_sse(sse, y):
    """
        Normalizes the sum of squared errors (SSE) by the sum of squares of `y`.

        This normalization accounts for differences in magnitudes of `y` values,
        providing a relative measure of the fitting error.

        Parameters
        ----------
        sse : float
            The sum of squared errors from the fitting process.
        y : array_like
            The actual observed data values.

        Returns
        -------
        float
            The normalized SSE.
        """
    return sse / np.sum(y ** 2)

# Update the fitting process accordingly
def fit_voigt_peak(dQdV_charge, peak_value, peak_mid_voltage, config):
    """
    Fits a Voigt profile to the detected peak in dQ/dV charge data.

    Parameters
    ----------
    dQdV_charge : pandas.DataFrame
        DataFrame containing 'mid_voltage' and 'filtered_dQ/dV' columns representing the charge data.
    peak_value : float
        The amplitude (peak height) of the detected peak.
    peak_mid_voltage : float
        The voltage at which the peak occurs.
    config : dict
        Configuration dictionary containing parameters for the fitting process, such as initial guesses and bounds.

    Returns
    -------
    parameters : ndarray
        The optimized parameters [amplitude, center, sigma, gamma] of the Voigt profile.
    fwhm : float
        The full width at half maximum of the fitted Voigt peak.
    area : float
        The area under the fitted Voigt peak.
    normalized_SSE : float
        The normalized sum of squared errors between the fitted Voigt profile and the actual data.
    x_fit : ndarray
        The x-values used for plotting the fitted Voigt profile.
    y_fit_optimized : ndarray
        The y-values of the optimized Voigt profile corresponding to `x_fit`.

    Raises
    ------
    RuntimeError
        If the fitting process does not converge.
    ValueError
        If the normalized SSE exceeds the specified threshold in the configuration.

    Notes
    -----
    This function prepares the initial guesses and bounds for the parameters, computes weights,
    performs the fitting using the `minimize` function from `scipy.optimize`, and calculates
    additional parameters such as FWHM and area based on the optimized parameters.
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
        weighted_voigt_objective_function,
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

def fit_gaussian_peak(dQdV_charge, peak_value, peak_mid_voltage, config):
    """
    Fits a Gaussian function to the detected peak in dQ/dV charge data.

    Parameters
    ----------
    dQdV_charge : pandas.DataFrame
        DataFrame containing 'mid_voltage' and 'filtered_dQ/dV' columns representing the charge data.
    peak_value : float
        The amplitude (peak height) of the detected peak.
    peak_mid_voltage : float
        The voltage at which the peak occurs.
    config : dict
        Configuration dictionary containing parameters for the fitting process, such as initial guesses and bounds.

    Returns
    -------
    parameters : ndarray
        The optimized parameters [amplitude, center, width] of the Gaussian function.
    fwhm : float
        The full width at half maximum of the fitted Gaussian peak.
    area : float
        The area under the fitted Gaussian peak.
    normalized_SSE : float
        The normalized sum of squared errors between the fitted Gaussian function and the actual data.
    x_fit : ndarray
        The x-values used for plotting the fitted Gaussian function.
    y_fit_optimized : ndarray
        The y-values of the optimized Gaussian function corresponding to `x_fit`.

    Raises
    ------
    RuntimeError
        If the fitting process does not converge.
    ValueError
        If the normalized SSE exceeds the specified threshold in the configuration.

    Notes
    -----
    This function prepares the initial guesses and bounds for the parameters, computes weights,
    performs the fitting using the `minimize` function from `scipy.optimize`, and calculates
    additional parameters such as FWHM and area based on the optimized parameters.
    """
    logging.debug(f"PEAK FITTING. Fitting peak.")
    # Initial guess
    initial_guess = np.array([
        peak_value * config['amplitude_initial_guess_percentage'],
        peak_mid_voltage,
        config['width_initial_guess']
    ])

    # Bounds
    lower_bound_center = peak_mid_voltage * config['peak_center_lower_bound_percentage']
    upper_bound_center = peak_mid_voltage * config['peak_center_upper_bound_percentage']

    bounds = [
        (None, np.max(dQdV_charge['filtered_dQ/dV'])),  # Amplitude
        (lower_bound_center, upper_bound_center),     # Center
        (config['width_lower_bound'], config['width_upper_bound'])  # Width
    ]

    # Weights
    min_value = dQdV_charge['filtered_dQ/dV'].min()
    peak_max_value = dQdV_charge['filtered_dQ/dV'].max()
    cutoff_value = min_value + config['weight_cutoff_value'] * (peak_max_value - min_value)
    weights = np.ones_like(dQdV_charge['filtered_dQ/dV'].values)
    weights[dQdV_charge['filtered_dQ/dV'] < cutoff_value] = config['lower_weight_value']

    # Fit
    result = minimize(
        weighted_gaussian_objective_function,
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
    y_fit_optimized = gaussian(x_fit, *parameters)

    # Calculate FWHM and area
    fwhm = 2 * np.sqrt(2 * np.log(2)) * parameters[2]
    area = parameters[0] * parameters[2] * np.sqrt(2 * np.pi)

    logging.debug(f"PEAK FITTING. Peak fitted successfully.")
    return parameters, fwhm, area, normalized_SSE, x_fit, y_fit_optimized


def find_peaks(dQdV_charge):
    """
    Detects peaks in the dQ/dV charge data based on changes in the first derivative.

    Identifies points where the derivative of 'filtered_dQ/dV' changes sign from positive to negative,
    indicating a local maximum (peak).

    Parameters
    ----------
    dQdV_charge : pandas.DataFrame
        DataFrame containing 'filtered_dQ/dV' and 'mid_voltage' columns representing the charge data.

    Returns
    -------
    peaks : ndarray
        Indices of detected peaks in the DataFrame.
    peak_value : float
        The maximum value of 'filtered_dQ/dV' corresponding to the detected peak.
    peak_mid_voltage : float
        The 'mid_voltage' corresponding to the detected peak.

    Notes
    -----
    The function calculates the first derivative of 'filtered_dQ/dV', identifies points where the sign
    of the derivative changes from positive to negative (indicating a peak), and returns the indices
    and values of these peaks.
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
    """
        Processes the fitting results, calculates summary statistics, and exports to CSV.

        Parameters
        ----------
        fitting_results_dict : dict
            A dictionary containing the fitting results for each DOE (Design of Experiments) group.
            The keys are DOE identifiers, and the values are lists of dictionaries with fitting results.
        export_dir : str
            The directory path where the summary CSV file will be saved.
        sample_id : str
            An identifier for the sample, used in the output file name.

        Returns
        -------
        summary_df : pandas.DataFrame
            A DataFrame containing the summary statistics for each DOE group.

        Raises
        ------
        Exception
            If any error occurs during processing, the exception is raised after printing an error message.

        Notes
        -----
        The function calculates the average and standard deviation for parameters like amplitude, center, FWHM, area, SSE,
        and the number of cells for each DOE group. It exports these summary statistics to a CSV file in the specified directory.

        The output CSV file will have the name format 'Summary_Peak_Fitting_{sample_id}.csv'.
        """
    try:
        logging.debug(f"PEAK FITTING. Processing fitting results.")
        
        #pseudo-voigt-fitting
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
