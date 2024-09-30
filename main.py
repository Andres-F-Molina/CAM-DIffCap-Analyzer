import tkinter as tk
from logger_configurator import configure_logging
from tkinter import filedialog
from dQdV_calculator import dqdv_calculator, extract_high_voltage_peaks
from data_cleaning import read_data
import yaml
import os
from peak_fitting_functions import *
from plotting_functions import *
from filename_processing_functions import *
import logging


#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
# LOAD DIRECTORY CONFIGURATION FILE ###################################################################################
try:
    with open("directory_config.yaml", "r") as file:
        config = yaml.safe_load(file)
except FileNotFoundError:
    logging.error("MAIN. Directory configuration file not found. Check that the file directory_config.yaml exists.")
    exit(1)

#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
# SET DIRECTORIES AND LOAD LOGGER CONFIGURATION #######################################################################
base_directory = config['directories']['base_directory']

# Output directory plots
output_plots_directory = config['directories']['output_plots_directory']

# Output directory fitting results
output_fitting_results_directory = config['directories']['output_fitting_results_directory']

# Load logger configuration
configure_logging(base_directory)

# Log the start of the program
logging.debug("MAIN. dQdV Analyser started")

#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
# CONFIGURE FILE SELECTION DIALOG #####################################################################################
# Create a new Tk root window
root = tk.Tk()

# Hide the main window
root.withdraw()

# Open the file selection dialog
mpr_files = filedialog.askopenfilenames(initialdir=base_directory)

# Prompt the user to enter sample ID (QCL or LIMS number)
sample_id = input("Please enter the sample ID (QCL number or another identifier): ")

# Ask the user to select between Gaussian and Voigt peak fitting
peak_fitting_method = input("Please enter the peak fitting method (Gaussian 'G' or Voigt 'V':")

#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
# LOAD FITTING CONFIGURATION PARAMETERS ###############################################################################
try:
    config_path = 'config.yaml'
    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)
except Exception as e:
    logging.error(f"MAIN. An error occurred while loading the configuration file: {e}")
    raise

########################################################################################################################
########################################################################################################################
########################################################################################################################
# Calculate dQdV curve
fitting_results = {}
for file in mpr_files:
    try:
        file_name = os.path.basename(file)
        file_stem = os.path.splitext(file_name)[0]
        df = read_data(file)
        # Calculate differential capacitance curve using the function dqdv_calculator from dQdV_calculator.py
        dQdV_curve = dqdv_calculator(df)
        dQdV_curve['filtered_dQ/dV'] = dQdV_curve['filtered_dQ/dV'] * 1000

        ################################################################################################################
        ################################################################################################################
        ################################################################################################################
        # Extract peak at high voltages
        charge_high_voltage_data = extract_high_voltage_peaks(dQdV_curve, config)
        peak, peak_value, peak_mid_voltage = find_peaks(charge_high_voltage_data)

        ################################################################################################################
        ################################################################################################################
        ################################################################################################################
        # Fit peak at high voltages
        if peak_fitting_method.lower() == 'v':
            parameters, fwhm, area, normalized_SSE, x_fit, y_fit = fit_voigt_peak(charge_high_voltage_data,
                                                                                peak_value,
                                                                                peak_mid_voltage,
                                                                                config)
        elif peak_fitting_method.lower() == 'g':
            parameters, fwhm, area, normalized_SSE, x_fit, y_fit = fit_gaussian_peak(charge_high_voltage_data,
                                                                                    peak_value,
                                                                                    peak_mid_voltage,
                                                                                    config)
        else:
            logging.error(f"MAIN. Invalid peak fitting method: {peak_fitting_method}.")
            continue

        # Adjust baseline based on the last value of the filtered dQ/dV curve
        y_fit = y_fit + charge_high_voltage_data['filtered_dQ/dV'].iloc[-1]
        ################################################################################################################
        ################################################################################################################
        ################################################################################################################
        # Extracting doe and cell number
        doe, cell = extract_doe_cell(file_stem)

        if doe not in fitting_results:
            fitting_results[doe] = []

        fitting_results[doe].append({
            'cell_ID': file_stem,
            'cell': cell,
            'amplitude': parameters[0],
            'center': parameters[1],
            'fwhm': fwhm,
            'area': area,
            'SSE': normalized_SSE
        })
        ################################################################################################################
        ################################################################################################################
        ################################################################################################################
        # Plotting
        plot_dqdv_peak_fitting(file_stem,
                                x_fit,
                                y_fit,
                                charge_high_voltage_data['mid_voltage'],
                                charge_high_voltage_data['filtered_dQ/dV'])
    except Exception as e:
        logging.error(f"MAIN. An error occurred while processing file {file_name}: {e}")
        continue  # Skip to the next file

# Close the Tk root window
root.destroy()

# Convert the fitting results from a dictionary to a DataFrame
fitting_results_df = pd.concat([pd.DataFrame(results) for results in fitting_results.values()],
                               ignore_index=True)

# Save fitting results to a CSV file in the output directory using the function process_fitting_results
summary_results_df = process_fitting_results(fitting_results, output_fitting_results_directory, sample_id)

# Log the end of the program
logging.debug("MAIN. dQdV Analyser finished")