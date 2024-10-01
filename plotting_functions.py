import plotly.graph_objects as go
import logging


def plot_dqdv_peak_fitting(file_stem, x_fit, y_fit, x_original, y_original, fitting_method):
    """
    Plots the fitted curve and original data using Plotly.

    Parameters
    ----------
    file_stem : str
        The title of the plot.
    x_fit : array-like
        The x-values of the fitted curve.
    y_fit : array-like
        The y-values of the fitted curve.
    x_original : array-like
        The x-values of the original data.
    y_original : array-like
        The y-values of the original data.
    fitting_method : str
        The fitting method used; 'v' for Voigt fitting, any other value for Gaussian fitting.

    Returns
    -------
    None
        Displays the plot using Plotly.
    """
    logging.debug(f"PLOTTING. Plotting peak fitting results.")
    fig = go.Figure()
    # Determine the name based on the fitting method
    if fitting_method.lower() == 'v':
        fit_name = 'Fitted Voigt Curve'
    elif fitting_method.lower() == 'g':
        fit_name = 'Fitted Gaussian Curve'
    else:
        logging.error(f"PLOTTING. Invalid fitting method: {fitting_method}.")
        return

    fig.add_trace(go.Scatter(
        x=x_fit,
        y=y_fit,
        mode='lines',
        name=fit_name
    ))
    fig.add_trace(go.Scatter(
        x=x_original,
        y=y_original,
        mode='lines',
        name='Original Data'
    ))
    fig.update_layout(
        title=f'{file_stem}',
        xaxis_title='Voltage (V)',
        yaxis_title='dQ/dV',
        legend=dict(x=0.7, y=1.1)
    )
    fig.show()
    logging.debug(f"PLOTTING. Peak fitting results fitted.")
