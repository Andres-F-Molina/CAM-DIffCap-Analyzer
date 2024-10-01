import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
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


def plot_dqdv_peak_fitting_with_residuals(file_stem, x_fit, y_fit, x_original, y_original, fitting_method):
    """
    Plots the fitted curve, original data, and residuals using Plotly with two subplots.

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
        The fitting method used; 'v' for Voigt fitting, 'g' for Gaussian fitting.

    Returns
    -------
    None
        Displays the plot using Plotly.
    """
    logging.debug(f"PLOTTING. Plotting peak fitting results with residuals.")

    # Interpolate y_fit at x_original to compute residuals accurately
    y_fit_interpolated = np.interp(x_original, x_fit, y_fit)

    # Calculate residuals
    residuals = y_original - y_fit_interpolated

    # Create subplots with shared x-axis
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.02, row_heights=[0.3, 0.7]
    )

    # Determine the name based on the fitting method
    if fitting_method.lower() == 'v':
        fit_name = 'Fitted Voigt Curve'
    elif fitting_method.lower() == 'g':
        fit_name = 'Fitted Gaussian Curve'
    else:
        logging.error(f"PLOTTING. Invalid fitting method: {fitting_method}.")
        return

    # Top panel: Residuals
    fig.add_trace(
        go.Scatter(
            x=x_original,
            y=residuals,
            mode='markers',
            name='Residuals',
            marker=dict(size=6, color='green')
        ),
        row=1, col=1
    )

    # Add a horizontal line at y=0 for reference
    fig.add_hline(
        y=0,
        line=dict(color='black', dash='dash'),
        row=1, col=1
    )

    # Bottom panel: Original data and fitted curve
    fig.add_trace(
        go.Scatter(
            x=x_original,
            y=y_original,
            mode='markers',
            name='Original Data',
            marker=dict(size=6, color='blue')
        ),
        row=2, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=x_fit,
            y=y_fit,
            mode='lines',
            name=fit_name,
            line=dict(color='red', width=2)
        ),
        row=2, col=1
    )

    # Update layout
    fig.update_layout(
        title=f'Peak Fitting Results for {file_stem}',
        showlegend=True,
        height=800
    )

    # Update x-axis properties
    fig.update_xaxes(title_text='Voltage (V)', row=2, col=1)
    fig.update_xaxes(showticklabels=False, row=1, col=1)

    # Update y-axis properties
    fig.update_yaxes(title_text='Residuals', row=1, col=1)
    fig.update_yaxes(title_text='dQ/dV', row=2, col=1)

    fig.show()
    logging.debug(f"PLOTTING. Peak fitting results with residuals plotted.")
