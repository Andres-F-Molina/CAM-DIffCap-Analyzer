import plotly.graph_objects as go
import logging


def plot_dqdv_peak_fitting(file_stem, x_fit, y_fit, x_original, y_original):
    """
    Plots the fitted curve and original data using Plotly.
    """
    logging.debug(f"PLOTTING. Plotting peak fitting results.")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_fit,
        y=y_fit,
        mode='lines',
        name='Fitted Gaussian Curve'
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
