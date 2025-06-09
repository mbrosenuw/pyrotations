import numpy as np
import plotly.graph_objects as go

def plotter(df, lims, w=0.05, shift = 0,  title="Interactive Spectrum", other = None):

    # Create Plotly figure
    fig = go.Figure()
    plotspectrum(df, 'Dimethyl Sulfide Code', fig, lims, w, shift, 'red')

    if other is not None:
        for key, df in other.items():
            plotspectrum(df, key, fig, lims, w, shift, 'blue')



    # Layout settings
    fig.update_layout(
        title=title,
        xaxis_title="Frequency (cm⁻¹)",
        yaxis_title="Intensity",
        hovermode="closest"
    )

    fig.show()


def plotspectrum(df, name, fig, lims, w, shift, color):
    def gaussian(x, dE, w, I, window):
        idx = np.abs(x - dE).argmin()
        samplespace = idx + window
        if samplespace[1] > x.shape[0]:
            samplespace[1] = x.shape[0] - 1
        elif samplespace[0] < 0:
            samplespace[0] = 0
        sample = x[samplespace[0]:samplespace[1]]
        spectra = np.zeros((x.shape[0]))
        spectra[samplespace[0]:samplespace[1]] = (I * np.exp(-((sample - dE) ** 2) / (2 * w ** 2)))
        return spectra

    # Define x-axis range
    x_vals = np.arange(lims[0], lims[1], w / 10) + shift
    spacing = w / np.diff(x_vals)[0]
    window = np.round(np.array([-5 * spacing, 5 * spacing]), 0).astype('int')

    # Generate spectrum
    spectrum = np.zeros_like(x_vals)
    df['frequency'] = df['frequency'] + shift
    for _, row in df.iterrows():
        gauss_curve = gaussian(x_vals, row['frequency'], w, row['intensity'], window)
        spectrum += gauss_curve

    df['intensity'] = df['intensity'] / np.max(spectrum)
    fig.add_trace(go.Scatter(x=x_vals, y=spectrum / np.max(spectrum), mode='lines', name=name, fillcolor = color))

    # Add markers at actual Gaussian peak positions
    for _, row in df.iterrows():
        peak_height = row['intensity']
        if peak_height > 10 ** (-3) * df['intensity'].max():
            fig.add_trace(go.Scatter(
                x=[row['frequency']], y=[peak_height],
                mode='markers',
                marker=dict(size=8, color=color),
                name=f"{row['lower_state']} → {row['upper_state']}",
                hovertemplate=(
                    f"Frequency: {row['frequency']} cm⁻¹<br>"
                    f"Intensity: {row['intensity']}<br>"
                    f"Lower State: {row['lower_state']}<br>"
                    f"Upper State: {row['upper_state']}"
                )
            ))