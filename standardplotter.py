import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

# def plotter(df, lims, w=0.05, shift = 0,title="Interactive Spectrum", other = None):
#     # Gaussian function
#     def gaussian(x, dE, w, I, window):
#         idx = np.abs(x - dE).argmin()
#         samplespace = idx + window
#         if samplespace[1] > x.shape[0]:
#             samplespace[1] = x.shape[0] - 1
#         elif samplespace[0] < 0:
#             samplespace[0] = 0
#         sample = x[samplespace[0]:samplespace[1]]
#         spectra = np.zeros((x.shape[0]))
#         spectra[samplespace[0]:samplespace[1]] = (I * np.exp(-((sample - dE) ** 2) / (2 * w ** 2)))
#         return spectra
#
#     # Define x-axis range
#     x_vals = np.arange(lims[0], lims[1], w / 10)
#     spacing = w / np.diff(x_vals)[0]
#     window = np.round(np.array([-5 * spacing, 5 * spacing]), 0).astype('int')
#
#     # Generate spectrum
#     spectrum = np.zeros_like(x_vals)
#
#     for _, row in df.iterrows():
#         gauss_curve = gaussian(x_vals, row['frequency'], w, row['intensity'], window)
#         spectrum += gauss_curve
#
#
#     fig = plt.figure(figsize=(15, 5))
#     plt.plot(x_vals + shift, spectrum / np.max(spectrum), linewidth=0.5, color='r', label='DMS Software')
#     if other is not None:
#         rotfreq = other['freq']
#         rotspec = other['spec']
#         plt.plot(rotfreq + shift, -rotspec / np.max(rotspec), linewidth=0.5, color='b', label='IHOD Software')
#     plt.title(title, fontsize=18)
#     plt.ylabel('Intensity', fontsize=14)
#     plt.xlabel('Energy $[cm^{-1}]$', fontsize=14)
#     plt.legend(loc='best')
#     plt.xlim(shift + np.array(lims))
#     # plt.savefig('dms_J20C54T20_ultrafine.png')
#     plt.show()

def plotter(df, lims, w=0.05, shift = 0,  title="Interactive Spectrum", other = None):

    # Create Plotly figure
    fig, ax = plt.subplots(figsize=(7.5, 3.5))
    sp,x = plotspectrum(df, 'Calculated', ax, lims, w, shift, 'black', linewidth = 0.5)

    if other is not None:
        for key, df in other.items():
            plotspectrum(df, key, ax, lims, w, shift)


    plt.xlabel('Frequency $[cm^{-1}]$', fontsize=14)
    plt.ylabel('Intensity', fontsize=14)
    plt.title(title, fontsize=16)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
    return sp,x

def plotspectrum(df, name, ax, lims, w, shift, color=None, linewidth=0.5):
    # Define x-axis range
    x_vals = np.arange(lims[0], lims[1], w / 10) + shift
    dx = np.diff(x_vals)[0]
    spacing = w / dx
    window_size = int(np.round(5 * spacing))

    # Precompute Gaussian template centered at 0
    template_x = np.arange(-window_size, window_size + 1) * dx
    gaussian_template = np.exp(-template_x**2 / (2 * w**2))

    # Normalize template to have peak at 1, scaling by intensity will be done later
    gaussian_template /= gaussian_template.max()

    # Generate full spectrum
    spectrum = np.zeros_like(x_vals)
    df['frequency'] += shift

    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing Rows"):
        center_idx = np.searchsorted(x_vals, row['frequency'])
        start_idx = center_idx - window_size
        end_idx = center_idx + window_size + 1

        # Check bounds
        g_start = max(0, -start_idx)
        g_end = gaussian_template.shape[0] - max(0, end_idx - spectrum.shape[0])
        s_start = max(0, start_idx)
        s_end = min(spectrum.shape[0], end_idx)

        # Add scaled Gaussian into the spectrum
        spectrum[s_start:s_end] += row['intensity'] * gaussian_template[g_start:g_end]

    # Normalize intensity
    df['intensity'] = df['intensity'] / np.max(spectrum)

    # Plot
    if color is not None:
        ax.plot(x_vals, spectrum / np.max(spectrum), label=name, linewidth=linewidth, color=color)
    else:
        ax.plot(x_vals, spectrum / np.max(spectrum), label=name, linewidth=linewidth)

    return spectrum, x_vals
