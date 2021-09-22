# https://ocefpaf.github.io/python4oceanographers/blog/2014/07/07/pytides/
# https://www.youtube.com/watch?v=q7sRdRJ0axI

import pytides.constituent as constituent
from pytides.astro import astro
import numpy as np
import scipy.fftpack
from scipy import fft, arange
import pandas.core.frame


# LEAST SQUARED METHOD FOR FIXED POINT IN SPACE
def least_squared_method(time, zeta, omega):
    M = len(time)

    if isinstance(zeta, pandas.core.frame.DataFrame):
        print('Function needs a Series, not a DataFrame')
    elif isinstance(zeta, pandas.core.frame.Series):
        zeta0 = zeta.mean()
        zeta = zeta.values - zeta0
    elif isinstance(zeta, np.ndarray or list):
        zeta0 = np.mean(zeta)
        zeta = zeta - zeta0

    N = len(omega)
    alpha = np.zeros(shape=(2 * N, 2 * N))
    beta = np.zeros(shape=(2 * N,))
    for n in range(0, N):
        for j in range(0, N):
            alpha[j, n] = sum(np.cos(omega[j] * time) * np.cos(omega[n] * time))
            alpha[j, N + n] = sum(np.sin(omega[j] * time) * np.cos(omega[n] * time))
            alpha[N + j, n] = sum(np.cos(omega[j] * time) * np.sin(omega[n] * time))
            alpha[N + j, N + n] = sum(np.sin(omega[j] * time) * np.sin(omega[n] * time))

        beta[n,] = sum(zeta * np.cos(omega[n] * time))
        beta[N + n,] = sum(zeta * np.sin(omega[n] * time))

    ampl = np.dot(np.linalg.inv(alpha), beta)
    C, S = np.split(ampl, 2)

    zeta_model = np.zeros(M)
    zeta_model.fill(zeta0)
    for n in range(0, N):
        model = lambda t: C[n] * np.cos(omega[n] * t) + S[n] * np.sin(omega[n] * t)
        zeta_model += model(time)

    return C, S, zeta_model


def fft_spectrum(time, zeta):
    """
    Plots a Single-Sided Amplitude Spectrum of y(t)
    """
    M = len(time)  # length of the signal
    F = np.fft.fft(zeta)
    freq = np.fft.fftfreq(M, time[-1] / M)

    # plot(f, abs(Y), 'r')  # plotting the spectrum
    # xlabel('Freq (Hz)')
    # ylabel('|Y(freq)|')
