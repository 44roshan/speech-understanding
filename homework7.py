import numpy as np


def major_chord(f, Fs):

    N = int(0.5 * Fs)
    t = np.arange(N) / Fs

    # Root, major third, major fifth
    f_root = f
    f_third = f * 2**(4/12)
    f_fifth = f * 2**(7/12)

    x = (np.cos(2*np.pi*f_root*t) +
         np.cos(2*np.pi*f_third*t) +
         np.cos(2*np.pi*f_fifth*t))

    return x


def dft_matrix(N):

    k = np.arange(N).reshape(N, 1)
    n = np.arange(N).reshape(1, N)

    W = np.cos(2*np.pi*k*n/N) - 1j*np.sin(2*np.pi*k*n/N)

    return W


def spectral_analysis(x, Fs):

    N = len(x)

    W = dft_matrix(N)
    X = W @ x

    magnitude = np.abs(X)

    # Keep only positive frequencies
    magnitude = magnitude[:N//2]

    # Indices of the three largest peaks
    indices = np.argsort(magnitude)[-3:]

    # Convert bin numbers to frequencies
    freqs = indices * Fs / N

    freqs = np.sort(freqs)

    f1, f2, f3 = freqs

    return f1, f2, f3
