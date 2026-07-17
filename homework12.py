import numpy as np


def voiced_excitation(duration, F0, Fs):
    '''
    Create voiced speech excitation.
    '''

    excitation = np.zeros(duration)

    pitch_period = int(np.round(Fs / F0))

    excitation[::pitch_period] = -1

    return excitation


def resonator(x, F, BW, Fs):
    '''
    Generate the output of a resonator.
    '''

    N = len(x)
    y = np.zeros(N)

    # Pole radius
    r = np.exp(-np.pi * BW / Fs)

    # Pole angle
    theta = 2 * np.pi * F / Fs

    # Filter coefficients
    a1 = 2 * r * np.cos(theta)
    a2 = -r ** 2

    for n in range(N):
        y[n] = x[n]

        if n >= 1:
            y[n] += a1 * y[n - 1]

        if n >= 2:
            y[n] += a2 * y[n - 2]

    return y


def synthesize_vowel(duration, F0,
                     F1, F2, F3, F4,
                     BW1, BW2, BW3, BW4,
                     Fs):
    '''
    Synthesize a vowel.
    '''

    # Excitation signal
    excitation = voiced_excitation(duration, F0, Fs)

    # Cascade four resonators
    y = resonator(excitation, F1, BW1, Fs)
    y = resonator(y, F2, BW2, Fs)
    y = resonator(y, F3, BW3, Fs)
    y = resonator(y, F4, BW4, Fs)

    return y
