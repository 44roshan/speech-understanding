import numpy as np


def waveform_to_frames(waveform, frame_length, step):

    num_frames = (len(waveform) - frame_length) // step + 1

    frames = np.zeros((num_frames, frame_length))

    for i in range(num_frames):
        start = i * step
        frames[i] = waveform[start:start + frame_length]

    return frames


def frames_to_mstft(frames):

    mstft = np.abs(np.fft.fft(frames, axis=1))
    return mstft


def mstft_to_spectrogram(mstft):

    # Set minimum magnitude to 0.001 * maximum magnitude (-60 dB floor)
    floor = 0.001 * np.amax(mstft)
    mstft = np.maximum(mstft, floor)

    # Convert to decibels
    spectrogram = 20 * np.log10(mstft)

    return spectrogram
