import numpy as np
import librosa


def lpc(speech, frame_length, frame_skip, order):
    '''
    Perform linear predictive analysis of input speech.
    '''

    nframes = 1 + (len(speech) - frame_length) // frame_skip

    A = np.zeros((nframes, order + 1))
    excitation = np.zeros((nframes, frame_length))

    for i in range(nframes):
        start = i * frame_skip
        frame = speech[start:start + frame_length]

        # LPC coefficients
        a = librosa.lpc(frame, order=order)
        A[i] = a

        # Prediction residual (excitation)
        residual = np.convolve(frame, a, mode='full')[:frame_length]
        excitation[i] = residual

    return A, excitation


def synthesize(e, A, frame_skip):
    '''
    Synthesize speech from LPC residual and coefficients.
    '''

    order = A.shape[1] - 1
    nframes = A.shape[0]

    duration = len(e)
    synthesis = np.zeros(duration)

    for i in range(nframes):

        start = i * frame_skip
        stop = min(start + frame_skip, duration)

        for n in range(start, stop):

            synthesis[n] = e[n]

            for k in range(1, order + 1):
                if n - k >= 0:
                    synthesis[n] -= A[i, k] * synthesis[n - k]

    return synthesis


def robot_voice(excitation, T0, frame_skip):
    '''
    Calculate gain and generate robot excitation.
    '''

    nframes = excitation.shape[0]

    gain = np.zeros(nframes)
    e_robot = np.zeros(nframes * frame_skip)

    for i in range(nframes):

        # Last frame_skip samples
        residual = excitation[i, -frame_skip:]

        # RMS gain
        gain[i] = np.sqrt(np.mean(residual ** 2))

        start = i * frame_skip

        # Periodic impulse train
        for n in range(frame_skip):
            if n % T0 == 0:
                e_robot[start + n] = gain[i]

    return gain, e_robot
