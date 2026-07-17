import numpy as np


def VAD(waveform, Fs):
    
    frame_length = int(0.025 * Fs)   # 25 ms
    step = int(0.010 * Fs)           # 10 ms

    frames = waveform_to_frames(waveform, frame_length, step)

    # Energy of each frame
    energy = np.sum(frames**2, axis=1)

    threshold = 0.1 * np.max(energy)

    speech_frames = energy > threshold

    segments = []
    start = None

    for i, flag in enumerate(speech_frames):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            s = start * step
            e = min(len(waveform), (i - 1) * step + frame_length)
            segments.append(waveform[s:e])
            start = None

    # Last segment
    if start is not None:
        s = start * step
        segments.append(waveform[s:])

    return segments


def segments_to_models(segments, Fs):
    '''
    Create average log spectrum model for each speech segment.
    '''
    models = []

    frame_length = int(0.004 * Fs)   # 4 ms
    step = int(0.002 * Fs)           # 2 ms

    for segment in segments:

        # Pre-emphasis
        emphasized = np.append(segment[0], segment[1:] - 0.97 * segment[:-1])

        frames = waveform_to_frames(emphasized, frame_length, step)

        mstft = frames_to_mstft(frames)

        spectrogram = mstft_to_spectrogram(mstft)

        # Keep low-frequency half
        spectrogram = spectrogram[:, :frame_length // 2]

        # Average spectrum
        model = np.mean(spectrogram, axis=0)

        models.append(model)

    return models


def recognize_speech(testspeech, Fs, models, labels):
    '''
    Recognize each speech segment using cosine similarity.
    '''
    segments = VAD(testspeech, Fs)

    test_models = segments_to_models(segments, Fs)

    Y = len(models)
    K = len(test_models)

    sims = np.zeros((Y, K))
    test_outputs = []

    for k, test_model in enumerate(test_models):

        for y, model in enumerate(models):
            sims[y, k] = np.dot(model, test_model) / (
                np.linalg.norm(model) * np.linalg.norm(test_model)
            )

        best = np.argmax(sims[:, k])
        test_outputs.append(labels[best])

    return sims, test_outputs
