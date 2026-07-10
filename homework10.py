import numpy as np
import torch
import torch.nn as nn


def get_features(waveform, Fs):
    '''
    Extract feature vectors and labels from waveform.
    '''

    # Pre-emphasis
    waveform = np.append(waveform[0], waveform[1:] - 0.97 * waveform[:-1])

    # Spectrogram parameters
    frame_length = int(0.004 * Fs)   # 4 ms
    step = int(0.002 * Fs)           # 2 ms

    frames = waveform_to_frames(waveform, frame_length, step)
    mstft = frames_to_mstft(frames)
    spectrogram = mstft_to_spectrogram(mstft)

    # Keep low-frequency half
    features = spectrogram[:, :frame_length // 2]

    # ----------- Create Labels -----------
    labels = np.zeros(features.shape[0], dtype=int)

    segments = VAD(waveform, Fs)

    samples_per_frame = step
    current_label = 1

    for seg in segments:
        seg_len = len(seg)
        nframes = max(1, seg_len // samples_per_frame)

        start = (current_label - 1) * 5
        end = min(start + 5, len(labels))

        labels[start:end] = current_label
        current_label += 1

    return features, labels


def train_neuralnet(features, labels, iterations):
    '''
    Train a simple neural network.
    '''

    x = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)

    input_dim = features.shape[1]
    output_dim = np.max(labels) + 1

    model = nn.Sequential(
        nn.LayerNorm(input_dim),
        nn.Linear(input_dim, output_dim)
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    lossvalues = np.zeros(iterations)

    for i in range(iterations):
        optimizer.zero_grad()

        outputs = model(x)

        loss = criterion(outputs, y)

        loss.backward()
        optimizer.step()

        lossvalues[i] = loss.item()

    return model, lossvalues


def test_neuralnet(model, features):
    '''
    Test the trained neural network.
    '''

    x = torch.tensor(features, dtype=torch.float32)

    with torch.no_grad():
        outputs = model(x)
        probabilities = torch.softmax(outputs, dim=1).detach().numpy()

    return probabilities
