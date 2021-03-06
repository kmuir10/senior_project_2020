import numpy as np
import matplotlib.pyplot as plt

from IPython.display import Audio
from scipy.io import wavfile
from io import BytesIO
from sys import argv


def stft(vx, vfs):
    vx_copy = np.copy(vx)

    # Stereo is two columns, mono is one
    if vx_copy.ndim == 2:
        vx_copy = vx[:, 0]
        print('Type: Stereo, taking one channel only')
    else:
        print('Type: Mono')

    # Getting sample size, adding zero padding to last sample set so all sets are the same size
    sample_size = 8192
    vx_copy = np.pad(vx_copy, (0, sample_size - vx_copy.shape[0] % sample_size), constant_values=(0,))
    vx_copy = np.split(vx_copy, vx_copy.shape[0] // sample_size)

    stft = np.apply_along_axis(np.fft.rfft, 1, vx_copy).T
    return np.abs(stft), np.angle(stft)


def nmf(V, k=63):
    f, t = V.shape
    W = np.random.uniform(np.min(V), np.max(V), (f, k))
    print(np.min(V), np.max(V))
    H = np.ones((k, t))
    v1 = np.ones(V.shape)
    print('f = {0}, t = {1}, W: {2}, H: {3}, v1: {4}'.format(f, t, W.shape, H.shape, v1.shape))

    for i in range(0, 100):
        if i % 10 == 0:
            print('nmf loop', i)
        H = H * np.dot(W.T, V / np.dot(W, H)) / np.dot(W.T, v1)
        W = W * np.dot(V / np.dot(W, H), H.T) / np.dot(v1, H.T)

    return W, H


def nmf_supervised(V, k, W_learned):
    t = V.shape[1]
    f = V.shape[0]
    rest = np.random.uniform(np.min(V), np.max(V), (f, k))
    W = np.concatenate((W_learned, rest), axis=1)
    print(np.min(V), np.max(V))
    H = np.ones((k+100, t))
    v1 = np.ones(V.shape)
    print('f = {0}, t = {1}, W: {2}, H: {3}, v1: {4}'.format(f, t, W.shape, H.shape, v1.shape))

    for i in range(0, 100):
        if i % 10 == 0:
            print('nmf loop', i)
        H = H * np.dot(W.T, V / np.dot(W, H)) / np.dot(W.T, v1)
        W = W * np.dot(V / np.dot(W, H), H.T) / np.dot(v1, H.T)

    return W, H


def istft(stft_magnitude, stft_phase):
    return np.apply_along_axis(np.fft.irfft, 1, stft_magnitude * np.exp(1j*stft_phase))


def main():
    music = Audio(filename=argv[2])
    vfs, vx = wavfile.read(BytesIO(music.data))
    sample_size, k = int(argv[1]), int(argv[3])
    start_time, stop_time = int(argv[4]) * vfs // sample_size, int(argv[5]) * vfs // sample_size
    print(start_time, stop_time)

    stft_mag, stft_phase = stft(vx, vfs)
    print('stft done', stft_mag.shape)

    basis_partial, activation_partial = nmf(stft_mag[:, start_time:stop_time], 30)
    print('partial nmf done', basis_partial.shape, activation_partial.shape)

    basis, activation = nmf_supervised(stft_mag, 60, basis_partial)
    print('main nmf done', basis.shape, activation.shape)

    reconstructed = np.concatenate(istft(np.dot(basis[:, 30:], activation[30:]).T, stft_phase.T)).ravel().astype(vx.dtype)
    print('reconstruction done')

    wavfile.write("reconstructed_{}.wav".format(argv[2]), vfs, reconstructed)


if __name__ == "__main__":
    main()
