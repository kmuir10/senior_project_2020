import numpy as np
import matplotlib.pyplot as plt

from math import *
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
    sample_size = int(argv[1])
    vx_copy = np.pad(vx_copy, (0, sample_size - vx_copy.shape[0] % sample_size), constant_values=(0,))
    vx_copy = np.split(vx_copy, vx_copy.shape[0] // sample_size)

    stft = np.apply_along_axis(np.fft.rfft, 1, vx_copy)
    return np.abs(stft), np.angle(stft)


def nmf(V, k=63):
    t = V.shape[0]
    f = V.shape[1]
    W = np.ones((k, f))
    H = np.ones((t, k))
    v1 = np.ones(V.shape)
    print('f = {0}, t = {1}, W: {2}, H: {3}, v1: {4}'.format(f, t, W.shape, H.shape, v1.shape))
    print(np.transpose(W).shape, V.shape, np.dot(H, W).shape)

    for i in range(0,100):
        print('nmf loop', i)
        # H = H * np.dot(np.transpose(W), V / np.dot(W, H)) / np.dot(np.transpose(W), v1)
        # W = W * np.dot(V / np.dot(W, H), np.transpose(H)) / np.dot(v1, np.transpose(H))
        H = H * np.dot(V / np.dot(H, W), np.transpose(W)) / np.dot(v1, np.transpose(W))
        W = W * np.dot(np.transpose(H), V / np.dot(H, W)) / np.dot(np.transpose(H), v1)

    return W, H


def istft(stft_magnitude, stft_phase):
    return np.apply_along_axis(np.fft.irfft, 1, stft_magnitude * np.exp(1j*stft_phase))


def main():
    music = Audio(filename=argv[2])
    vfs, vx = wavfile.read(BytesIO(music.data))

    stft_mag, stft_phase = stft(vx, vfs)
    print('stft done', stft_mag.shape)

    # basis_partial, activation_partial = nmf(stft_mag[:35], 200)
    # print('partial nmf done', basis_partial.shape, activation_partial.shape, stft_mag.shape)

    basis, activation = nmf(stft_mag, stft_mag.shape[0])
    print('main nmf done')

    reconstructed = np.concatenate(istft(np.dot(activation, basis), stft_phase)).ravel().astype(vx.dtype)
    print('reconstruction done')

    wavfile.write("reconstructed_violin.wav", vfs, reconstructed)


if __name__ == "__main__":
    main()
