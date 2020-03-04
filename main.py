import numpy as np
import cmath
import matplotlib.pyplot as plt

from math import *
from IPython.display import Audio
from scipy.io import wavfile
from scipy.signal import argrelextrema
import scipy.signal as sps
from io import BytesIO
from sys import argv

import warnings


def stft(vx, vfs):
    vx_copy = np.copy(vx)

    # Stereo is represented as a list of two item lists, while mono is just a single big list
    try:
        vx_copy = [stereo[0] for stereo in vx_copy]
        print('Type: Stereo, taking left channel only')
    except IndexError:
        print('Type: Mono')

    # Getting sample size, adding zero padding to last sample set so all sets are the same size
    sample_size = int(argv[1])
    sample_set = [vx_copy[i*sample_size:(i+1)*sample_size] for i in range((len(vx_copy)+sample_size-1)//sample_size)]
    sample_set[-1] = np.append(sample_set[-1], [0] * (sample_size-len(sample_set[-1])))

    stft = np.apply_along_axis(np.fft.rfft, 0, np.copy(sample_set).T)
    return np.abs(stft), np.angle(stft)


def nmf(V, k=63):
    t = V.shape[0]
    f = V.shape[1]
    W = V.copy()
    H = np.ones((k, t))
    v1 = np.ones(V.shape)
    print('f = {0}, t = {1}, W: {2}, H: {3}, v1: {4}'.format(f, t, W.shape, H.shape, v1.shape))

    for i in range(0,100):
        print('nmf loop', i)
        H = H * np.dot(np.transpose(W), V / np.dot(W, H)) / np.dot(np.transpose(W), v1)
        W = W * np.dot(V / np.dot(W, H), np.transpose(H)) / np.dot(v1, np.transpose(H))

    return W, H


def istft(stft_magnitude, stft_phase):
    return np.apply_along_axis(np.fft.irfft, 0, stft_magnitude * np.exp(1j*stft_phase)).T


def main():
    music = Audio(filename="megalovania.wav")
    vfs, vx = wavfile.read(BytesIO(music.data))

    stft_mag, stft_phase = stft(vx, vfs)
    print('stft done', stft_mag.shape)

    basis_partial, activation_partial = nmf(stft_mag[:35], len(stft_mag[0]))
    print('partial nmf done', basis_partial.shape, activation_partial.shape, stft_mag.shape)

    basis, activation = nmf(stft_mag, len(stft_mag[0]))
    print('main nmf done')

    reconstructed = np.concatenate(istft(np.dot(basis_partial, activation), stft_phase)).ravel().astype(vx.dtype)
    print('reconstruction done')

    wavfile.write("reconstructed_violin.wav", vfs, reconstructed)


if __name__ == "__main__":
    main()
