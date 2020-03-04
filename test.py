'#1!/usr/bin/env senior_project'
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


def fft_tests():
    audio = Audio("pianoCMajor.wav")
    fs, x = wavfile.read(BytesIO(audio.data))
    print(fs)

    #sample = x[15000:23192]
    sample = [0]*128000
    freq = 99.397
    for n in range(len(sample)):
        sample[n] = sin(2*pi*freq*n/fs)
        sample[n] += .5 * sin(2*pi*2*freq*n/fs)
        sample[n] += .25 * sin(2 * pi * 3 * freq * n / fs)
        sample[n] += .125 * sin(2 * pi * 4 * freq * n / fs)
        sample[n] += .0625 * sin(2 * pi * 5 * freq * n / fs)
    SAMPLE = np.fft.fft(sample)
    SAMPLE_pow = np.abs(SAMPLE) ** 2

    N = len(sample)
    f_pos = np.arange(0, fs/2, step=fs / N)
    plt.plot(f_pos, SAMPLE_pow[:(N//2)])

    f_neg = np.arange(-fs / 2, 0, step=fs/N)

    plt.xlabel("Frequency (Hz)")
    max_indices = argrelextrema(SAMPLE_pow, np.greater)

    very_max = [i for i in max_indices[0] if (SAMPLE_pow[i] > max(SAMPLE_pow) / 100)]

    max_frequencies = [i*fs/N for i in very_max if i*fs/N < 20000]
    print(max_frequencies, fs/N)

    alpha, beta, gamma = SAMPLE_pow[very_max[0]-1], SAMPLE_pow[very_max[0]], SAMPLE_pow[very_max[0]+1]
    interpolated_offset = .5*(alpha - gamma) / (alpha - 2*beta + gamma)
    interpolated_fundamental = (very_max[0] + interpolated_offset) * fs / N

    print(interpolated_fundamental, interpolated_offset)
    print('fft estimate accuracy: ', max_frequencies[0]/freq * 100)
    print('spectral interpolation accuracy: ', interpolated_fundamental/freq * 100)

    plt.show()


def spectrogram(vx, vfs):
    # violin = Audio("megalovania.wav")
    # vfs, vx = wavfile.read(BytesIO(violin.data))
    print('start')

    vx_copy = np.copy(vx)
    vx_copy = [item for sublist in vx_copy for item in sublist]
    sample_size = int(argv[1])
    sample_set = [vx_copy[i*sample_size:(i+1)*sample_size] for i in range((len(vx_copy)+sample_size-1)//sample_size)]
    sample_set[-1] = np.append(sample_set[-1], [0] * (sample_size-len(sample_set[-1])))

    stft_magnitude = []
    stft_phase = []
    for sample in sample_set:
        fft = np.fft.fft(sample)]
        stft_magnitude += [[abs(complex_number) for complex_number in fft]]
        stft_phase += [[np.angle(complex_number) for complex_number in fft]]
    plt.imshow(np.array(stft_magnitude).T.tolist(), 'gray_r', origin='lower', aspect='auto')
    plt.show()