import numpy as np
from scipy.io import wavfile
import moviepy.editor
import pytube
from IPython.display import Audio
from scipy.io import wavfile
from io import BytesIO
import os
import glob
from sys import argv
import scipy.signal

name_vocals = "vocals.wav"
vocals = Audio(filename=name_vocals)
vvfs, vvx = wavfile.read(BytesIO(vocals.data))
if vvx.ndim == 2:
    vvx = vvx[:, 0]
    print('Type: Stereo, taking one channel only')
else:
    print('Type: Mono')

name_instrumentals = "instrumentals.wav"
instrumentals = Audio(filename=name_instrumentals)
ivfs, ivx = wavfile.read(BytesIO(instrumentals.data))
if ivx.ndim == 2:
    ivx = ivx[:, 0]
    print('Type: Stereo, taking one channel only')
else:
    print('Type: Mono')

print(vvx.shape, ivx.shape)
cross_correlation = np.correlate(vvx[:10000], ivx[21:10000+21], "valid")
print(cross_correlation.max(), cross_correlation.mean(), cross_correlation.shape, cross_correlation.argmax())

max_index = 0
maximum = 0
print(vvx.shape, ivx.shape)
larger = max(ivx.shape[0], vvx.shape[0])
for i in range(0, larger // 100):
    try:
        count = np.count_nonzero(vvx[i:i+1000000] == ivx[0:1000000])
        if count > maximum:
            maximum = count
            print("New max vvx:", count, i)
    except IndexError:
        pass
    try:
        count = np.count_nonzero(ivx[i:i+1000000] == vvx[0:1000000])
        if count > maximum:
            maximum = count
            print("New max ivx:", count, i)
    except IndexError:
        pass

print(maximum)

