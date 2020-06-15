import numpy as np
from scipy.io import wavfile
from IPython.display import Audio
from io import BytesIO

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

inds = np.where(ivx > 1000)
ivx = ivx[inds[0][0]:inds[0][-1]]
inds = np.where(vvx > 1000)
vvx = vvx[inds[0][0]:inds[0][-1]]

max_index = 0
maximum = 0
to_offset = "vvx"
print(vvx.shape, ivx.shape)
larger = max(ivx.shape[0], vvx.shape[0])

for i in range(0, larger // 10000):
    count = np.count_nonzero(vvx[i:i+1000000] == ivx[0:1000000])
    if count > maximum:
        maximum = count
        print("New max vvx:", count, i)
        to_offset = "vvx"
        max_index = i
    count = np.count_nonzero(ivx[i:i+1000000] == vvx[0:1000000])
    if count > maximum:
        maximum = count
        print("New max ivx:", count, i)
        to_offset = "ivx"
        max_index = i

if to_offset == "vvx":
    vvx = vvx[max_index:]
    print("trimming vvx to:", max_index)
if to_offset == "ivx":
    ivx = ivx[max_index:]
    print("trimming ivx to:", max_index)

if vvx.shape[0] > ivx.shape[0]:
    vvx = vvx[:ivx.shape[0]]
else:
    ivx = ivx[:vvx.shape[0]]

wavfile.write("inst.wav", ivfs, ivx)
wavfile.write("voc.wav", vvfs, vvx)