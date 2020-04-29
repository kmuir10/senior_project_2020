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

fileList = glob.glob('*.wav')
for filePath in fileList:
    try:
        os.remove(filePath)
    except:
        print("Error while deleting file : ", filePath)

video_url_instrumentals = argv[1]
youtube_instrumentals = pytube.YouTube(video_url_instrumentals)
video_instrumentals = youtube_instrumentals.streams.first()
name_instrumentals = video_instrumentals.default_filename
print(name_instrumentals)
video_instrumentals.download()

video_instrumentals = moviepy.editor.VideoFileClip(name_instrumentals)
audio_instrumentals = video_instrumentals.audio
audio_instrumentals.write_audiofile(name_instrumentals.replace('mp4', 'wav'))

video_url_vocals = argv[2]
youtube_vocals = pytube.YouTube(video_url_vocals)
video_vocals = youtube_vocals.streams.first()
name_vocals = video_vocals.default_filename
print(name_vocals)
video_vocals.download()

video_vocals = moviepy.editor.VideoFileClip(name_vocals)
audio_vocals = video_vocals.audio
audio_vocals.write_audiofile(name_vocals.replace('mp4', 'wav'))

# Clean up unnecessary video files
video_vocals.close()
video_instrumentals.close()
audio_instrumentals.close()
audio_vocals.close()
os.remove(name_vocals)
os.remove(name_instrumentals)

name_vocals = name_vocals.replace('mp4', 'wav')
vocals = Audio(filename=name_vocals)
vvfs, vvx = wavfile.read(BytesIO(vocals.data))
if vvx.ndim == 2:
    vvx = vvx[:, 0]
    print('Type: Stereo, taking one channel only')
else:
    print('Type: Mono')

name_instrumentals = name_instrumentals.replace('mp4', 'wav')
instrumentals = Audio(filename=name_instrumentals)
ivfs, ivx = wavfile.read(BytesIO(instrumentals.data))
if ivx.ndim == 2:
    ivx = ivx[:, 0]
    print('Type: Stereo, taking one channel only')
else:
    print('Type: Mono')

print(vvx.shape, ivx.shape)
cross_correlation = scipy.signal.correlate(vvx[:4000000], ivx[:4000000])
print(cross_correlation.max(), cross_correlation.mean())
