"""
========================
Spectrum Representations
========================

The plots show different spectrum representations of a sine signal with
additive noise. A (frequency) spectrum of a discrete-time signal is calculated
by utilizing the fast Fourier transform (FFT).
"""
import matplotlib.pyplot as plt
import numpy as np
from Features.DEMON import *
import librosa.display
import re

np.random.seed(0)
datapath = "/home/atticus/Documents/ShipData/ShipsEar/ABCDE/A/A-96__A__Draga_4.wav"
ID = int(re.findall(r'\d+',datapath)[0])
plt.figure(figsize=(21,9))

"""
Colormap	Description
jet	a spectral map with dark endpoints, blue-cyan-yellow-red; based on a fluid-jet simulation by NCSA [1]
ocean	green-blue-white
"""
origin, rate = librosa.load(datapath, sr=None)
print("RATE of sample=", rate)
cmap = 'jet'
low = 20000
high = rate/2 - 1
demon_rate = 2000


envelop,envrate = DemonAnalysis(origin, fs=rate, low=low, high=high, DEMON_rate=demon_rate, order=50)
S = librosa.feature.melspectrogram(y=envelop,sr=envrate,n_fft=envrate*0.09,hop_length=envrate*0.045)
librosa.display.specshow(librosa.power_to_db(S,
                         ref=np.max),
                         y_axis='mel',
                         x_axis='time',sr=envrate)
plt.show()


# Origin Magnitude Spectrum
dt = 1/rate  # sampling interval
t = np.arange(0, len(origin)/rate,dt)
ok = min(len(origin),len(t))
y = origin[0:ok]#保证时间和采样点一样多
t = t[0:ok]
#plt.subplot(2, 3, 1)
plt.magnitude_spectrum(y, Fs=rate)
plt.title(f'Origin Magnitude Spectrum of {ID}')
plt.show()

#Bandpassed  Magnitude Spectrum
p = band_pass(origin, fs=rate, low=low, high=high, order=50)
plt.subplot(2, 3, 2)
plt.magnitude_spectrum(p, Fs=rate)
plt.title(f'Bandpassed  Magnitude Spectrum of {ID}')

#envelop Magnitude Spectrum
envelop,envrate = DemonAnalysis(origin, fs=rate, low=low, high=high, DEMON_rate=demon_rate, order=50)
#plt.subplot(2, 3, 3)
plt.magnitude_spectrum(envelop, Fs=envrate)
plt.title(f'Envelop Magnitude Spectrum of {ID}')
plt.show()
#Origin Linear-frequency power spectrogram
D = librosa.amplitude_to_db(np.abs(librosa.stft(origin)), ref=np.max)
plt.subplot(2, 3, 4)
librosa.display.specshow(D,cmap=cmap,x_axis='time', y_axis='linear',sr=rate,hop_length=512)
print("RATE=",rate)
plt.colorbar(format='%+2.0f dB')
plt.title(f'Origin Linear-frequency power spectrogram of {ID}')

#Visualize an STFT power spectrum
no_pass_env,_= DemonAnalysis(origin, bandpass=False,fs=rate, low=low, high=high, DEMON_rate=demon_rate, order=50)
DEMON_0 = librosa.amplitude_to_db(np.abs(librosa.stft(no_pass_env)), ref=np.max)
plt.subplot(2, 3, 5)
librosa.display.specshow(DEMON_0,cmap=cmap,x_axis='time', y_axis='linear',sr=envrate,hop_length=envrate/4)
plt.colorbar(format='%+2.0f dB')
plt.title(f'DEMON spectrogram without bandpass of {ID}')


#Visualize an STFT power spectrum
DEMON = librosa.amplitude_to_db(np.abs(librosa.stft(envelop)), ref=np.max)
plt.subplot(2, 3, 6)
librosa.display.specshow(DEMON,cmap=cmap,x_axis='time', y_axis='linear',sr=envrate,hop_length=envrate/4)
plt.colorbar(format='%+2.0f dB')
plt.title(f'DEMON spectrogram of {ID}')
plt.show()



















"""
# plot time signal:
plt.subplots(3,2,1)
plt.title("Signal")
plt.plot(t, s, color='C0')
plt.xlabel("Time")
plt.ylabel("Amplitude")


#plot DEMON spectrum
plt.subplots(3,2,2)
D = np.abs(librosa.stft(s,n_fft=Fs))
librosa.display.specshow(librosa.amplitude_to_db(D,ref = np.max),
                             sr=Fs,hop_length=Fs/2,
                             y_axis = 'linear', x_axis = 'time')
plt.title(f'DEMON spectrogram of {ID}')
#axes[0, 1].colorbar(format='%+2.0f dB')
#axes[0, 1].tight_layout()

# plot different spectrum types:
axes[1, 0].set_title(f'Magnitude Spectrum of {ID}')
axes[1, 0].magnitude_spectrum(s, Fs=Fs, color='C1')

axes[1, 1].set_title(f'Log Magnitude Spectrum of {ID}')
axes[1, 1].magnitude_spectrum(s, Fs=Fs, scale='dB', color='C1')

axes[2, 0].set_title(f'Phase Spectrum of {ID}')
axes[2, 0].phase_spectrum(s, Fs=Fs, color='C2')

axes[2, 1].set_title(f'Angle Spectrum of ')
axes[2, 1].angle_spectrum(s, Fs=Fs, color='C2')

axes[0, 1].remove()  # don't display empty ax

fig.tight_layout()
plt.show()
"""
