import librosa
import librosa.display as dis
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import signal
import math


def band_pass(data, fs, low, high, order=50):
    nyq = 0.5 * fs
    normalized_low = low / nyq
    normalized_high = high / nyq
    sos = signal.butter(order, [normalized_low, normalized_high], btype='bandpass', output='sos')
    y = signal.sosfilt(sos, data)
    return y


def test(sample, fs, low=10000, high=25000, order=9):
    sample = band_pass(sample, fs=fs, low=low, high=high, order=order)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(sample)), ref=np.max)
    dis.specshow(D, y_axis='linear', sr=fs, x_axis='s')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Log-frequency power spectrogram')
    plt.show()


def DemonAnalysis(sample, fs, bandpass=True,low=10000, high=260000, DEMON_rate=512, order=50):
    #band_pass & hilbert
    if bandpass:
        sample = band_pass(sample, fs=fs, low=low, high=high, order=order)
    #h_signal = signal.hilbert(sample)
    print("sample length:",len(sample),len(sample)/fs)

    # averaged envelope
    N = math.floor(fs / DEMON_rate)  # 每一个包络窗口的长度
    frame_length = math.floor(len(sample) / N)#窗口个数
    arv_envlope = np.zeros(frame_length)
    for frame in range(frame_length):
        sum = 0
        for i in range(N):
            #sum += sample[frame*N+i]**2 + h_signal[frame*N+i]**2
            sum += sample[frame * N + i] ** 2
        #root mean square
        arv_envlope[frame] = (sum/N) ** 0.5
        #plt.plot(np.sqrt(sample**2+h_signal**2), "r", linewidth=2, label=u"检出的包络信号")
    print("arv-Envlope:",arv_envlope.shape)
    #plt.plot(arv_envlope)
    #plt.title('arv-Envlope')
    #plt.show()

    return arv_envlope,DEMON_rate


