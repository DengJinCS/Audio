# coding: utf-8

import csv
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import glob
import math
from scipy.io import wavfile
import pylab
import time
import gc
path='/home/atticus/PycharmProjects/ship/Data/ShipsEar/ABCDE/'
spec_path = '/home/atticus/PycharmProjects/ship/Data/ShipsEar/ABCDE5_spec/'

if not os.path.exists(spec_path):
    os.makedirs(spec_path)
    print("Cereat spec pic folder:",spec_path)
folders=glob.glob(path+'*')
all = 0
for folder in folders:
    spec_type_folder = spec_path + folder.split(path)[1]
    files = glob.glob(folder + '/' + '*.wav')
    for file in files:
        all +=1
print("ALL:",all)
count = 0

for folder in folders:
    spec_type_folder = spec_path + folder.split(path)[1]
    if not os.path.exists(spec_type_folder):
        os.makedirs(spec_type_folder)
        print("\nCreate type folder:",spec_type_folder)
    files = glob.glob(folder + '/' + '*.wav')
    for file in files:
        data, fs = librosa.load(file, sr=None)  # 分为一秒长的片段
        lenth = math.floor(len(data) / fs / 3) #3秒一段
        if lenth == 0:
            continue
        for i in range(lenth):
            count += 1
            pic = file.split('.wav')[0]  # 去掉后缀
            pic1 = pic.split(folder)[1]  # 保留文件名 /xxxx
            picname = spec_type_folder + pic1 + '_' + str(i) + '.png'
            if os.path.exists(picname):
                os.remove(picname)
            else:
                datam = data[i*3*fs:(i+1)*3*fs]

                nfft = 1024  # Length of the windowing segments
                melspec = librosa.feature.melspectrogram(datam, fs, n_fft=1024, hop_length=512, n_mels=128)
                logmelspec = librosa.power_to_db(melspec)
                #spec pic

                plt.figure(figsize=(3.86,3.89),dpi=100,frameon=False)#inception_v3 input size 299 * 299
                ax = plt.axes()
                ax.set_axis_off()
                plt.set_cmap('hot')
                librosa.display.specshow(logmelspec)#inception_v3 input size 299 * 299
                print(logmelspec.shape,len(logmelspec))
                #edge
                plt.savefig(picname,bbox_inches='tight', transparent=True, pad_inches=0.0 )  # Spectrogram saved as a .png
                # inception_v3 input size 299 * 299!
                # inception_v3 input size 299 * 299!
                # inception_v3 input size 299 * 299!
                plt.close()
                print(picname)
                print("Transforming:", count, "/", all, '%.2f%%' % (count / all * 100))
