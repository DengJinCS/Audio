# feature extractoring and preprocessing data
import librosa
import re
import numpy as np
import matplotlib.pyplot as plt
import os

import math
import csv

# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

#Keras
import keras

import warnings
warnings.filterwarnings('ignore')



#Extracting the Spectrogram for every Audio

cmap = plt.get_cmap('inferno')

path = '/home/atticus/PycharmProjects/ship/Data/Type/'
genres = 'Fishboat Mussel_boat Ocean_liner Tugboat Sailboat Trawler RORO Passengers Natural_noise Motorboat Pilot_ship Dredger'.split()


###################
#Writing data to csv file
###################

header = 'filename chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
for i in range(1, 21):
    header += f' mfcc{i}'
for i in range(21, 41):
    header += f' dmfcc{i}'
for i in range(41, 61):
    header += f' ddmfcc{i}'
header += ' label'
header = header.split()

csvfile = 'ship_66_11.csv'
if os.path.exists(csvfile):
    os.remove(csvfile)

file = open(csvfile, 'w', newline='')
with file:
    writer = csv.writer(file)
    writer.writerow(header)

for g in genres:
    for shipname in os.listdir(f'{path}/{g}'):
        print("shipname:",shipname)
        data, sr = librosa.load(f'{path}/{g}/{shipname}', mono=True, sr=None)
        #lenth = int((math.floor(len(data) / sr)) / 5)  # 计算每5秒的特征
        lenth = math.floor(len(data) / sr)  # 计算每1秒的特征
        for i in range(lenth):
            filename = f'{i}_{shipname}'
            y = data[i*sr:(i+1)*sr]# 计算每1秒的特征
            #y = data[5 * i * sr:(i + 1) * sr * 5]  # 计算每5秒的特征
            chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
            rmse = librosa.feature.rmse(y=y)
            spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
            spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            zcr = librosa.feature.zero_crossing_rate(y)
            mfcc = librosa.feature.mfcc(y=y, sr=sr)
            dmfcc = librosa.feature.delta(mfcc)
            ddmfcc = librosa.feature.delta(mfcc, order=2)
            #空格非常重要！！！！！
            ID = re.findall(r'\d+',filename)[0]
            to_append = f'{ID} {np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'
            #A-15__10_07_13_radaUno_Pasa_1.wav
            #只保留文件ID ，eg. 15
            # 空格非常重要！！！！！
            for e in mfcc:
                to_append += f' {np.mean(e)}'
            for e in dmfcc:
                to_append += f' {np.mean(e)}'
            for e in ddmfcc:
                to_append += f' {np.mean(e)}'
            to_append += f' {g}'
            file = open(csvfile, 'a', newline='')
            with file:
                writer = csv.writer(file)
                writer.writerow(to_append.split())
                print(f'{i}...writing feature for{filename}')