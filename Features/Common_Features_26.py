# feature extractoring and preprocessing data
import librosa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import pathlib
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
plt.figure(figsize=(10,10))
genres = 'A B C D E'.split()
path = '/home/atticus/PycharmProjects/ship/Data/ShipsEar/ABCDE/'


###################
#Writing data to spec pic
###################
"""
count = 0
for g in genres:
    pathlib.Path(f'img_data/{g}').mkdir(parents=True, exist_ok=True)
    for shipname in os.listdir(f'{path}/{g}'):
        filename = f'{path}/{g}/{shipname}'
        y, sr = librosa.load(filename, mono=True,sr=None)
        lenth = math.floor(len(y)/sr)#计算每一秒的特征
        for i in range(lenth):
            count += 1
            plt.specgram(y[i*sr*2:(i+1)*sr*2], NFFT=2048, Fs=2, Fc=0, noverlap=128,
                         cmap=cmap,
                         sides='default',
                         mode='default',
                         scale='dB')
            plt.axis('off')
            picname = f'img_data/{g}/{shipname[:-3].replace(".", "")}_{i}.png'
            if os.path.exists(picname):
                os.remove(picname)
            print(f'SAVE:{count}-{picname}')
            plt.savefig(picname)
print("count:",count)
"""

###################
#Writing data to csv file
###################

header = 'filename chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
for i in range(1, 21):
    header += f' mfcc{i}'
header += ' label'
header = header.split()


file = open('ship.csv', 'w', newline='')
with file:
    writer = csv.writer(file)
    writer.writerow(header)

genres = 'A B C D E'.split()
for g in genres:
    for shipname in os.listdir(f'{path}/{g}'):
        print("shipname:",shipname)
        data, sr = librosa.load(f'{path}/{g}/{shipname}', mono=True, sr=None)
        lenth = math.floor(len(data) / sr)  # 计算每一秒的特征
        for i in range(lenth):
            filename = f'{i}_{shipname}'
            y = data[i*sr:(i+1)*sr]# 计算每一秒的特征
            chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
            rmse = librosa.feature.rmse(y=y)
            spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
            spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            zcr = librosa.feature.zero_crossing_rate(y)
            mfcc = librosa.feature.mfcc(y=y, sr=sr)
            #空格非常重要！！！！！
            to_append = f'{filename.split("-")[1].split("__")[0]} {np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'
            #A-15__10_07_13_radaUno_Pasa_1.wav
            #只保留文件ID ，eg. 15
            # 空格非常重要！！！！！
            for e in mfcc:
                to_append += f' {np.mean(e)}'
            to_append += f' {g}'
            file = open('ship.csv', 'a', newline='')
            with file:
                writer = csv.writer(file)
                writer.writerow(to_append.split())
                print(f'{i}...writing feature for{filename}')