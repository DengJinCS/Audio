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
from Features.DEMON import *


#Extracting the Spectrogram for every Audio

cmap = plt.get_cmap('inferno')

path = '/home/atticus/Documents/ShipData/ShipsEar/ABCDE/'
#genres = 'Fishboat Mussel_boat Ocean_liner Tugboat Sailboat Trawler RORO Passengers Natural_noise Motorboat Pilot_ship Dredger'.split()
genres = 'A B C D E'.split()

###################
#Writing data to csv file
###################
"""
header = 'filename rmse spectral_centroid spectral_bandwidth rolloff'
for i in range(1, 21):
    header += f' mfcc{i} '

for i in range(1, 14):
    header += f' dmfcc{i}'
for i in range(1, 14):
    header += f' ddmfcc{i}'
"""

header = 'filename l_rmse l_spectral_centroid l_spectral_bandwidth l_rolloff l_zero_crossing_rate'

for i in range(1, 21):
    header += f' l_mfcc{i}'
"""
for i in range(1, 21):
    header += f' l_dmfcc{i}'
for i in range(1, 21):
    header += f' l_ddmfcc{i}'
"""


header += ' label'
header = header.split()

csvfile = 'ship_25_5.csv'
if os.path.exists(csvfile):
    os.remove(csvfile)

file = open(csvfile, 'w', newline='')
with file:
    writer = csv.writer(file)
    writer.writerow(header)
countS = 0
countN = 0
for g in genres:
    for shipname in os.listdir(f'{path}/{g}'):
        print("shipname:",shipname)
        """
        data, sr = librosa.load(f'{path}/{g}/{shipname}', mono=True, sr=None)
        #lenth = int((math.floor(len(data) / sr)) / 5)  # 计算每5秒的特征

        demon, demon_sr = DemonAnalysis(data, sr, bandpass=True, low=10000, high=sr / 2 - 1, DEMON_rate=2000, order=50)
        lenth = math.floor(len(demon) / demon_sr)-1  # 计算每1秒的特征

        # 留下低频########################################################################
        low, low_sr = librosa.load(f'{path}/{g}/{shipname}', mono=True, sr=8000)

        for i in range(lenth):
            d = demon[i*demon_sr:(i+1)*demon_sr]# 计算每1秒DEMON的特征
            rmse = librosa.feature.rmse(y=demon)
            spec_cent = librosa.feature.spectral_centroid(y=d, sr=demon_sr)
            spec_bw = librosa.feature.spectral_bandwidth(y=d, sr=demon_sr)
            rolloff = librosa.feature.spectral_rolloff(y=d, sr=demon_sr)
            mfcc = librosa.feature.mfcc(y=d, sr=demon_sr)
            #dmfcc = librosa.feature.delta(mfcc)
            #ddmfcc = librosa.feature.delta(mfcc, order=2)
            # 空格非常重要！！！！！
            
            ID = re.findall(r'\d+', shipname)[0]
        
            to_append = f'{ID} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)}'# A-15__10_07_13_radaUno_Pasa_1.wav
            # 只保留文件ID ，eg. 15
            # 空格非常重要！！！！！
            for e in mfcc:
                to_append += f' {np.mean(e)}'
            
            for e in dmfcc:
                to_append += f' {np.mean(e)}'
            for e in ddmfcc:
                to_append += f' {np.mean(e)}'
            """
        # 留下低频########################################################################
        low, low_sr = librosa.load(f'{path}/{g}/{shipname}', mono=True, sr=None)
        lenth = math.floor(len(low) / low_sr * 10)   # 计算每0.1秒的特征
        each_sample = math.floor(low_sr / 10)
        for i in range(lenth):


            # 留下低频########################################################################
            l = low[i * each_sample : (i + 1) * each_sample]  # 计算每0.1秒低频的特征
            l_rmse = librosa.feature.rmse(y=l)
            l_spec_cent = librosa.feature.spectral_centroid(y=l, sr=low_sr)
            l_spec_bw = librosa.feature.spectral_bandwidth(y=l, sr=low_sr)
            l_rolloff = librosa.feature.spectral_rolloff(y=l, sr=low_sr)
            l_zcr = librosa.feature.zero_crossing_rate(l)
            l_mfcc = librosa.feature.mfcc(y=l, sr=low_sr)
            #l_dmfcc = librosa.feature.delta(l_mfcc)
            #l_ddmfcc = librosa.feature.delta(l_mfcc, order=2)
            # 空格非常重要！！！！！
            ID = re.findall(r'\d+', shipname)[0]
            to_append = f'{ID} {np.mean(l_rmse)} {np.mean(l_spec_cent)} {np.mean(l_spec_bw)} {np.mean(l_rolloff)} {np.mean(l_zcr)}'
            # A-15__10_07_13_radaUno_Pasa_1.wav
            # 只保留文件ID ，eg. 15
            # 空格非常重要！！！！！
            for e in l_mfcc:
                to_append += f' {np.mean(e)}'
            """
            for e in l_dmfcc:
                to_append += f' {np.mean(e)}'
            for e in l_ddmfcc:
                to_append += f' {np.mean(e)}'
            """

            #加上标签
            to_append += f' {g}'

            countN += 1
            print(f'{g} Count:{countN}')


            file = open(csvfile, 'a', newline='')
            with file:
                writer = csv.writer(file)
                writer.writerow(to_append.split())
                print(f'{i}...writing feature for {shipname}')