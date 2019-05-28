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

header = 'filename'

for i in range(1, 41):
    header += f' mfcc{i}'

for i in range(1, 41):
    header += f' dmfcc{i}'
for i in range(1, 41):
    header += f' ddmfcc{i}'
header += ' label'
header = header.split()

csvfile = 'ship_mfcc39.csv'
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

        # 留下低频########################################################################
        low, low_sr = librosa.load(f'{path}/{g}/{shipname}', mono=True, sr=8000)
        lenth = math.floor(len(low) / low_sr)   # 计算每1秒的特征

        for i in range(lenth):

            # 留下低频########################################################################
            l = low[i * low_sr:(i + 1) * low_sr]  # 计算每1秒低频的特征
            l_mfcc = librosa.feature.mfcc(y=l, sr=low_sr,n_mfcc=40)
            l_dmfcc = librosa.feature.delta(l_mfcc)
            l_ddmfcc = librosa.feature.delta(l_mfcc, order=2)
            # 空格非常重要！！！！！
            ID = re.findall(r'\d+', shipname)[0]
            to_append = f'{ID}'
            # A-15__10_07_13_radaUno_Pasa_1.wav
            # 只保留文件ID ，eg. 15
            # 空格非常重要！！！！！
            for e in l_mfcc:
                to_append += f' {np.mean(e)}'
            for e in l_dmfcc:
                to_append += f' {np.mean(e)}'
            for e in l_ddmfcc:
                to_append += f' {np.mean(e)}'


            #加上标签
            to_append += f' {g}'

            countN += 1
            print(f'{g} Count:{countN}')


            file = open(csvfile, 'a', newline='')
            with file:
                writer = csv.writer(file)
                writer.writerow(to_append.split())
                print(f'{i}...writing feature for {shipname}')