import os
import csv
import numpy as np
import re

path = '/home/atticus/Documents/ship/'
csvfile = 'ship_incp.csv'

if os.path.exists(csvfile):
    os.remove(csvfile)

header = 'filename'
for i in range(1, 2049):
    header += f' incp{i}'
header += ' label'
header = header.split()
file = open(csvfile, 'w', newline='')
with file:
    writer = csv.writer(file)
    writer.writerow(header)


genres = 'A B C D E'.split()
count =0
for g in genres:
    for shipname in os.listdir(f'{path}/{g}'):
        count +=1
        filename = f'{path}/{g}/{shipname}'
        with open(filename, 'r') as bottleneck_file:
            bottleneck_string = bottleneck_file.read()
            bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
            ID = re.findall(r'\d+',shipname)[0]
            print(f'count:{count}—{ID}')
            to_append = f'{ID}'
            # A-15__10_07_13_radaUno_Pasa_1.wav
            # 只保留文件ID ，eg. 15
            # 空格非常重要！！！！！
            for e in bottleneck_values:
                to_append += f' {e}'
            to_append += f' {g}'
            file = open(csvfile, 'a', newline='')
            with file:
                writer = csv.writer(file)
                writer.writerow(to_append.split())
                #print(f'...writing feature for{filename}')





