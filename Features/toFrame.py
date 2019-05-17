#in case the audio files from being too long
#cut the long audio into 100 sec


import librosa
from pydub import AudioSegment
import glob
import os
from shutil import move
p = '/home/atticus/PycharmProjects/ship/Data/ShipsEar/ABCDE/'
large = '/home/atticus/PycharmProjects/ship/Data/ShipsEar/large/'
frame = 100000#100s

folders = glob.glob(p+'*')
for folder in folders:
    print(folder)
    files = os.listdir(folder)
    for file in files:
        filename = f'{folder}/{file}'
        print(filename)
        y,sr = librosa.load(filename,sr=None)
        if int(len(y)/sr)<1:
            os.remove(filename)
        if int(len(y)/sr)>100:
            myaudio = AudioSegment.from_file(filename, "wav")
            # 保存切割的音频到文件
            for i in range(frame, int(len(y) / sr * 1000) + frame, frame):
                print(i - frame, "-", i)
                fname = f'{folder}/{file.split(".wav")[0]}_{int(i / frame)}.wav'
                print("fname:", fname)
                if os.path.exists(fname):
                    os.remove(fname)
                sound = myaudio[i - frame:i]
                sound.export(fname, format="wav")
                print("EXPORT to:", fname)
            move(filename,f'{large}{file}')
            print(f'move large file {filename} to {large}{file}')

