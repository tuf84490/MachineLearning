from sklearn.utils.class_weight import compute_class_weight
from python_speech_features import mfcc, logfbank
from tqdm import tqdm
import os
import pandas as pd
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
import librosa

#return the sections of the audio files that register above the specified threshold so as to filter out sounds too quite to be part of the audio event 
def envelope(y, rate, threshold):
    mask = []
    y = pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window=int(rate/10), min_periods=1, center=True).mean()
    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask  

#read in the csv with the training metadata, replace this line to train with other data
df = pd.read_csv("data/homedata.csv")
#get a list of all possible classes as well as the distrobution of the number of the particular class found compared to the total number of audio events
classes = list(np.unique(df.category))
df.reset_index(inplace=True)

audio_dir = 'splitData'
for fn in tqdm(os.listdir(audio_dir)):
    signal, rate = librosa.load(os.path.join(audio_dir, fn), sr=16000)
    mask = envelope(signal, rate, 0.0005)
    wavfile.write(filename='cleanHomeData/'+fn, rate=rate, data=signal[mask])
