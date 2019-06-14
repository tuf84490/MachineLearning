'''
from keras.layers import Conv2D, MaxPool2D, Flatten, LSTM
from keras.layers import Dropout, Dense, TimeDistributed
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
from python_speech_features import mfcc
'''
from tqdm import tqdm
import os
import pandas as pd
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
'''
def build_rand_feat():
    X = []
    y = []
    _min, _max = float('inf'), -float('inf')
    for _ in tqdm(range(n_samples)):
        rand_class = np.random.choice(class_dist.index, p=prob_dist)
        file = np.random.choice(df[df.category==rand_class].index)
        rate, wav = wavfile.read(file)
        label = df.at[file, 'label']
        rand_index = np.random.randint(0, wav.shape[0]-config.step)
        sample = wav[rand_index:rand_index+config.step]
        X_sample = mfcc(sample, rate, numcep=config.nfeat, nfilt=config.nfilt,nfft=config.nfft).T
        _min = min(np.amin(X_sample), _min)
        _max = max(np.amax(X_sample), _max)
        X.append(X_sample if config.mode == 'conv' else X_sample.T)
        y.append(classes.index(label))
    X,y = np.array(X), np.array(y)
    X = (X-_min) / (_max - _min)
    if config.mode == 'conv':
        print('aaaaaaaa')
    elif(config.mode == 'time'):
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2])
    y = to_categorical(y, num_classes=50)
    return X,y
'''
class Config:
    def __init__(self, mode='time', nfilt=26, nfeat=13, nfft=512, rate=44100):
        self.mode = mode
        self.nfilt = nfilt
        self.nfeat = nfeat
        self.nfft = nfft
        self.rate = rate
        self.step = int(rate/10)

#read in the csv with the training metadata, replace this line to train with other data
df = pd.read_csv("ESC-50-master/meta/esc50.csv")
df = df.set_index('filename')

#sample rate is 44100
#read in the rate and signal of all the audio files
#store the audio data in a dataframe. This dataframe contains all the audio data, their class, and how long the audio file is
for f in df.index:
    rate, signal = wavfile.read("ESC-50-master/audio/" + f)
    df.at[f,'length'] = signal.shape[0]/rate
    print(rate)
    print(signal)

#get a list of all possible classes as well as the distrobution of the number of the particular class found compared to the total number of audio events
classes = list(np.unique(df.category))
class_dist = df.groupby(['category'])['length'].mean()
print(classes)
print(class_dist)

#plot a pie graph of the class distrobution 
fig, ax = plt.subplots()
ax.set_title('Class distribution', y=1.08)
ax.pie(class_dist, labels=class_dist.index, autopct='%1.0f%%', shadow=False, startangle=0)
ax.axis('equal')
plt.show()

#set a large samplesize and choose a random class
n_samples = 2 * int(df['length'].sum() / 0.1 )
prob_dist = class_dist / class_dist.sum()
choices = np.random.choice(class_dist.index, p = prob_dist)
print(choices)

config = Config(mode='time')

if(config.mode == 'time'):
    X, y = build_rand_feat()

