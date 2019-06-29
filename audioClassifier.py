from keras.layers import Conv2D, MaxPool2D, Flatten, LSTM
from keras.layers import Dropout, Dense, TimeDistributed
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
from python_speech_features import mfcc, logfbank
from tqdm import tqdm
import os
import pandas as pd
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
import librosa


def build_rand_feat():
    X = []
    y = []
    _min, _max = float('inf'), -float('inf')
    for _ in tqdm(range(n_samples)):
        rand_class = np.random.choice(class_dist.index, p=prob_dist)
        file = np.random.choice(df[df.category==rand_class].index)
        rate, wav = wavfile.read('ESC-50-master/audio/' + file)
        label = df.at[file, 'category']
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
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    elif(config.mode == 'time'):
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2])
    y = to_categorical(y, num_classes=50)
    return X,y

def get_conv_model():
    model = Sequential()
    model.add(Conv2D(16, (3,3), activation='relu', strides=(1,1), padding='same', input_shape=input_shape))
    model.add(Conv2D(32, (3,3), activation='relu', strides=(1,1), padding='same', input_shape=input_shape))
    model.add(Conv2D(64, (3,3), activation='relu', strides=(1,1), padding='same', input_shape=input_shape))
    model.add(Conv2D(128, (3,3), activation='relu', strides=(1,1), padding='same', input_shape=input_shape))
    model.add(MaxPool2D((2,2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    return model

def get_recurrent_model():
    model = Sequential()
    model.add(LSTM(128,return_sequences=True, input_shape=input_shape))
    model.add(LSTM(128,return_sequences=True))
    model.add(LSTM(128,return_sequences=True))
    model.add(LSTM(128,return_sequences=True))
    model.add(LSTM(128,return_sequences=True))
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(64, activation='relu')))
    model.add(TimeDistributed(Dense(32, activation='relu')))
    model.add(TimeDistributed(Dense(16, activation='relu')))
    model.add(TimeDistributed(Dense(8, activation='relu')))
    model.add(Flatten())
    model.add(Dense(50, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    return model

class Config:
    def __init__(self, mode='time', nfilt=26, nfeat=13, nfft=1103, rate=44100):
        self.mode = mode
        self.nfilt = nfilt
        self.nfeat = nfeat
        self.nfft = nfft
        self.rate = rate
        self.step = int(rate/10)

df = pd.read_csv("ESC-50-master/meta/esc50.csv")
df = df.set_index('filename')

#sample rate is 44100
#read in the rate and signal of all the audio files
#store the audio data in a dataframe. This dataframe contains all the audio data, their class, and how long the audio file is
for f in df.index:
    rate, signal = wavfile.read("clean/" + f)
    df.at[f,'length'] = signal.shape[0]/rate
    #print(rate)
    #print(signal)

#get a list of all possible classes as well as the distrobution of the number of the particular class found compared to the total number of audio events
classes = list(np.unique(df.category))
class_dist = df.groupby(['category'])['length'].mean()

#set a large samplesize and choose a random class
n_samples = int(df['length'].sum() / 0.4 )
prob_dist = class_dist / class_dist.sum()
#choices = np.random.choice(class_dist.index, p = prob_dist)
#print(choices)

config = Config(mode='time')

if(config.mode == 'time'):
    X, y = build_rand_feat()
    y_flat = np.argmax(y,axis=1)
    input_shape = (X.shape[1], X.shape[2])
    model = get_recurrent_model()
elif(config.mode == 'conv'):
    X, y = build_rand_feat()
    y_flat = np.argmax(y,axis=1)
    input_shape = (X.shape[1], X.shape[2], 1)
    model = get_conv_model()

class_weight = compute_class_weight('balanced', np.unique(y_flat), y_flat)

model.fit(X, y, epochs=80, batch_size=16, shuffle=True, class_weight=class_weight)