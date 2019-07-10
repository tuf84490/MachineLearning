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
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import librosa
import pickle

def check_data():
    if os.path.isfile(config.p_path):
        print('loading in model {}'.format(config.name))
        with open(config.p_path, 'rb') as handle:
            tmp = pickle.load(handle)
            return tmp
    else:
        return None
def build_rand_feat():
    tmp = check_data()
    if tmp:
        return tmp.data[0], tmp.data[1]
    X = []
    y = []
    _min, _max = float('inf'), -float('inf')
    for _ in tqdm(range(n_samples)):
        rand_class = np.random.choice(class_dist.index, p=prob_dist)
        file = np.random.choice(df[df.category==rand_class].index)
        rate, wav = wavfile.read('data/audioFiles/' + file)
        label = df.at[file, 'category']
        rand_index = np.random.randint(0, wav.shape[0]-config.step)
        sample = wav[rand_index:rand_index+config.step]
        X_sample = mfcc(sample, rate, numcep=config.nfeat, nfilt=config.nfilt,nfft=config.nfft)
        _min = min(np.amin(X_sample), _min)
        _max = max(np.amax(X_sample), _max)
        print(X_sample)
        X.append(X_sample)
        y.append(classes.index(label))
    config.min = _min
    config.max = _max
    X,y = np.array(X), np.array(y)
    X = (X-_min) / (_max - _min)
    if config.mode == 'conv':
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    elif(config.mode == 'time'):
        print(X.ndim)
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2])
    y = to_categorical(y, num_classes=6)   #here is where you change the number of classes
    config.data = (X,y)
    with open(config.p_path, 'wb') as handle:
        pickle.dump(config,handle,protocol=2)
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
    model.add(Dense(6, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    return model

class Config:
    def __init__(self, mode='time', nfilt=26, nfeat=13, nfft=1103, rate=44100, name='default'):
        self.name = name
        self.mode = mode
        self.nfilt = nfilt
        self.nfeat = nfeat
        self.nfft = nfft
        self.rate = rate
        self.step = int(rate/10)
        self.model_path = os.path.join('models', name + '.model')
        self.p_path = os.path.join('bin', name + '.p')

df = pd.read_csv("data/homedata.csv")
df = df.set_index('filename')

#sample rate is 44100
#read in the rate and signal of all the audio files
#store the audio data in a dataframe. This dataframe contains all the audio data, their class, and how long the audio file is
for f in df.index:
    rate, signal = wavfile.read("data/audioFiles/" + f)
    df.at[f,'length'] = signal.shape[0]/rate
    #print(rate)
    #print(signal)

#get a list of all possible classes as well as the distrobution of the number of the particular class found compared to the total number of audio events
classes = list(np.unique(df.category))
class_dist = df.groupby(['category'])['length'].mean()

#set a large samplesize and choose a random class
n_samples = 2 * int(df['length'].sum() / 0.1 )
prob_dist = class_dist / class_dist.sum()
#choices = np.random.choice(class_dist.index, p = prob_dist)
#print(choices)

config = Config(mode='time', name='homedata')

if(config.mode == 'time'):
    X, y = build_rand_feat()
    y_flat = np.argmax(y,axis=1)
    print(X.ndim)
    input_shape = (X.shape[1], X.shape[2])
    model = get_recurrent_model()
elif(config.mode == 'conv'):
    X, y = build_rand_feat()
    y_flat = np.argmax(y,axis=1)
    input_shape = (X.shape[1], X.shape[2], 1)
    model = get_conv_model()

class_weight = compute_class_weight('balanced', np.unique(y_flat), y_flat)

checkpoint = ModelCheckpoint(config.model_path, monitor='val_acc', verbose=1, mode='max',save_best_only=True, save_weights_only=False, period=1)

model.fit(X, y, epochs=40, batch_size=16, shuffle=True, class_weight=class_weight, validation_split=0.1, callbacks=[checkpoint])
model.save(config.model_path)