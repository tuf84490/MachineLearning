from sklearn.utils.class_weight import compute_class_weight
from python_speech_features import mfcc, logfbank
from tqdm import tqdm
import os
import pandas as pd
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
import librosa
import pickle
'''
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
'''
#graph the signals
def plot_signals():
    Time=np.linspace(0, len(signal)/rate, num=len(signal))

    plt.figure(1)
    plt.title('Signal Wave of test file (with filter)')
    plt.xlabel('Time (s)')
    plt.ylabel('frequency')
    plt.plot(Time,signal)
    plt.show()

rate, signal = wavfile.read("data/recording-0.wav")

#mask = envelope(signal, rate, 100)#this works at filtering out silence but thats actually not what I want because then I cant tell when events start and when they end
#signal = signal[mask]

print(rate)
print(np.info(signal))

plot_signals()

data_size = len(signal)
#focus_size = int(0.15 * rate)
length = int(4 * rate)  #rate is how many samples per seccond, so for 5 seconds we will need the number of samples x 5 ahead of the peak
min_val = 1000
#noise_thresh = 500
focuses = []
#distances = []
sub_signals = []
idx = 0
while idx < data_size:
    if( abs(signal[idx]) > min_val):
        point = {
            'index' : idx,
            'signal' : signal[idx]
        }
        focuses.append(point)
        '''
        for j  in range(idx+1, data_size):
            if( abs(signal[j]) < noise_thresh ):
                length = j - idx
                break
        '''
        i = 0
        sub_signal = []
        while(i < length):
            sub_signal.append(signal[idx])
            idx += 1
            i += 1
        numpyArray = np.asarray(sub_signal)
        sub_file = {
            'data' : numpyArray,
            'start' : idx/rate,
            'end' : idx/rate + 4 * rate
        }
        sub_signals.append(sub_file)
        print('signal peaked at : ' + str(point['index']/rate))
        #print(sub_signal)
    else:
        idx += 1

#print(focuses)
#print(sub_signals)

path = 'splitData/'
fileIndicator = 1000

metaData = []
for i in range(0, len(sub_signals)):
    fileString = str(fileIndicator + i) + '.wav'
    print('found an event and saving it to file : ' + fileString)
    wavfile.write(path + fileString, rate, sub_signals[i]['data'])
    mdata = {
        'filename' : fileString,
        'start' : sub_signals[i]['start'],
        'end' : sub_signals[i]['end'],
        'data' : sub_signals[i]['data']
    }
    metaData.append(mdata)


with open('splitHomeMetadata.p', 'wb') as handle:
    pickle.dump(metaData, handle)