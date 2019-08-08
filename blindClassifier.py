import pickle
import os
import numpy as np
from tqdm import tqdm
from scipy.io import wavfile
from python_speech_features import mfcc
from keras.models import load_model
import pandas as pd
from sklearn.metrics import accuracy_score

def build_predictions(audio_dir):
    y_pred = []
    fn_prob = {}
    print('extracting feature form audio')
    for fn in tqdm(os.listdir(audio_dir)):
        rate, wav = wavfile.read(os.path.join(audio_dir, fn))
        #label = fn2class[fn]
        #c = classes.index(label)
        y_prob = []
        #print(fn)
        for i in range(0, wav.shape[0]-config.step, config.step):
            sample = wav[i:i+config.step]
            #print(sample)
            x = mfcc(sample, rate, numcep=config.nfeat, nfilt=config.nfilt,nfft=config.nfft)
            x = (x - config.min) / (config.max - config.min)
            if(config.mode == 'time'):
                x = x.reshape(1, x.shape[0], x.shape[1])
            #print(x.shape)
            y_hat = model.predict(x)
            y_prob.append(y_hat)
            y_pred.append(np.argmax(y_hat))
            #y_true.append(c)
        fn_prob[fn] = np.mean(y_prob, axis=0).flatten()
    return y_pred, fn_prob

with open('splitHomeMetadata.p', 'rb') as handle:
    metaData = pickle.load(handle)

df = pd.read_csv("data/homedata.csv")
classes = list(np.unique(df.category))
#fn2class = dict(zip(df.filename, df.category))
p_path = os.path.join('bin', 'homedata.p')

with open(p_path, 'rb') as handle:
    config = pickle.load(handle)

model = load_model(config.model_path)

y_pred, fn_prob = build_predictions('cleanHomeData')
#acc_score = accuracy_score(y_true=y_true, y_pred=y_pred)


y_probs = []

filenames = []
filenames.append('')
for file in os.listdir('cleanHomeData'):
    filenames.append(file)
#data needs to be 2D array with structs entry name, then each entry in the array, then the next row, etc
#columns needs to be all the file names
newDf = pd.DataFrame(data=np.nan,columns=['filename','start','end','alarm-siren','coffee-making','door-close','door-open','kettle-heating', 'microwave-heating', 'guess'], index=filenames)

i = 1
for entry in fn_prob:
    rowData = fn_prob.get(entry, '')
    guess = np.argmax(rowData)
    if(rowData[guess] > 0.5):
        guess = classes[guess]
    else:
        guess = 'other event'
    start = 0
    end = 4
    for j in range(0,len(metaData)):
        if(metaData[j]['filename'] == entry):
            start = metaData[j]['start']
            end = start + 4
    row = [entry, start, end, rowData[0], rowData[1], rowData[2], rowData[3], rowData[4], rowData[5], guess]
    newDf.iloc[i] = row
    i = i+1
'''
for i, row in df.iterrows():
    y_prob = fn_prob[row.filename]
    y_probs.append(y_prob)
    #print(type(y_prob))
    for c, p in zip(classes, y_prob):
        df.at[i,c] = p

print(len(y_probs))
print(y_probs)
print(len(classes))
print(classes)
print(y_pred)
print(len(y_pred))
'''
#y_pred = [classes[np.argmax(y)] for y in y_probs]
#df['y_pred'] = y_pred

#df.to_csv('predictions.csv', index=False)
newDf.to_csv('predictions.csv', index=False)