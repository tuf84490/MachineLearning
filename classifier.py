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
    y_true = []
    y_pred = []
    fn_prob = {}
    print('extracting feature form audio')
    for fn in tqdm(os.listdir(audio_dir)):
        rate, wav = wavfile.read(os.path.join(audio_dir, fn))
        label = fn2class[fn]
        c = classes.index(label)
        y_prob = []

        for i in range(0, wav.shape[0]-config.step, config.step):
            sample = wav[i:i+config.step]
            x = mfcc(sample, rate, numcep=config.nfeat, nfilt=config.nfilt,nfft=config.nfft)
            x = (x - config.min) / (config.max - config.min)
            if(config.mode == 'time'):
                x = np.expand_dims(x, axis=0)
            print(x)
            y_hat = model.predict(x)
            y_prob.append(y_hat)
            y_pred.append(np.argmax(y_hat))
            y_true.append(c)
        fn_prob[fn] = np.mean(y_prob, axis=0).flatten()
    return y_true, y_pred, fn_prob

df = pd.read_csv("ESC-50-master/meta/esc50.csv")
classes = list(np.unique(df.category))
fn2class = dict(zip(df.filename, df.category))
p_path = os.path.join('bin', 'ESC_50.p')
print(p_path)
with open(p_path, 'rb') as handle:
    config = pickle.load(handle)
    print(config)

model = load_model(config.model_path)

y_true, y_pred, fn_prob = build_predictions('clean')
acc_score = accuracy_score(y_true=y_true, y_pred=y_pred)

y_probs = []
for i, row in df.iterrows():
    y_prob = fn_prob[row.filename]
    y_probs.append(y_prob)
    for c, p in zip(classes, y_prob):
        df.at[i,c] = p

y_pred = [classes[np.argmax(y)] for y in y_probs]
df['y_pred'] = y_pred

df.to_csv('predictions.csv', index=False)