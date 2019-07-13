import os

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