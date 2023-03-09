import librosa
import librosa.display
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join
from pydub import AudioSegment
import sys
from tqdm import tqdm

class WavConverter:
    def __init__(self, root_path, file_name):
        self.root_path = root_path # e.g. './coughvid_20211012/'
        self.file_name = file_name # e.g. 'XXXX.ogg'
        self.input_file = root_path+file_name # e.g. './coughvid_20211012/XXXX.ogg'

    def convert_to_wav(self):
        # Audio conversion code goes here
        # This method converts the input file to .wav format
        uuid, file_type = self.file_name.split('.')
        if file_type == 'ogg':
            ogg_audio = AudioSegment.from_ogg(self.input_file)
            ogg_audio.export(self.root_path + uuid + ".wav", format="wav")
        else: # webm
            webm_audio = AudioSegment.from_file(self.input_file, format="webm")
            webm_audio.export(self.root_path + uuid + ".wav", format="wav")

if __name__ == '__main__':
    metadata = pd.read_csv('./coughvid_20211012/metadata_compiled.csv')
    metadata = metadata[["uuid", "cough_detected","age","gender","status"]]
    metadata = metadata.dropna()
    metadata = metadata[(metadata['cough_detected']>=0.9)]
    uuid_set = set(metadata['uuid'])
    audiofiles = [f for f in listdir('./coughvid_20211012/') if isfile(join('./coughvid_20211012/', f))]
    root_path = './coughvid_20211012/'

    if sys.argv[1] == 'convert2wav':
        for f in tqdm(audiofiles): # f: XXXX.ogg
            uuid, file_type = f.split('.')
            if uuid not in uuid_set:
                continue
            if file_type == 'ogg' or file_type == 'webm':
                WavConverter(root_path, f).convert_to_wav()
    
    elif sys.argv[1] == 'wav2mfcc':
        mfcc_root_path = root_path + 'mfcc/'
        for uuid in tqdm(list(uuid_set)):
            wav_path = root_path + uuid + '.wav'
            y, sr = librosa.load(wav_path)
            S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)
            log_S = librosa.power_to_db(S, ref=np.max)
            mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=13)
            fig = plt.Figure()
            canvas = FigureCanvas(fig)
            ax = fig.add_subplot(111)
            p = librosa.display.specshow(mfcc, ax=ax)
            fig_path = mfcc_root_path + uuid + '.png'
            fig.savefig(fig_path)

    elif sys.argv[1] == 'wav2spec':
        spec_root_path = root_path + 'spec/'
        for uuid in tqdm(list(uuid_set)):
            wav_path = root_path + uuid + '.wav'
            y, sr = librosa.load(wav_path)
            fig = plt.Figure()
            canvas = FigureCanvas(fig)
            ax = fig.add_subplot(111)
            S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)
            log_S = librosa.power_to_db(S, ref=np.max)
            librosa.display.specshow(log_S, sr=sr, ax=ax)
            fig_path = spec_root_path + uuid + '.png'
            fig.savefig(fig_path)
    