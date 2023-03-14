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
        mfcc_root_path = root_path + 'mfcc_chunk/'
        total_count = 0
        for uuid in tqdm(list(uuid_set)):
            # wav_path = root_path + uuid + '.wav'
            # y, sr = librosa.load(wav_path)
            # S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)
            # log_S = librosa.power_to_db(S, ref=np.max)
            # mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=13)
            # fig = plt.Figure()
            # canvas = FigureCanvas(fig)
            # ax = fig.add_subplot(111)
            # p = librosa.display.specshow(mfcc, ax=ax)
            # fig_path = mfcc_root_path + uuid + '.png'
            # fig.savefig(fig_path)
            wav_path = root_path + uuid + '.wav'
            y, sr = librosa.load(wav_path)
            y = librosa.resample(y, sr, sr // 2)
            sr = sr // 2
            chunk_size = 2 * sr  # 2 seconds per chunk
            chunks = librosa.util.frame(y, frame_length=chunk_size, hop_length=chunk_size)
            for chunk in chunks.T:
                total_count += 1
                S = librosa.feature.melspectrogram(y=chunk, sr=sr, n_mels=128)
                log_S = librosa.power_to_db(S, ref=np.max)
                mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=13)

        print(total_count)

    elif sys.argv[1] == 'wav2spec':
        spec_root_path = root_path + 'spec_chunk/'
        total_count = 0
        for uuid in tqdm(list(uuid_set)):
            wav_path = root_path + uuid + '.wav'
            y, sr = librosa.load(wav_path)
            sr_new = 11025
            y = librosa.resample(y, sr, sr_new)
            sr = sr_new
            chunk_size = 2 * sr  # 2 seconds per chunk
            y_padded = librosa.util.fix_length(y, ((len(y) // chunk_size) + 1)* chunk_size)
            chunks = librosa.util.frame(y_padded, frame_length=chunk_size, hop_length=chunk_size)
            for i, chunk in enumerate(chunks.T):
                total_count += 1
                fig = plt.Figure()
                canvas = FigureCanvas(fig)
                ax = fig.add_subplot(111)
                S = librosa.feature.melspectrogram(y=chunk, sr=sr, n_mels=128)
                log_S = librosa.power_to_db(S, ref=np.max)
                librosa.display.specshow(log_S, sr=sr, ax=ax)
                fig_path = spec_root_path + uuid + '-' + str(i) + '.png'
                fig.savefig(fig_path)
        print(total_count)
    