from keras.datasets import mnist
import numpy as np
import os
from pathlib import Path
from base.base_preprocessor import BasePreprocessor
import librosa
import pretty_midi

class Wav2MidPreprocessor(BasePreprocessor):
    def __init__(self, config):
        super().__init__(config)

        # TODO: for now these are hard coded but move them to config json later
        self.config.preprocessor.sr = 22050
        self.config.preprocessor.hop_length = 512
        self.config.preprocessor.window_size = 7
        self.config.preprocessor.min_midi = 21
        self.config.preprocessor.max_midi = 108
        self.config.preprocessor.bin_multiple = 3
        self.config.preprocessor.window_size = 7

    # read a directory (recursively?) with mp3s and matchings midi files.
    def preprocess(self):
        data_dir = self.config.preprocessor.data_dir.format(self.config)
        raw_data = Path(data_dir) / 'raw'

        for file in os.listdir(raw_data):
            fn, ext = os.path.splitext(file)
            # if both mp3/wav and matching named midi file exists:
            # print(file)
            # print(fn)
            # print(ext)
            # print(Path(raw_data) / f"{fn}.mid")
            if ext in ('.mp3', '.wav') and os.path.exists(Path(raw_data) / f"{fn}.mid"):
                # process and join them
                print(('processing',file,f"{fn}.mid)"))
                wavnp = self.wav2np(Path(raw_data) / file, **self.config.preprocessor)
                times = librosa.frames_to_time(
                    np.arange(wavnp.shape[0]), 
                    sr=self.config.preprocessor.sr, 
                    hop_length=self.config.preprocessor.hop_length)
                midnp = self.mid2np(os.path.join(raw_data,f"{fn}.mid"), times, **self.config.preprocessor)
                
                np.save(Path(self.config.preprocessor.data_dir) / "preprocessed" / f"{fn}_input", wavnp)
                np.save(Path(self.config.preprocessor.data_dir) / "preprocessed" / f"{fn}_output", midnp)

    def wav2np(self, audio_fn, sr, bin_multiple, max_midi, min_midi, window_size, hop_length, **kwargs):        
        bins_per_octave = 12 * bin_multiple #should be a multiple of 12
        n_bins = (max_midi - min_midi + 1) * bin_multiple

        # down-sample, mono-channel
        y, _ = librosa.load(audio_fn,sr)
        S = librosa.cqt(y,fmin=librosa.midi_to_hz(min_midi), sr=sr, hop_length=hop_length,
                          bins_per_octave=bins_per_octave, n_bins=n_bins)
        S = S.T

        # scale logarithmically 
        S = librosa.amplitude_to_db(np.abs(S))
        
        minDB = np.min(S)
        print(np.min(S),np.max(S),np.mean(S))

        S = np.pad(S, ((window_size//2,window_size//2),(0,0)), 'constant', constant_values=minDB)

        windows = []
        for i in range(S.shape[0]-window_size+1):
            w = S[i:i+window_size,:]
            windows.append(w)
        x = np.array(windows)

        return x

    def mid2np(self, fn, times, sr, min_midi, max_midi, **kwargs):
        pm = pretty_midi.PrettyMIDI(fn)
        piano_roll = pm.get_piano_roll(fs=sr,times=times)[min_midi:max_midi+1].T
        piano_roll[piano_roll > 0] = 1
        return piano_roll