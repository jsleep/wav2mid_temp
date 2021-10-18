from keras.datasets import mnist
import numpy as np
import os
import Path
from base.base_preprocessor import BasePreprocessor
import librosa

class Mp3Preprocessor(BasePreprocessor):
    def __init__(self, config):
        super().__init__(config)

    # read a directory (recursively?) with mp3s and matchings midi files.
    def preprocess(self):
        data_dir = self.config.preprocessor.data_dir.format(self.config)
        raw_data = Path(data_dir) / 'raw'

        for file in os.listdir(raw_data):
            fn, ext = os.path.splitext(file)
            # if both mp3/wav and matching named midi file exists:
            if ext in ('mp3', 'wav') and os.path.exists(f"{fn}.mid"):
                # process and join them
                wavnp = wav2np(file)
                midnp = mid2np(file)
                data = join(wavnp, midnp)

                np.save(os.path.join(
                    self.config.preprocessor.data_dir,
                    self.config.preprocessor.data_fmt.format(self.config)),
                    data)

    # TODO: for now these are hard coded but move them to 
    sr = 22050
    hop_length = 512
    window_size = 7
    min_midi = 21
    max_midi = 108

    def wav2np(self):
        print("wav2inputnp")
        bins_per_octave = 12 * bin_multiple #should be a multiple of 12
        n_bins = (max_midi - min_midi + 1) * bin_multiple

        # down-sample, mono-channel
        y, _ = librosa.load(audio_fn,sr)
        S = librosa.cqt(y,fmin=librosa.midi_to_hz(min_midi), sr=sr, hop_length=hop_length,
                        bins_per_octave=bins_per_octave, n_bins=n_bins)
        S = S.T

        S = librosa.amplitude_to_db(S)
        S = np.abs(S)
        minDB = np.min(S)
        print(np.min(S),np.max(S),np.mean(S))

        S = np.pad(S, ((window_size//2,window_size//2),(0,0)), 'constant', constant_values=minDB)

        windows = []
        for i in range(S.shape[0]-window_size+1):
            w = S[i:i+window_size,:]
            windows.append(w)
        x = np.array(windows)

        return x

    def mid2np(self, fn):
        raise NotImplementedError()