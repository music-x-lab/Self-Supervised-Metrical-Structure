from mir.io.feature_io_base import *
import librosa
import soundfile as sf


class MusicIO(FeatureIO):
    def read(self, filename, entry):
        y, sr = librosa.load(filename, sr=entry.prop.sr, mono=True)
        return y  # (y-np.mean(y))/np.std(y)

    def write(self, data, filename, entry):
        sr = entry.prop.sr
        sf.write(filename, data, sr, format='wav')

    def visualize(self, data, filename, entry, override_sr):
        sr = entry.prop.sr
        sf.write(filename, data, sr)

    def get_visualize_extention_name(self):
        return "wav"
