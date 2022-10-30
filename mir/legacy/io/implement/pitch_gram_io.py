from mir.io.feature_io_base import *
from mir.common import PACKAGE_PATH
import numpy as np
import librosa

class PitchGramIO(FeatureIO):
    def read(self, filename, entry):
        return pickle_read(self, filename)

    def write(self, data, filename, entry):
        pickle_write(self, data, filename)

    def visualize(self, data, filename, entry, override_sr):
        pitch_data=data[1]
        labels=data[0]
        f = open(os.path.join(PACKAGE_PATH, 'data/spectrogram_template.svl'), 'r')
        content = f.read()
        f.close()
        content = content.replace('[__SR__]', str(100))
        content = content.replace('[__WIN_SHIFT__]', str(1))
        content = content.replace('[__SHAPE_1__]', str(pitch_data.shape[1]))
        content = content.replace('[__COLOR__]', str(1))
        content = content.replace('[__DATA__]', create_svl_3d_data(labels, pitch_data))
        f = open(filename, 'w')
        f.write(content)
        f.close()
    def get_visualize_extention_name(self):
        return "svl"