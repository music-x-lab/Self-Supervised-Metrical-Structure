from mir.io.feature_io_base import *
from mir.common import PACKAGE_PATH
from ...chord import Chord


class ChordogramIO(FeatureIO):
    def read(self, filename, entry):
        return pickle_read(self, filename)

    def write(self, data, filename, entry):
        pickle_write(self, data, filename)

    def visualize(self, data, filename, entry, override_sr):
        f = open(os.path.join(PACKAGE_PATH,'data/spectrogram_template.svl'), 'r')
        sr=entry.prop.sr
        win_shift=entry.prop.hop_length
        content = f.read()
        f.close()
        content = content.replace('[__SR__]', str(sr))
        content = content.replace('[__WIN_SHIFT__]', str(win_shift))
        content = content.replace('[__SHAPE_1__]', str(data.shape[1]))
        content = content.replace('[__COLOR__]', str(3))
        labels = [Chord.from_id(i).to_string() for i in range(data.shape[1])]
        content = content.replace('[__DATA__]',create_svl_3d_data(labels,data))
        f=open(filename,'w')
        f.write(content)
        f.close()

    def get_visualize_extention_name(self):
        return "svl"