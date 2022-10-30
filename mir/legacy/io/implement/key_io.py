from mir.io.feature_io_base import *
from legacy.tonalty import *

class KeyIO(FeatureIO):
    def read(self, filename, entry):
        n_frame=entry.prop.n_frame
        f = open(filename, 'r')
        line_list = f.readlines()
        tonalties = [no_tonalty] * n_frame
        sr=entry.prop.sr
        win_shift=entry.prop.hop_length
        for line in line_list:
            tokens = line.replace('\n', '').split('\t')
            if (tokens[2] == 'Key'):
                tonaltyname = tokens[3]
            elif (tokens[2] == 'Silence'):
                tonaltyname = '?'
            else:
                tonaltyname = tokens[2]
            begin = int(round(float(tokens[0]) / (win_shift / sr)))
            end = int(round(float(tokens[1]) / (win_shift / sr)))
            if (end > n_frame):
                end = n_frame
            for i in range(begin, end):
                tonalties[i] = Tonalty.from_string(tonaltyname)
        f.close()
        return tonalties

    def write(self, data, filename, entry):
        raise NotImplementedError()