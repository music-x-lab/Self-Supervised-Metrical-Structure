from mir.io.feature_io_base import *
import numpy as np

class BeatIO(FeatureIO):
    def read(self, filename, entry):
        sr=entry.prop.sr
        beat_hop_length=entry.prop.beat_hop_length
        n_frame=entry.prop.n_frame
        f = open(filename, 'r')
        line_list = f.readlines()
        beats = [0] * int(entry.music.shape[0]/beat_hop_length)
        for line in line_list:
            tokens = line.replace('\n', '').replace('  ','\t').split('\t')
            frame = int(round(float(tokens[0]) / (beat_hop_length / sr)))
            if (frame < n_frame):
                beats[frame]=int(tokens[-1])
        f.close()
        return beats

    def write(self, data, filename, entry):
        sr=entry.prop.sr
        beat_hop_length=entry.prop.beat_hop_length
        beats = data
        f = open(filename, 'w')
        result = '0\t(b)\n'
        for i in range(1, len(beats)):
            if (beats[i] != 0):
                time = beat_hop_length * i / sr
                result=result+str(time)+'\t'+str(beats[i])+'\n'
        f.write(result)
        f.close()