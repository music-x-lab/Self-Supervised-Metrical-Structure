from mir.io.feature_io_base import *
from mir.common import PACKAGE_PATH
import numpy as np
import librosa

class PitchIO(FeatureIO):
    def read(self, filename, entry):
        f = open(filename, 'r')
        lines=f.readlines()
        lines=[line.strip('\n\r') for line in lines]
        lines=[line for line in lines if line!='']
        f.close()
        result=np.zeros((len(lines),3))
        for i in range(len(lines)):
            line=lines[i]
            tokens=line.split('\t')
            assert(len(tokens)==3)
            result[i,0]=float(tokens[0])
            result[i,1]=float(tokens[1])
            result[i,2]=float(tokens[2])
        return result

    def write(self, data, filename, entry):
        f = open(filename, 'w')
        for i in range(0, data.shape[0]):
            f.write('\t'.join([str(item) for item in data[i,:]]))
            f.write('\n')
        f.close()

    def visualize(self, data, filename, entry, override_sr):
        if(np.min(data[:,2])<0):
            output_type='midi'
        else:
            output_type='line'
        if(output_type=='midi'):
            f = open(os.path.join(PACKAGE_PATH, 'data/midi_template.svl'), 'r')
            sr = override_sr
            content = f.read()
            f.close()
            content = content.replace('[__SR__]', str(sr))
            content = content.replace('[__WIN_SHIFT__]', '1')
            last_note=-1
            last_start=0.0
            last_end=0.0
            output_text=''
            for frame in data:
                midi_note=-1
                if(frame[1]>0):
                    midi_note=int(round(librosa.hz_to_midi(frame[1])))
                if(midi_note==last_note and frame[0]-last_end<0.05):
                    last_end=frame[0]
                else:
                    if(last_note>=0):
                        output_text+=self.__get_midi_note_text(last_start*sr,(last_end+0.01)*sr,last_note)
                    last_note=midi_note
                    last_start=frame[0]
                    last_end=frame[0]
            if(last_note>=0):
                output_text += self.__get_midi_note_text(last_start*sr,(last_end+0.01)*sr, last_note)
            content = content.replace('[__DATA__]', output_text)
        elif(output_type=='line'):
            f = open(os.path.join(PACKAGE_PATH, 'data/pitch_template.svl'), 'r')
            sr = override_sr
            content = f.read()
            f.close()
            content = content.replace('[__SR__]', str(sr))
            # voice range
            content = content.replace('[__FMIN__]', str(librosa.midi_to_hz(35.5)))
            content = content.replace('[__FMAX__]', str(librosa.midi_to_hz(81.5)))
            output_text='\n'.join(['<point frame="%d" value="%f" label="" />'%(int(np.round(d[0]*sr)),d[1])
                         for d in data if d[1]>1e-6])
            content = content.replace('[__DATA_FREQ__]', output_text)
            output_text='\n'.join(['<point frame="%d" value="%f" label="" />'%(int(np.round(d[0]*sr)),d[2])
                         for d in data]) if data[:,2].max()-data[:,2].min()>1e-6 else ''
            content = content.replace('[__DATA_ENERGY__]', output_text)
        f = open(filename, 'w')
        f.write(content)
        f.close()

    def __get_midi_note_text(self,start_frame,end_frame,midi_height,level=0.78125):
        return '<point frame="%d" value="%d" duration="%d" level="%f" label="" />\n'\
               %(int(round(start_frame)),midi_height,int(round(end_frame-start_frame)),level)

    def get_visualize_extention_name(self):
        return "svl"