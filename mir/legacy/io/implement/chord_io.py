from mir.io.feature_io_base import *
from legacy.chord import *
from mir.music_base import get_scale_and_suffix
from mir.common import PACKAGE_PATH
from collections import defaultdict

class ChordIO(FeatureIO):

    def __init__(self,chord_class=Chord):
        self.chord_class=chord_class
        self.loaded_chord_count=defaultdict(int)

    def read(self, filename, entry):
        n_frame=entry.prop.n_frame
        f = open(filename, 'r')
        line_list = f.readlines()
        tags = [self.chord_class.x_chord] * n_frame
        for line in line_list:
            line=line.strip()
            if(line==''):
                continue
            if ('\t' in line):
                tokens = line.split('\t')
            else:
                tokens = line.split(' ')
            sr=entry.prop.sr
            win_shift=entry.prop.hop_length
            begin=int(round(float(tokens[0])*sr/win_shift))
            end = int(round(float(tokens[1])*sr/win_shift))
            if (end > n_frame):
                end = n_frame
            if(begin<0):
                begin=0
            if(tokens[2]=='N'):
                self.loaded_chord_count['N']+=end-begin
            elif(tokens[2]!='X'):
                self.loaded_chord_count[get_scale_and_suffix(tokens[2])[1]]+=end-begin
            tags[begin:end]=[self.chord_class.from_string(tokens[2],complexity=
                ChordTypeComplexity.__dict__[entry.prop.chord_dict])]*(end-begin)
        f.close()
        return tags

    def write(self, data, filename, entry):
        raise NotImplementedError()
        chords = data
        f = open(filename, 'w')
        result = ''
        last_chord_tag=str(chords[0])
        last_time=0
        sr=entry.prop.sr
        win_shift=entry.prop.hop_length
        for i in range(1, len(chords)):
            if (chords[i] != chords[i - 1]):
                time = win_shift * i / sr
                result=result+str(last_time)+'\t'+str(time)+'\t'+last_chord_tag+'\n'
                last_time=time
                last_chord_tag=str(chords[i])
        time=win_shift*len(chords)/sr
        result = result + str(last_time) + '\t' + str(time) + '\t' + last_chord_tag + '\n'
        f.write(result)
        f.close()

    def visualize(self, data, filename, entry, override_sr):
        chords = data
        sr = override_sr
        win_shift=entry.prop.hop_length/entry.prop.sr*override_sr
        chord_dict=ChordTypeComplexity.__dict__[entry.prop.chord_dict]
        f = open(os.path.join(PACKAGE_PATH,'data/sparse_tag_template.svl'), 'r')
        content = f.read()
        f.close()
        content = content.replace('[__SR__]', str(sr))
        content = content.replace('[__STYLE__]', str(1))
        results=[]
        for i in range(0, len(chords)):
            if (i==0 or chords[i] != chords[i - 1]):
                results.append('<point frame="%d" label="%s" />'%(int(win_shift*i),chords[i].to_string(chord_dict)))
        content=content.replace('[__DATA__]','\n'.join(results))
        f=open(filename,'w')
        f.write(content)
        f.close()

    def auralize_with_beat(self, data, filename, entry, beats):
        chords = data
        last_time=0
        last_id=0
        sr=entry.prop.sr
        win_shift=entry.prop.hop_length
        f = open(os.path.join(PACKAGE_PATH,'data/midi_template.svl'), 'r')
        content = f.read()
        f.close()
        content = content.replace('[__SR__]', str(sr))
        content = content.replace('[__WIN_SHIFT__]', str(win_shift))
        content_data=''
        beat_count=0
        for i in range(1, len(beats)):
            if (beats[i] > 0):
                time = win_shift * i / sr
                if(beat_count%2==0):
                    if(beat_count==0):
                        root_volume=0.8
                    else:
                        root_volume=0.6
                    content_data += self.__append_chord_notes(chords[i - 1], int(last_time * sr),
                                                              int(time * sr - win_shift), sr, root_volume, True, False)
                    content_data += self.__append_chord_notes(chords[i - 1], int(last_time * sr),
                                                              int(time * sr - win_shift), sr, 0.3, False, True)
                else:
                    intertime=(last_time+time)/2
                    interid=(last_id+i)//2
                    content_data += self.__append_chord_notes(chords[interid - 1], int(last_time * sr),
                                                              int((time * sr - win_shift)), sr, 0.4, False, True)
                    content_data += self.__append_chord_notes(chords[i - 1], int(intertime * sr),
                                                              int((time * sr - win_shift)), sr, 0.4, True, False)
                last_time=time
                last_id=i
                beat_count+=1
                if(chords[i]!=chords[i-1]):
                    beat_count=0
        content=content.replace('[__DATA__]',content_data)
        f=open(filename,'w')
        f.write(content)
        f.close()


    def auralize(self, data, filename, entry):
        chords = data
        last_time=0
        sr=entry.prop.sr
        win_shift=entry.prop.hop_length
        f = open(os.path.join(PACKAGE_PATH,'data/midi_template.svl'), 'r')
        content = f.read()
        f.close()
        content = content.replace('[__SR__]', str(sr))
        content = content.replace('[__WIN_SHIFT__]', str(win_shift))
        content_data=''
        for i in range(1, len(chords)):
            if (chords[i] != chords[i - 1]):
                time = win_shift * i / sr
                content_data+=self.__append_chord_notes(chords[i-1],int(last_time*sr),int(time*sr-win_shift),sr)
                last_time=time
        time=win_shift*len(chords)/sr
        content_data +=self.__append_chord_notes(chords[len(chords)-1],int(last_time*sr),int(time*sr-win_shift),sr)
        content=content.replace('[__DATA__]',content_data)
        f=open(filename,'w')
        f.write(content)
        f.close()

    def __append_chord_notes(self,chord,start,end,sr,volume=1.0,lower=True,upper=True):
        delta=int(0.00*sr)
        notes=chord.to_midi(lower,upper)
        return (''.join(['<point frame="%d" value="%d" duration="%d" level="%f" label="" />\n'%
                (start+i*delta,note,end-start-i*delta,volume*0.393701)
                         for (note,i) in zip(notes,range(len(notes)))]))

    def get_visualize_extention_name(self):
        return "svl"