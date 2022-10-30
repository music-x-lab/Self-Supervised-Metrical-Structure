from mir.extractors.extractor_base import *
from mir.common import WORKING_PATH,SONIC_ANNOTATOR_PATH,PACKAGE_PATH
from mir.cache import hasher
import subprocess
from ..chord import Chord,ChordTypeComplexity
from mir.extractors.vamp_extractor import rewrite_extract_n3

class Chordino(ExtractorBase):

    def get_feature_class(self):
        return io.ChordIO

    def extract(self,entry,**kwargs):
        type=ChordTypeComplexity.majmin
        print('Chordino working on entry '+entry.name)
        length=entry.prop.n_frame
        music_io = io.MusicIO()
        temp_path=os.path.join(WORKING_PATH,'temp/chordino_extractor_%s.wav'%hasher(entry.name))
        temp_n3_path=temp_path+'.n3'
        rewrite_extract_n3(entry,os.path.join(PACKAGE_PATH,'data/chordino.n3'),temp_n3_path)
        music_io.write(entry.music,temp_path,entry)
        proc=subprocess.Popen([SONIC_ANNOTATOR_PATH,
                               '-t',temp_n3_path,
                               temp_path,
                               '-w','lab','--lab-stdout'
                               ],stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
        # print('Begin processing')
        tags=[Chord.no_chord]*length
        last_begin=0
        last_chord=Chord.no_chord
        for line in proc.stdout:
            line=bytes.decode(line)
            if(line.endswith('\r\n')):
                line=line[:len(line)-2]
            if (line.endswith('\r')):
                line=line[:len(line)-1]

            tokens=line.split('\t')
            begin = int(round(float(tokens[0]) * entry.prop.sr/entry.prop.hop_length))
            #todo: check if time needs shift
            if (begin < 0):
                begin = 0
            for i in range(last_begin,begin):
                tags[i]=last_chord
            last_begin=begin
            tokens[1]=tokens[1][1:len(tokens[1])-1]
            # print(tokens[1])
            last_chord=Chord.from_string(tokens[1],complexity=type)
        for i in range(last_begin,len(tags)):
            tags[i]=last_chord
        try:
            os.unlink(temp_path)
            os.unlink(temp_n3_path)
        except:
            pass
        if(len(tags)==0):
            raise Exception('Empty response')
        return tags

