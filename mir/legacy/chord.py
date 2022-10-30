from mir.music_base import get_scale_and_suffix,scale_name_to_value

class ChordTypeComplexity:
    majmin = 0
    majmininv = 1
    majmin7 = 2
    majmin7inv = majmininv | majmin7
    full = 15
    mirex = 31
    master = 63

class ChordTypes:
    none=0
    x=-1
    maj=1
    min=2
    maj_inv3=3
    maj_inv5=4
    min_invb3=5
    min_inv5=6
    mirex_chord_suffix_list=['N',':maj',':min',':maj/3',':maj/5',':min/b3',':min/5',
                       ':7',':min7',':maj7',':7(#9)',':maj(9)',':maj6',':min9',':maj/2',':maj9',
                       ':min/b7',':11',':maj/4',':maj/b7',':13',':9',
                       ':maj6(9)',':min11',':min/4',':min7/5',':min6',':7/3',':7/5',':7(b9)',':maj/7',':min/6',':7/b7']
    full_chord_suffix_list = ['N', ':maj', ':min', ':maj/3', ':maj/5', ':min/b3', ':min/5',
                         ':7', ':min7', ':maj7', ':7(#9)', ':maj(9)', ':sus4', ':maj6', ':sus4(b7,9)', ':min9', ':sus4(b7)', ':maj9',
                         ':min7/b7', ':11', ':7/b7', ':13', ':9',
                         ':maj6(9)', ':min11', ':sus2', ':min7/5', ':dim', ':min6', ':7/3', ':hdim7', ':7/5', ':7(b9)', ':maj7/7',
                         ':sus4(9)', ':min/6', ':aug']
    full_chord_suffix_note_list=[[],[0,4,7],[0,3,7],[4,7,0],[7,0,4],[3,7,0],[7,0,3],
                            [0,4,7,10],[0,3,7,10],[0,4,7,11],[0,4,7,10,3],[0,4,7,2],[0,4,7,9],[0,3,7,10,2],[2,0,4,7],[0,4,7,11,2],
                            [10,0,3,7],[0,4,7,10,2,5],[5,0,4,7],[10,0,4,7],[0,4,7,5,9],[0,4,7,10,2],[0,4,7,9,2]]
    master_chord_suffix_list=['N',':maj',':min',':maj/3',':maj/5',':min/b3',':min/5'
        ,':7',':min7',':maj7',':maj(9)',':7(#9)',':min9',':sus4',':sus4(b7,9)'
        ,':maj6',':sus4(b7)',':maj/2',':maj9',':11',':9',':maj/4',':13'
        ,':min11',':maj6(9)',':min/4',':sus2',':dim',':min6',':min7/5',':7/5'
        ,':hdim7',':7/3',':min(9)',':min/6',':sus4(9)',':7/b7',':7(b9)',':maj7/2'
        ,':aug(b7)',':maj13',':maj7/7',':min7(11)',':maj6/5',':maj/6',':sus4(b7,9,13)'
        ,':min7/b7',':min(b13)',':7(#11)',':min6/5',':maj(11)',':maj(9)/3',':min(11)'
        ,':min7/4',':sus2/2',':maj(9)/5',':maj7(#11)',':aug']
    '''
    chord_suffix_simplify_majmin=['N',':maj',':min',':maj',':maj',':min',':min',
                       ':maj',':min',':maj',':maj',':maj',':maj',':min',':maj',':maj',
                       ':min',':maj',':maj',':maj',':maj',':maj',
                       ':maj',':min',':min',':min',':min',':maj',':maj',':maj',':maj',':min',':maj']
    chord_suffix_simplify_majmininv = ['N',':maj',':min',':maj/3',':maj/5',':min/b3',':min/5',
                       ':maj',':min',':maj',':maj',':maj',':maj',':min',':maj',':maj',
                       ':min',':maj',':maj',':maj',':maj',':maj',
                       ':maj',':min',':min',':min/5',':min',':maj/3',':maj/5',':maj',':maj',':min',':maj']
    chord_suffix_simplify_majmin7 = ['N', ':maj', ':min', ':maj', ':maj', ':min', ':min',
                         ':7', ':min7', ':maj7', ':7', ':maj', ':maj', ':min7', ':maj', ':maj7',
                         ':min', ':7', ':maj', ':maj', ':7', ':7',
                         ':maj', ':min7', ':min', ':min7', ':min', ':7', ':7', ':7', ':maj',
                         ':min', ':7']
    chord_suffix_simplify_majmin7inv = ['N', ':maj', ':min', ':maj/3', ':maj/5', ':min/b3', ':min/5',
                         ':7', ':min7', ':maj7', ':7', ':maj', ':maj', ':min7', ':maj', ':maj7',
                         ':min', ':7', ':maj', ':maj', ':7', ':7',
                         ':maj', ':min7', ':min', ':min7', ':min', ':7', ':7', ':7', ':maj',
                         ':min', ':7']'''
    @staticmethod
    def __get_maj_min(chord_suffix):
        if('(*3)' in chord_suffix or ':5' in chord_suffix or ':1' in chord_suffix): # 1+5
            maj_min = ChordTypes.x
        elif ('minmaj' in chord_suffix):
            maj_min = ChordTypes.min
        elif ('maj' in chord_suffix):
            maj_min = ChordTypes.maj
        elif('dim' in chord_suffix or 'aug' in chord_suffix or 'sus' in chord_suffix or 'b5' in chord_suffix or '#5' in chord_suffix):
            maj_min = ChordTypes.x
        elif ('m' in chord_suffix or 'b3' in chord_suffix):
            maj_min = ChordTypes.min
        else:
            maj_min = ChordTypes.maj
        return maj_min

    @staticmethod
    def __get_maj_min_inv(chord_suffix):
        maj_min=ChordTypes.__get_maj_min(chord_suffix)
        if(maj_min==ChordTypes.x):
            return ChordTypes.x
        if('/5' in chord_suffix):
            return ChordTypes.min_inv5
        if('/3' in chord_suffix):
            if(maj_min==ChordTypes.maj):
                return ChordTypes.maj_inv3
            else:
                return ChordTypes.x
        if ('/b3' in chord_suffix):
            if (maj_min == ChordTypes.min):
                return ChordTypes.min_invb3
            else:
                print('Weird chord suffix: %s' % chord_suffix)
                return ChordTypes.x
        return maj_min
    @staticmethod
    def from_suffix(chord_suffix,complexity):
        if(complexity==ChordTypeComplexity.majmin):
            return ChordTypes.__get_maj_min(chord_suffix)
        elif(complexity==ChordTypeComplexity.majmininv):
            return ChordTypes.__get_maj_min_inv(chord_suffix)
        elif(complexity==ChordTypeComplexity.full):
            if(chord_suffix!='' and chord_suffix[0]==':'): # Mirex style
                if(chord_suffix==':maj/b7'):
                    chord_suffix=':7/b7'
                if(chord_suffix==':maj/7'):
                    chord_suffix=':maj7/7'
                if(chord_suffix==':min/b7'):
                    chord_suffix=':min7/b7'
                if(chord_suffix not in ChordTypes.full_chord_suffix_list):
                    return ChordTypes.x
                else:
                    return ChordTypes.full_chord_suffix_list.index(chord_suffix)
            else:
                return ChordTypes.__get_maj_min_inv(chord_suffix)
        elif(complexity==ChordTypeComplexity.master):
            if(chord_suffix!='' and chord_suffix[0]==':'): # Mirex style
                if(chord_suffix==':maj/b7'):
                    chord_suffix=':7/b7'
                if(chord_suffix==':maj/7'):
                    chord_suffix=':maj7/7'
                if(chord_suffix==':min/b7'):
                    chord_suffix=':min7/b7'
                if(chord_suffix not in ChordTypes.master_chord_suffix_list):
                    return ChordTypes.x
                else:
                    return ChordTypes.master_chord_suffix_list.index(chord_suffix)
            else:
                return ChordTypes.__get_maj_min_inv(chord_suffix)
        elif(complexity==ChordTypeComplexity.mirex):
            if(chord_suffix!='' and chord_suffix[0]==':'): # Mirex style
                if(chord_suffix not in ChordTypes.mirex_chord_suffix_list):
                    return ChordTypes.x
                else:
                    return ChordTypes.mirex_chord_suffix_list.index(chord_suffix)
            else:
                return ChordTypes.__get_maj_min_inv(chord_suffix)
        else:
            raise NotImplementedError()

    @staticmethod
    def to_string(chordtype, complexity, relative=False):
        assert(isinstance(complexity,int))
        if(chordtype==-1):
            return 'X'
        if(complexity==ChordTypeComplexity.mirex):
            return ChordTypes.mirex_chord_suffix_list[chordtype]
        elif(complexity==ChordTypeComplexity.full):
            return ChordTypes.full_chord_suffix_list[chordtype]
        else:
            return ChordTypes.master_chord_suffix_list[chordtype]



class Chord():

    def __init__(self,scale,type):
        self.scale=scale
        self.type=type

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.scale==other.scale and (self.type)==(other.type)
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    @staticmethod
    def from_string(chord_name,complexity=ChordTypeComplexity.majmin):
        # Absolute chord
        if(chord_name=='N'):
            scale=-1
            type=ChordTypes.none
        elif(chord_name=='X'):
            scale=-1
            type=ChordTypes.x
        else:
            scale,suffix=get_scale_and_suffix(chord_name)
            type=ChordTypes.from_suffix(suffix,complexity=complexity)
            if(type==ChordTypes.x):
                scale=-1
        return Chord(scale,type)

    def shift_pitch(self,pitch):
        if(self.scale==-1):
            return self
        else:
            return Chord((self.scale+pitch%12+12)%12,self.type)

    def to_relative(self,tonalty_do):
        if(self.scale==-1):
            return RelativeChord(-1,self.type)
        else:
            if(tonalty_do==-1):
                raise Exception("Invalid tonalty environment")
            else:
                return RelativeChord((self.scale+12-tonalty_do)%12,self.type)
    def simplify(self,complexity):
        if(self.scale==-1):
            return self
        if(complexity==ChordTypeComplexity.majmin):
            new_type=ChordTypes.from_suffix(ChordTypes.chord_suffix_simplify_majmin[self.type],complexity=complexity)
        elif(complexity==ChordTypeComplexity.majmininv):
            new_type=ChordTypes.from_suffix(ChordTypes.chord_suffix_simplify_majmininv[self.type],complexity=complexity)
        elif (complexity == ChordTypeComplexity.majmin7):
            new_type = ChordTypes.from_suffix(ChordTypes.chord_suffix_simplify_majmin7[self.type],
                                              complexity=ChordTypeComplexity.full)
        elif (complexity == ChordTypeComplexity.majmin7inv):
            new_type = ChordTypes.from_suffix(ChordTypes.chord_suffix_simplify_majmin7inv[self.type],
                                              complexity=ChordTypeComplexity.full)
        else:
            raise NotImplementedError()
        return __class__(self.scale,new_type)
    def to_string(self,complexity=ChordTypeComplexity.full):
        if(self.scale==-1):
            return ChordTypes.to_string(self.type,complexity)
        else:
            template = "C#DbEF#G#AbB"
            result=template[self.scale]
            if(result=='#'):
                result=template[self.scale-1]+result
            elif(result=='b'):
                result=template[self.scale+1]+result
            return result+ChordTypes.to_string(self.type,complexity)

    def get_bass(self,complexity=ChordTypeComplexity.full):
        str=self.to_string(complexity)
        result=self.scale
        if('/' in str):
            result=(result+scale_name_to_value(str[str.index('/')+1:]))%12
        return result

    def to_id(self):
        if(self.type==ChordTypes.x):
            return -1
        elif(self.type==ChordTypes.none):
            return 0
        else:
            return self.scale+self.type*12-11

    @staticmethod
    def from_id(id):
        scale=(id-1)%12
        if(id==-1):
            return __class__(-1,ChordTypes.x)
        elif(id==0):
            return __class__(-1,ChordTypes.none)
        else:
            return __class__(scale,(id+11)//12)

    def to_midi(self):
        if(self.scale==-1):
            return []
        midi_base=48
        notes=ChordTypes.full_chord_suffix_note_list[self.type]
        notes=[note+self.scale+midi_base for note in notes]
        if(notes[2]>notes[1]>notes[0] and notes[2]-notes[0]<=8):
            notes[1],notes[2]=notes[2],notes[1]
            notes.insert(2,notes[0])
        else:
            notes.insert(3,notes[0])
        for i in range(1,len(notes)):
            while(notes[i]<notes[i-1]):
                notes[i]+=12
        return notes

class RelativeChord():

    scale_names_big=["I", "I#", "II", "IIIb", "III", "IV", "IV#", "V", "VIb", "VI", "VIIb", "VII"]
    scale_names_small=["i", "i#", "ii", "iiib", "iii", "iv", "iv#", "v", "vib", "vi", "viib", "vii"]

    def __init__(self,scale,type):
        self.scale=scale
        self.type=type

    @staticmethod
    def from_string(chord_name,complexity=ChordTypeComplexity.majmin):
        # Absolute chord
        if(chord_name=='N'):
            scale=-1
            type=ChordTypes.none
            return RelativeChord(scale, type)
        elif(chord_name=='X'):
            scale=-1
            type=ChordTypes.x
            return RelativeChord(scale, type)
        else:
            for i in range(12):
                if(RelativeChord.scale_names_big[i]==chord_name):
                    scale=i
                    type=ChordTypes.maj
                    return RelativeChord(scale,type)
                elif(RelativeChord.scale_names_small[i]==chord_name):
                    scale=i
                    type=ChordTypes.min
                    return RelativeChord(scale,type)
        raise NotImplementedError()


    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.scale==other.scale and self.type==other.type
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return self.to_string()

    def to_absolute(self,tonalty_do):
        if(self.scale==-1):
            return Chord(-1, self.type)
        else:
            if(tonalty_do==-1):
                raise Exception("Invalid tonalty environment")
            else:
                return Chord((tonalty_do+self.scale)%12,self.type)

    def to_string(self,complexity=ChordTypeComplexity.full):
        if(self.scale==-1):
            return ChordTypes.to_string(self.type,complexity)
        else:
            if(self.type==ChordTypes.maj):
                return self.scale_names_big[self.scale]
            elif(self.type==ChordTypes.min):
                return self.scale_names_small[self.scale]
            else:
                return NotImplementedError()

    def to_id(self):
        raise NotImplementedError()
        if(self.type==ChordTypes.x):
            return -1
        elif(self.type==ChordTypes.none):
            return 0
        elif (self.type == ChordTypes.maj):
            return self.scale+1
        elif (self.type == ChordTypes.min):
            return self.scale+13

    @staticmethod
    def from_id(id):
        raise NotImplementedError()#
        if(id==-1):
            return __class__(-1,ChordTypes.x)
        elif(id==0):
            return __class__(-1,ChordTypes.none)
        elif(id>0 and id<=12):
            return __class__(id-1,ChordTypes.maj)
        elif(id>12 and id<=24):
            return __class__(id-13,ChordTypes.min)

Chord.no_chord=Chord.from_string('N')
RelativeChord.no_chord=RelativeChord.from_string('N')
Chord.x_chord=Chord.from_string('X')
RelativeChord.x_chord=RelativeChord.from_string('X')

def chord_compare(c1,c2,complexity=ChordTypeComplexity.majmin):
    return Chord.from_string(c1.to_string(),complexity=complexity)==Chord.from_string(c2.to_string(),complexity=complexity)

if __name__=="__main__":
    # perform some tests
    Am=Chord.from_string("Am")
    Cmaj7=Chord.from_string("C:maj7",complexity=ChordTypeComplexity.full)
    EbminMaj7=Chord.from_string("EbmM7")
    Ebmin7=Chord.from_string("D#m7")
    print(EbminMaj7.to_string())
    print(Ebmin7.to_string())
    print('Am=',Am.to_string())
    print(Cmaj7.to_string())
    print(Ebmin7==EbminMaj7)
    #print('vi=',Am.to_relative(0).to_string())
    #print('vib=',Am.to_relative(1).to_string())
    #print('v=',Am.to_relative(2).to_string())
    vi=Am.to_relative(0)
    Vmaj7=Cmaj7.to_relative(5)
    #print(Vmaj7.to_string())
    print(Vmaj7.to_absolute(0).to_string())
    print(Vmaj7.to_absolute(7).to_string())
    X=Chord.from_string("X")
    N=Chord.from_string("N")
    print('False=',N==X)
    #print('X=',X.to_relative(4).to_string())
    print('N=',N.to_relative(4).to_absolute(7).to_string())

    Am=Chord.from_string('Am/b3',complexity=ChordTypeComplexity.majmin)
    Am3=Chord.from_string('Am/b3',complexity=ChordTypeComplexity.majmininv)
    print('Am=',Am.to_string())
    print('Am/b3=', Am3.to_string())
    print('Am=',Am.to_string())
    print('False=', Am==Am3)
    C5=Chord.from_string('C/5',complexity=ChordTypeComplexity.majmininv)
    print('C/5=',C5.to_string())

    Cmaj75=Chord.from_string("C:maj7/5",complexity=ChordTypeComplexity.full)
    print('CM7/5 (or X)=',Cmaj75.to_string())
    Cmajadd9=Chord.from_string("C:maj(9)",complexity=ChordTypeComplexity.full)
    print('Cadd9 (or X)=',Cmajadd9.to_string())
    print('True=',chord_compare(Cmajadd9,Cmaj7,complexity=ChordTypeComplexity.majmininv))
    print('False=',chord_compare(Cmajadd9,Cmaj7,complexity=ChordTypeComplexity.full))