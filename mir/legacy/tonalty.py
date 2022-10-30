from mir.music_base import get_scale_and_suffix

class TonaltyType:
    none=0
    x=1
    minor=2
    major=3
    dorian=4
    mixolydian=5
    aeolian=6
    __relative_do_positions=[-1,-1,3,0,10,5,3]

    @staticmethod
    def from_suffix(tonalty_suffix):
        if (tonalty_suffix == ' Maj' or tonalty_suffix == ''):
            return TonaltyType.major
        elif(tonalty_suffix==' min' or tonalty_suffix==':minor'):
            return TonaltyType.minor
        elif(tonalty_suffix==':dorian'):
            return TonaltyType.dorian
        elif(tonalty_suffix==':mixolydian'):
            return TonaltyType.mixolydian
        elif(tonalty_suffix==':aeolian'):
            return TonaltyType.aeolian

    @staticmethod
    def get_relative_do_position(tonalty_type):
        return TonaltyType.__relative_do_positions[tonalty_type]


class Tonalty:

    def __init__(self,do_pos,type):
        self.do_pos=do_pos
        self.type=type


    def shift_pitch(self,pitch):
        if(self.do_pos==-1):
            return self
        else:
            return Tonalty((self.do_pos+pitch%12+12)%12,self.type)

    @staticmethod
    def from_string(tonaltyname):
        if(tonaltyname=='?'):
            do_pos=-1
            type=TonaltyType.none
        else:
            scale,suffix=get_scale_and_suffix(tonaltyname)
            type=TonaltyType.from_suffix(suffix)
            relative_do_pos=TonaltyType.get_relative_do_position(type)
            do_pos=(scale+relative_do_pos)%12
        return Tonalty(do_pos,type)

    def to_string(self):
        if(self.do_pos==-1):
            return '?'
        else:
            template = "C#DbEF#G#AbB"
            result=template[self.do_pos]
            if(result=='#'):
                result=template[self.do_pos-1]+result
            elif(result=='b'):
                result=template[self.do_pos+1]+result
            return '1='+result

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.do_pos==other.do_pos and self.type==other.type
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return self.to_string()

no_tonalty=Tonalty.from_string('?')

if __name__=="__main__":
    # perform some tests
    Cmajor=Tonalty.from_string('C Maj')
    Cmajor2=Tonalty.from_string('C')
    Dminor=Tonalty.from_string('D min')
    Dminor2=Tonalty.from_string('D:minor')
    Fmixolydian=Tonalty.from_string('F:mixolydian')
    print('True=',Cmajor==Cmajor2)
    print('True=',Dminor==Dminor2)
    print('False=',Cmajor==Dminor)
    print('0=',Cmajor.do_pos)
    print('5=',Dminor.do_pos)
    print(Fmixolydian.do_pos)