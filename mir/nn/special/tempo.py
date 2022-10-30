from ..data_decorator import AbstractPitchShifter
from ..data_provider import FramedDataProvider,data_type_fix
import cv2
import numpy as np

class SpectrogramStretchedPitchShifter(AbstractPitchShifter):

    def __init__(self,target_size):
        super(SpectrogramStretchedPitchShifter, self).__init__()
        self.target_size=target_size

    def pitch_shift(self,data,shift):
        if(shift!=0):
            raise NotImplementedError()
        return cv2.resize(data.astype(np.float32),(data.shape[1],self.target_size))

class TempoStretchedPitchShifter(AbstractPitchShifter):

    def __init__(self,target_size,base_speed=2.0):
        super(TempoStretchedPitchShifter, self).__init__()
        self.target_size=target_size
        self.base_speed=base_speed

    def pitch_shift(self,data,shift):
        bpm_change=float(data.shape[0])/self.target_size/self.base_speed

        return bpm_change*cv2.resize(data.reshape((-1,1)).astype(np.float32),
                          (1,self.target_size),interpolation=cv2.INTER_).reshape((-1))

class StretchedFramedDataProvider(FramedDataProvider):

    def __init__(self,train_sample_length,speed_count,average_samples_per_song=10,shift_low=0,shift_high=0,base_speed=2.0,
                 slow_limit=np.sqrt(0.5),fast_limit=np.sqrt(2.0),num_workers=0):
        self.slow_limit=slow_limit
        self.fast_limit=fast_limit
        self.speed_count=speed_count
        delta=(np.log2(self.fast_limit)-np.log2(self.slow_limit))/max(speed_count-1,1)
        speed_changes=2**(delta*(np.arange(speed_count,dtype=np.float32)-(speed_count-1)/2))
        self.raw_sample_lengths=sorted([int(np.round(train_sample_length*base_speed/x)) for x in speed_changes])
        print('Input frame counts:',self.raw_sample_lengths)
        super(StretchedFramedDataProvider, self).__init__(
            train_sample_length=self.raw_sample_lengths[0],
            average_samples_per_song=average_samples_per_song,
            shift_low=shift_low,
            shift_high=shift_high,
            num_workers=num_workers
        )

    def get_length(self):
        return len(self.start)*self.average_samples_per_song*(self.shift_high-self.shift_low+1)*self.speed_count

    def get_sample(self,id):
        raw_id=id%self.valid_song_count
        speed_index=(id//self.valid_song_count)%self.speed_count
        raw_length=self.raw_sample_lengths[speed_index]
        if(raw_length>self.length[raw_id]):
            #new_index=np.searchsorted(self.raw_sample_lengths,raw_length,side='right')
            #assert(new_index>0)
            #raw_length=self.raw_sample_lengths[new_index-1]
            # todo: why???
            raw_length=self.length[raw_id]
        shift=(id//self.valid_song_count//self.speed_count)%(self.shift_high-self.shift_low+1)
        sample_id=np.random.randint(self.length[raw_id]-raw_length+1)
        return [
            data_type_fix(pitch_shifter.pitch_shift(
                storage.locate(valid_indices[raw_id],sample_id,raw_length),shift+self.shift_low))
            for (storage,valid_indices,pitch_shifter) in self.storage
        ]
