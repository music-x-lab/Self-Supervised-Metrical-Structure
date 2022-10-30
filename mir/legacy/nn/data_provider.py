import torch.utils.data as torch_data
from .data_storage import FramedStorage
from .data_decorator import AbstractPitchShifter,NoPitchShifter,data_type_fix
import numpy as np

class DataProvider(torch_data.Dataset):

    def __init__(self):
        super(DataProvider, self).__init__()

    def get_length(self):
        raise NotImplementedError()

    def get_sample(self,id):
        raise NotImplementedError()

    def __len__(self):
        return self.get_length()

    def __getitem__(self, item):
        return self.get_sample(item)

    def need_shuffle(self):
        raise NotImplementedError()

class FramedDataProvider(DataProvider):

    def __init__(self,train_sample_length,average_samples_per_song=10,shift_low=0,shift_high=0):
        super(FramedDataProvider, self).__init__()
        self.train_sample_length=train_sample_length
        self.average_samples_per_song=average_samples_per_song
        self.shift_low=shift_low
        self.shift_high=shift_high
        self.start=None
        self.length=None
        self.storage=[]
        self.valid_song_count=-1

    def need_shuffle(self):
        return True

    def link(self,storage:FramedStorage,pitch_shifter:AbstractPitchShifter=None,subrange=None):
        if(pitch_shifter is None):
            pitch_shifter=NoPitchShifter()
        total_song_count=storage.get_length()
        if(subrange is None):
            subrange=np.arange(total_song_count)
        if(len(self.storage)==0):
            valid_indices=subrange[storage.length[subrange]>=self.train_sample_length]
            self.valid_song_count=len(valid_indices)
            self.start=storage.start[valid_indices]
            self.length=storage.length[valid_indices]
            if(self.valid_song_count==0):
                print('Warning: No valid song was detected in %s'%self.__class__.__name__)
        storage.get_length()
        valid_indices=subrange[storage.length[subrange]>=self.train_sample_length]
        new_start=storage.start[valid_indices]
        if(len(new_start)!=self.valid_song_count):
            raise Exception('Inconsistent song count encountered in %s'%self.__class__.__name__)

        if(np.any(self.start!=new_start)):
            raise Exception('Inconsistent data lengths encountered in %s'%self.__class__.__name__)
        self.storage.append((storage,valid_indices,pitch_shifter))

    def get_length(self):
        return len(self.start)*self.average_samples_per_song*(self.shift_high-self.shift_low+1)

    def get_sample(self,id):
        raw_id=id%self.valid_song_count
        shift=(id//self.valid_song_count)%(self.shift_high-self.shift_low+1)
        sample_id=np.random.randint(self.length[raw_id]-self.train_sample_length+1)
        return [
            data_type_fix(pitch_shifter.pitch_shift(
                storage.locate(valid_indices[raw_id],sample_id,self.train_sample_length),shift+self.shift_low))
            for (storage,valid_indices,pitch_shifter) in self.storage
        ]

