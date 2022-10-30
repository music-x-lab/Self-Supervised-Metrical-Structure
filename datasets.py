import os
from mir import DataPool, DataEntry, io
from settings import *
from io_new.osu_io import OsuMapIO
from io_new.downbeat_io import DownbeatIO
from io_new.chordlab_io import ChordLabIO


def set_default_dataset_properties(dataset):
    dataset.set_property('sr', DEFAULT_SR)
    dataset.set_property('hop_length', DEFAULT_HOP_LENGTH)
    dataset.set_property('win_size', DEFAULT_WIN_SIZE)

def create_rwc_midi_synthesized_dataset(hop_length=DEFAULT_HOP_LENGTH):
    rwc = DataPool('rwc_midi')
    set_default_dataset_properties(rwc)
    rwc.set_property('hop_length', hop_length)
    rwc.append_folder(os.path.join(RWC_DATASET_PATH, 'MidiSynthAudio'), '.mp3', io.MusicIO, 'music')
    rwc.append_folder(os.path.join(RWC_DATASET_PATH, 'AIST.RWC-MDB-P-2001.SMF_SYNC'), '.MID', io.MidiIO, 'midi')
    return rwc

def create_rwc_dataset(hop_length=DEFAULT_HOP_LENGTH, raw=False):
    rwc = DataPool('rwc')
    set_default_dataset_properties(rwc)
    rwc.set_property('hop_length', hop_length)
    rwc.append_folder(os.path.join(RWC_DATASET_PATH, 'AUDIO'), '.wav', io.MusicIO, 'music')
    file_list = os.listdir(os.path.join(RWC_DATASET_PATH, 'LAB'))
    file_list.sort()
    assert (len(file_list) == len(rwc.entries))
    for i in range(len(rwc.entries)):
        rwc.entries[i].append_file(os.path.join(RWC_DATASET_PATH, 'LAB', file_list[i]), ChordLabIO, 'chordlab')
    if (not raw):
        file_list = os.listdir(os.path.join(RWC_DATASET_PATH, 'BEATS'))
        file_list.sort()
        for i in range(len(rwc.entries)):
            rwc.entries[i].append_file(os.path.join(RWC_DATASET_PATH, 'BEATS', file_list[i]), DownbeatIO, 'beat')
    return rwc

def create_osu_map_dataset(map_list="data/map_list.txt", map_directory=OSU_DATA_PATH, hop_length=DEFAULT_HOP_LENGTH):
    f=open(map_list, encoding='UTF-8')
    lines=f.readlines()
    f.close()
    osu=DataPool('osu')
    set_default_dataset_properties(osu)
    osu.set_property('hop_length', hop_length)
    last_entry=None
    last_music=''
    map_count=0
    for line in lines:
        line=line.strip()
        if(line==''):
            continue
        tokens=line.split('\t')
        folder_name=tokens[0][:tokens[0].index('_')]
        if(last_entry is None or tokens[0]!=last_entry.name[4:]):
            if(last_entry is not None):
                last_entry.append_data(map_count,io.IntegerIO,'map_count')
            last_entry=osu.new_entry(tokens[0])
            last_entry.declare_proxy_array('map')
            last_entry.declare_proxy_array('star')
            map_count=0
            last_music=tokens[2]
            last_entry.append_file(os.path.join(map_directory,folder_name,tokens[2]),io.MusicIO,'music',file_exist_check=False)
        else:
            if(last_music!=tokens[2]):
                print('Warning: inconsistent music for %s'%line)
        last_entry.append_file(os.path.join(map_directory,folder_name,tokens[1]),OsuMapIO,'map[%d]'%map_count,file_exist_check=False)
        last_entry.append_data(float(tokens[3]),io.FloatIO,'star[%d]'%map_count)
        map_count+=1
    if(last_entry is not None):
        last_entry.append_data(map_count,io.IntegerIO,'map_count')
    return osu

def main():
    rwc = create_rwc_dataset()
    for entry in rwc.entries:
        entry.visualize(['beat'])

if __name__ == '__main__':
    main()

