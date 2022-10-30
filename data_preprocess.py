import numpy as np
import os
import pretty_midi
from joblib import Parallel, delayed
from settings import LMD_MATCHED_FOLDER
import matplotlib.pyplot as plt

def is_good_midi(file_name):
    try:
        midi = pretty_midi.PrettyMIDI(file_name)
    except:
        return False
    has_drum = False
    for ins in midi.instruments:
        if (ins.is_drum):
            has_drum = True
            break
    if (len(midi.instruments) < 6 or not has_drum):
        return False
    for i, ins in enumerate(midi.instruments):
        lower_name = ins.name.lower()
        if (lower_name == 'mel' or 'melody' in lower_name or 'vocal' in lower_name):
            return True
    return False


def process_folder(folder, files):
    for file in files:
        file_path = os.path.join(LMD_MATCHED_FOLDER, folder, file)
        if (is_good_midi(file_path)):
            return os.path.join(folder, file)
    return None


def preprocess_lmd_dataset():
    file_dict = set()
    folder_list = []
    for dir1 in os.listdir(LMD_MATCHED_FOLDER):
        if not (os.path.isdir(os.path.join(LMD_MATCHED_FOLDER, dir1))):
            continue
        for dir2 in os.listdir(os.path.join(LMD_MATCHED_FOLDER, dir1)):
            for dir3 in os.listdir(os.path.join(LMD_MATCHED_FOLDER, dir1, dir2)):
                for dir4 in os.listdir(os.path.join(LMD_MATCHED_FOLDER, dir1, dir2, dir3)):
                    files = []
                    for file in os.listdir(os.path.join(LMD_MATCHED_FOLDER, dir1, dir2, dir3, dir4)):
                        if not (file in file_dict):
                            files.append(file)
                            file_dict.add(file)
                    if (len(files) > 0):
                        folder_list.append([os.path.join(dir1, dir2, dir3, dir4), files])
    result = Parallel(n_jobs=24, verbose=1)(delayed(process_folder)(folder[0], folder[1]) for folder in folder_list)
    f = open('data/lmd_matched_usable_midi.txt', 'w')
    for file in result:
        if (file is not None):
            f.write(file + '\n')
    f.close()

def get_quantized_melody(ins, boundaries):
    piano_roll = np.zeros((len(boundaries), ), dtype=np.int16)
    if (ins.is_drum):
        return piano_roll
    for note in ins.notes:
        start_bin = np.searchsorted(boundaries, note.start)
        end_bin = np.searchsorted(boundaries, note.end)
        if (end_bin == start_bin):
            end_bin += 1
        piano_roll[start_bin:end_bin] = np.maximum(piano_roll[start_bin:end_bin], 1)
        piano_roll[start_bin:start_bin + 1] = note.pitch + 2
    return piano_roll

def get_piano_roll(ins, boundaries, onset_only, ignore_drums=True, ignore_non_drums=False):
    piano_roll = np.zeros((len(boundaries), 128), dtype=bool)
    if ((ins.is_drum and ignore_drums) or (not ins.is_drum and ignore_non_drums)):
        return piano_roll
    for note in ins.notes:
        start_bin = np.searchsorted(boundaries, note.start)
        if (onset_only):
            piano_roll[start_bin:start_bin + 1, note.pitch] = True
        else:
            end_bin = np.searchsorted(boundaries, note.end)
            if (end_bin == start_bin):
                end_bin += 1
            piano_roll[start_bin:end_bin, note.pitch] = True
    return piano_roll

def get_drum_roll(midi, boundaries):
    drum_roll = np.zeros((len(boundaries), 128), dtype=bool)
    for ins in midi.instruments:
        if (ins.is_drum):
            for note in ins.notes:
                start_bin = np.searchsorted(boundaries, note.start)
                drum_roll[start_bin:start_bin + 1, note.pitch] = True
    return drum_roll

def prepare_quantization(midi, subbeat_count=4):
    '''

    :param midi:
    :param subbeat_count:
    :return:
    '''
    beats = midi.get_beats()
    downbeats = midi.get_downbeats()

    n_beat = len(beats)
    n_subbeat = (n_beat - 1) * subbeat_count + 1
    subbeat_indices = np.arange(n_subbeat) / subbeat_count
    subbeat_time = np.interp(subbeat_indices, np.arange(n_beat), beats)
    boundaries = (subbeat_time[1:] + subbeat_time[:-1]) / 2
    is_downbeat = np.zeros(n_subbeat, dtype=bool)
    downbeat_bins = np.searchsorted(boundaries, downbeats)
    is_downbeat[downbeat_bins] = True
    return n_subbeat, downbeat_bins, boundaries, subbeat_time

def extract_features(seed, file_name, melody_track_id, subbeat_count=4, context_length=64, max_results=20, show=-1):
    midi = pretty_midi.PrettyMIDI(os.path.join(LMD_MATCHED_FOLDER, file_name))
    n_subbeat, downbeat_bins, boundaries, _ = prepare_quantization(midi, subbeat_count)
    piano_rolls = np.stack([get_piano_roll(ins, boundaries, False) for ins in midi.instruments], axis=0)
    melody = get_quantized_melody(midi.instruments[melody_track_id], boundaries)
    drum_rolls = get_drum_roll(midi, boundaries)
    cymbals = np.logical_or(drum_rolls[:, 49], drum_rolls[:, 57])
    ins_activity = piano_rolls.sum(axis=2)
    activity = np.concatenate([ins_activity.T, drum_rolls], axis=1)
    non_zero_activity = activity[:, activity.sum(axis=0) > 0] > 0
    results = [[], []]
    last_downbeat_bin = 0
    for i, downbeat_bin in enumerate(downbeat_bins):
        if (i > 0 and last_downbeat_bin + 4 < downbeat_bin):
            non_zero_activity[last_downbeat_bin:downbeat_bin] = np.max(non_zero_activity[last_downbeat_bin + 2:downbeat_bin - 2], axis=0, keepdims=True)
            last_downbeat_bin = downbeat_bin
    for i, downbeat_bin in enumerate(downbeat_bins):
        context_left = downbeat_bin - context_length
        context_right = downbeat_bin + context_length
        if (context_left >= 0 and context_right < n_subbeat):
            if (np.any(melody[context_left: downbeat_bin]) or
                    np.any(melody[downbeat_bin: context_right])):
                entry_points = np.sum(
                    np.all(non_zero_activity[context_left: downbeat_bin] == 0, axis=0) *
                    np.all(non_zero_activity[downbeat_bin: context_right] != 0, axis=0)
                )
                exit_points = np.sum(
                    np.all(non_zero_activity[context_left: downbeat_bin] != 0, axis=0) *
                    np.all(non_zero_activity[downbeat_bin: context_right] == 0, axis=0)
                )
                label = cymbals[downbeat_bin] > 0
                labels = [label, entry_points, exit_points]
                results[1 if np.any(labels) else 0].append(
                    np.concatenate((labels + [non_zero_activity.shape[1]], melody[context_left:context_right]))
                )
    np.random.seed(seed)
    final_results = []
    for k in [0, 1]:
        if (len(results[k]) > 0):
            results[k] = np.stack(results[k], axis=0)
            final_results.append(
                results[k][np.random.choice(np.arange(len(results[k])), min(max_results, len(results[k])), replace=False)])
        else:
            return None
    if (show >= 0):
        piano_roll = np.eye(130)[melody]
        length = min(show, n_subbeat - 1)
        plt.figure(figsize=(26, 6))
        plt.imshow(np.logjical_or(piano_roll[:length], cymbals[:length, None]).T)
        plt.title(os.path.basename(file_name))
        plt.gca().invert_yaxis()
        plt.show()
    return np.concatenate(final_results, axis=0)

def extract_all_features_split(lines, indices, out_file, show):
    results = []
    for t, i in enumerate(indices):
        if (t % 100 == 0):
            print('Processing %s %d / %d' % (out_file, t, len(indices)), flush=True)
        (file, melody_track_id) = lines[i]
        result = extract_features(i, file, int(melody_track_id), show=show)
        if (result is not None):
            results.append(result)
    print('Concatenating', flush=True)
    np.save(out_file, np.concatenate(results, axis=0))
    print('Done %s' % out_file, flush=True)

def extract_all_features(show=-1):
    f = open('data/lmd_matched_with_melody.txt', 'r')
    lines = [line.strip().split('\t') for line in f.readlines() if line.strip() != '']
    f.close()
    n_songs = len(lines)
    val_indices = np.arange(0, n_songs, 5)
    train_indices = np.setdiff1d(np.arange(0, n_songs), val_indices)
    extract_all_features_split(lines, train_indices, 'data/lmd_entry_exit_train_v3.npy', show)
    extract_all_features_split(lines, val_indices, 'data/lmd_entry_exit_val_v3.npy', show)

if __name__ == '__main__':
    preprocess_lmd_dataset()
    # extract_all_features(False)