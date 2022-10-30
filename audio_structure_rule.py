from settings import LMD_MATCHED_FOLDER, RWC_DATASET_PATH
import os
import pretty_midi
from data_preprocess import prepare_quantization, get_piano_roll, get_quantized_melody
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics
from crf import CRFDecoder
from metrical_crf import get_ternary_transition
import torch
from midi_structure import hierarchical_viterbi_decode
import datasets
from io_new.downbeat_io import DownbeatIO
from extractors.audio_helpers import BeatSyncEnergySpec, BeatSyncAudio
from mir import io
import sklearn
from midi_structure import evaluations_to_latex, get_split

BEAT_DIV = 4
SUBBEAT_DIV = 8

def get_split_entries(split_files):
    results = []
    rwc = datasets.create_rwc_dataset()
    for entry in rwc.entries:
        id = int(entry.name[4:])
        if ('RM-P%03d.SMF_SYNC.MID' % (id + 1) in split_files):
            results.append(entry)
    return results

def evaluate_result(predicted_result, gt_result, n_layers):
    if (len(predicted_result) >= len(gt_result)):
        predicted_result = predicted_result[:len(gt_result)]
    else:
        predicted_result = np.pad(predicted_result, (0, len(gt_result) - len(predicted_result)))
    return [sklearn.metrics.f1_score(gt_result >= i, predicted_result >= i) for i in range(1, n_layers + 1)]

def get_rwc_annotation_gt(id, audio_beats, offset_mismatch_threshold=0.2):
    audio_downbeats = np.array([b[0] for b in audio_beats if b[1] == 1])
    file_name = 'RM-P%03d.SMF_SYNC.MID' % (id + 1)
    gt_file_path = 'annotation/%s_gt.mid' % os.path.basename(file_name)
    file_path = os.path.join(RWC_DATASET_PATH, 'AIST.RWC-MDB-P-2001.SMF_SYNC', os.path.basename(file_name))
    # assert (save_type == 'numpy')
    midi = pretty_midi.PrettyMIDI(file_path)
    midi_gt = pretty_midi.PrettyMIDI(gt_file_path)
    n_subbeat, downbeat_bins, boundaries, subbeat_time = prepare_quantization(midi, BEAT_DIV)
    n_subbeat_gt, downbeat_bins_gt, boundaries_gt, subbeat_time_gt = prepare_quantization(midi_gt, BEAT_DIV)
    notes = [[np.searchsorted(boundaries_gt, note.start), note.pitch] for note in midi_gt.instruments[0].notes]
    note_dict = np.zeros(n_subbeat_gt, dtype=np.int16)
    for note in notes:
        if (note[1] >= 40):
            note_dict[note[0]] = max(note_dict[note[0]], note[1] - 39)
    # result = []
    audio_downbeat_tags = np.zeros_like(audio_downbeats, dtype=np.int)
    audio_boundary = (audio_downbeats[1:] + audio_downbeats[:-1]) / 2
    max_id = 0
    for i in range(min(len(subbeat_time), len(downbeat_bins_gt))):
        # 1.0 is the global offset between audio and midi
        time = subbeat_time[downbeat_bins_gt[i]] - 1.0
        # result.append([time, note_dict[downbeat_bins_gt[i]]])
        target_id = np.searchsorted(audio_boundary, time)
        if (np.abs(audio_downbeats[target_id] - time) < offset_mismatch_threshold):
            max_id = max(max_id, target_id)
            audio_downbeat_tags[target_id] = note_dict[downbeat_bins_gt[i]]
    return audio_downbeat_tags[:max_id]

def joint_analyze(entry, subbeat_count=4, visualize=False):
    beat = np.array(entry.beat)
    audio_length=len(entry.music) / entry.prop.sr
    subbeat = np.interp(np.arange(len(beat) * BEAT_DIV * SUBBEAT_DIV) / (BEAT_DIV * SUBBEAT_DIV), np.arange(len(beat)), beat[:, 0])
    time_steps = subbeat / audio_length
    hop_length = entry.prop.hop_length
    spec = entry.apply_extractor(BeatSyncEnergySpec, beat_div=BEAT_DIV, subbeat_div=SUBBEAT_DIV)
    spec = spec.reshape((spec.shape[0] // SUBBEAT_DIV, -1)).astype(np.float64)
    downbeat_bins = np.where(beat[:, 1] == 1)[0] * BEAT_DIV
    print(downbeat_bins)
    context_length = subbeat_count * 4
    novelty_scores = []
    n_layers = 4
    for k in range(n_layers):
        way_mean_ssm = []
        for way in [0, 1]:
            ssms = []
            features = []
            for i, downbeat_bin in enumerate(downbeat_bins):
                feature = spec[downbeat_bin:downbeat_bin + context_length] if way == 1 else \
                    spec[downbeat_bin - context_length:downbeat_bin]
                if (feature.shape[0] < context_length):
                    pad = (0, context_length - feature.shape[0]) if way == 1 else \
                        (context_length - feature.shape[0], 0)
                    feature = np.pad(feature, (pad, (0, 0)))
                feature = feature.reshape(-1)
                if (np.abs(feature.sum()) > 1e-6):
                    feature = feature / np.linalg.norm(feature)
                features.append(feature)
            features = np.stack(features, axis=0)
            ssm = np.matmul(features, features.T)
            mean_ssm = (ssm - ssm.mean()) / np.std(ssm)
            way_mean_ssm.append(mean_ssm)
        novelty_score = np.abs(way_mean_ssm[1] - way_mean_ssm[0]).mean(axis=0)
        novelty_scores.append(novelty_score)
        context_length *= 2
    novelty_scores = np.stack(novelty_scores, axis=1)
    result = hierarchical_viterbi_decode(novelty_scores)
    downbeat_tags = get_rwc_annotation_gt(int(entry.name[4:]), entry.beat)
    if (visualize):
        tag = [[downbeat_bins[i] * hop_length * SUBBEAT_DIV / entry.prop.sr, level] for i, level in enumerate(result)]
        gt_tag = [[downbeat_bins[i] * hop_length * SUBBEAT_DIV / entry.prop.sr, x] for i, x in enumerate(downbeat_tags)]
        entry.append_data(tag, DownbeatIO, 'tag')
        entry.append_data(gt_tag, DownbeatIO, 'gt_tag')
        entry.append_extractor(BeatSyncAudio, 'music_stretch', beat_div=BEAT_DIV, subbeat_div=SUBBEAT_DIV)
        entry.visualize(['tag', 'gt_tag'],music='music_stretch')
    evaluation = evaluate_result(result, downbeat_tags, 4)
    return evaluation

def main():
    split_files = get_split('rwc_multitrack_hierarchy_v6_supervised', 'test')
    print(evaluations_to_latex('Rule',
        [joint_analyze(entry, visualize=False) for entry in get_split_entries(split_files)]))


if __name__ == '__main__':
    main()
    exit(0)
    rwc = datasets.create_rwc_dataset()
    for entry in rwc.entries:
        print(joint_analyze(entry, visualize=True))
