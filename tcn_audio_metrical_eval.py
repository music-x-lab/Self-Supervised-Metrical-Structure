from tcn_audio_metrical_structure import TCNClassifier, NetworkInterface
import numpy as np
import pretty_midi
import os
from settings import LMD_MATCHED_FOLDER
import matplotlib.pyplot as plt
import torch
from crf import CRFDecoder
from metrical_crf import get_ternary_transition
import datasets
import librosa
from extractors.osu_map_analyzer import OsuHitObjects, OsuTimingPointsToSubBeat
from io_new.downbeat_io import DownbeatIO
from mir import io, DataEntry
from extractors.audio_helpers import BeatSyncAudio, BeatSyncEnergySpec
from tcn_audio_metrical_supervised import TCNClassifierSupervised
from midi_structure import prepare_quantization, get_split, evaluations_to_latex, eval_beat_result
from settings import RWC_DATASET_PATH
from scipy.special import logsumexp
from mir.extractors import ExtractorBase
from scipy.ndimage.filters import maximum_filter1d
from audio_structure_rule import evaluate_result, get_rwc_annotation_gt, get_split_entries
import sys
from settings import DEFAULT_HOP_LENGTH, DEFAULT_SR, DEFAULT_WIN_SIZE
offset_bin_count_downbeat = np.zeros(16)
offset_bin_count_beat = np.zeros(4)

SR = 22050
BEAT_DIV = 4
SUBBEAT_DIV = 8
MAX_OFFSET = 64


def decode(log_observations):
    log_observations = torch.tensor(log_observations)
    # log_transitions = get_log_transitions(4)
    log_transitions, indices = get_ternary_transition(np.array([-5.0, -4.0, -3.0, -2.0]), np.array([-8.0, -7.0, -6.0, -5.0]))
    log_observations = log_observations[:, indices]
    crf = CRFDecoder(torch.tensor(log_transitions))
    result = crf.viterbi_decode(log_observations[None]).squeeze(0).numpy()
    return indices[result]

def evaluate_subbeat_label(subbeat_label, pred):
    labels = subbeat_label[32:-32].astype(int)
    # labels = np.zeros(len(subbeat_label) * 4, dtype=np.int)
    # labels[::4] = subbeat_label
    n_frames = len(labels)
    likelihoods = []
    for i in range(64):
        prob = np.zeros((n_frames, 3))
        prob[:, 0] = pred[i:i + n_frames, 0:2].sum(axis=-1)
        prob[:, 1] = pred[i:i + n_frames, 2:4].sum(axis=-1)
        prob[:, 2] = pred[i:i + n_frames, 4:].sum(axis=-1)
        likelihoods.append(np.sum(np.log(prob[np.arange(n_frames), labels])))
    return np.array(likelihoods), np.argmax(likelihoods) - 32

def is_4_4_song(entry, map_id):
    tps = entry.map[map_id].timingpoints
    for tp in tps:
        tokens = tp.split(',')
        time_signature=int(tokens[2]) if len(tokens)>2 else 4
        if (time_signature != 4):
            return False
    return True

def eval_entry_downbeats(model, entry, visualize=False):
    print(entry.name)
    hop_length = entry.prop.hop_length
    spec = entry.apply_extractor(BeatSyncEnergySpec, beat_div=BEAT_DIV, subbeat_div=SUBBEAT_DIV)
    spec = spec.reshape((spec.shape[0] // SUBBEAT_DIV, -1))
    original_length = len(spec)
    spec = np.pad(spec, ((MAX_OFFSET, MAX_OFFSET), (0, 0)))
    log_pred = model.inference(spec)[MAX_OFFSET - model.net.remap_offset:]
    log_downbeat_pred = logsumexp(log_pred[:, 4:] if log_pred.shape[1] >= 4 else log_pred[:, 2:], axis=1)
    # compensate for some beat shift errors
    log_downbeat_pred = maximum_filter1d(log_downbeat_pred, size=3, axis=0)
    max_values = maximum_filter1d(log_downbeat_pred[::BEAT_DIV], size=5, axis=0)
    downbeat_pred = max_values == log_downbeat_pred[::BEAT_DIV]
    downbeat_bins = np.array([i for i, beat in enumerate(entry.beat) if beat[1] == 1])
    downbeat_gt = np.zeros(original_length // BEAT_DIV)
    downbeat_gt[downbeat_bins] = 1
    if (visualize):
        entry.append_extractor(BeatSyncAudio, 'music_stretch', beat_div=BEAT_DIV, subbeat_div=SUBBEAT_DIV)
        downbeat_tag = [[i * hop_length * SUBBEAT_DIV * BEAT_DIV / entry.prop.sr, 2] for i, x in enumerate(downbeat_pred) if x > 0]
        gt_downbeat_tag = [[downbeat_bins[i] * hop_length * SUBBEAT_DIV * BEAT_DIV / entry.prop.sr, x] for i, x in enumerate(downbeat_bins)]
        entry.append_data(downbeat_tag, DownbeatIO, 'downbeat_tag')
        entry.append_data(gt_downbeat_tag, DownbeatIO, 'gt_downbeat_tag')
        entry.visualize(['downbeat_tag', 'gt_downbeat_tag'], music='music_stretch')

    return eval_beat_result(downbeat_pred, downbeat_gt)


def eval_entry(model, entry, visualize=False):
    print(entry.name)
    hop_length = entry.prop.hop_length
    spec = entry.apply_extractor(BeatSyncEnergySpec, beat_div=BEAT_DIV, subbeat_div=SUBBEAT_DIV)
    spec = spec.reshape((spec.shape[0] // SUBBEAT_DIV, -1))
    original_length = len(spec)
    spec = np.pad(spec, ((MAX_OFFSET, MAX_OFFSET), (0, 0)))
    log_pred = model.inference(spec)
    log_final_pred = np.zeros((log_pred.shape[0], 5))
    log_final_pred[:, 1:] = log_pred[:, -4:]
    log_final_pred[:, 0] = logsumexp(log_pred[:, :-4], axis=1)
    downbeat_bins = np.array([i for i, beat in enumerate(entry.beat) if beat[1] == 1]) * BEAT_DIV
    filtered_pred = maximum_filter1d(log_final_pred, size=5, axis=0)
    log_downbeat_pred = filtered_pred[downbeat_bins + MAX_OFFSET - model.net.remap_offset]
    result = decode(log_downbeat_pred)
    if (entry.name != ''):
        downbeat_tags = get_rwc_annotation_gt(int(entry.name[4:]), entry.beat)
    else:
        downbeat_tags = None
    if (visualize):
        visualize_pred = np.cumsum(np.exp(log_pred)[:, ::-1], axis=-1)[:, ::-1]
        # pred = np.roll(pred[:, 1:], model.net.remap_offset, axis=0)
        visualize_pred = visualize_pred[MAX_OFFSET - model.net.remap_offset:MAX_OFFSET - model.net.remap_offset + original_length, 1:]
        entry.append_extractor(BeatSyncAudio, 'music_stretch', beat_div=BEAT_DIV, subbeat_div=SUBBEAT_DIV)
        # alignment issue (visualize only)
        entry.append_data(np.repeat(visualize_pred, SUBBEAT_DIV, axis=0)[int(SUBBEAT_DIV * 2.5):], io.SpectrogramIO, 'pred')

        tag = [[downbeat_bins[i] * hop_length * SUBBEAT_DIV / entry.prop.sr, x] for i, x in enumerate(result)]
        entry.append_data(tag, DownbeatIO, 'tag')
        entry.visualize(['pred', 'tag'], music='music_stretch')
    if (entry.name != ''):
        evaluation = evaluate_result(result, downbeat_tags, 4)
        return evaluation
    else:
        return None

def main():
    split_files = get_split('rwc_multitrack_hierarchy_v6_supervised', 'test')
    model = NetworkInterface(TCNClassifier(513, 256, 6, 9, 0.5, remap_offset=20),
        'tcn_audio_metrical_v1.0_crf.cp.back3', load_checkpoint=False)
    print(evaluations_to_latex('Unsupervised',
        [eval_entry(model, entry, visualize=False) for entry in get_split_entries(split_files)]))

def main_downbeat():
    models = {'Supervised': NetworkInterface(TCNClassifier(513, 256, 6, 9, 0.5, remap_offset=20),
        'tcn_audio_metrical_v1.0_crf.cp.back3', load_checkpoint=False),
              'Unsupervised': NetworkInterface(TCNClassifierSupervised(513, 256, 6, 3, 0.5),
        'tcn_audio_metrical_supervised_v1.0_crf.cp', load_checkpoint=False)}

    # eval_osu(model, visualize=True)
    rwc = datasets.create_rwc_dataset()
    for model_name, model in models.items():
        results = []
        for entry in rwc.entries:
            results.append(eval_entry_downbeats(model, entry, visualize=False))
        print(model_name, np.mean(results), np.std(results))

if __name__ == '__main__':
    model = NetworkInterface(TCNClassifier(513, 256, 6, 9, 0.5, remap_offset=20),
        'tcn_audio_metrical_v1.0_crf.cp', load_checkpoint=False)
    assert(model.finalized)
    # eval_osu(model, visualize=True)
    if (len(sys.argv) != 3):
        print('Usage:', 'tcn_audio_metrical_eval.py', 'audio_path', 'beat_annotation.lab')
        exit(0)
    midi_path = sys.argv[1]
    entry = DataEntry()
    entry.prop.set('hop_length', DEFAULT_HOP_LENGTH)
    entry.prop.set('sr', DEFAULT_SR)
    entry.prop.set('win_size', DEFAULT_WIN_SIZE)
    entry.append_file(sys.argv[1], io.MusicIO, 'music')
    entry.append_file(sys.argv[2], DownbeatIO, 'beat')
    eval_entry(model, entry, visualize=True)
