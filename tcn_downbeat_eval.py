from tcn_downbeat_unsupervised import TCNClassifier, NetworkInterface, N_MIDI_PITCH, CONTEXT_LENGTH
import numpy as np
from midi_structure import get_piano_roll, prepare_quantization, evaluate_result, evaluations_to_latex, eval_beat_result
import pretty_midi
import os
from settings import LMD_MATCHED_FOLDER, RWC_DATASET_PATH
import matplotlib.pyplot as plt
import torch
from crf import CRFDecoder
from metrical_crf import get_ternary_transition
from scipy.special import logsumexp
from tcn_downbeat_supervised import TCNClassifierSupervised
from scipy.ndimage.filters import maximum_filter1d
import sys

offset_bin_count_downbeat = np.zeros(16)
offset_bin_count_beat = np.zeros(4)
MAX_OFFSET = 64

def decode(log_observations):
    log_observations = torch.tensor(log_observations)
    # log_transitions = get_log_transitions(4)
    log_transitions, indices = get_ternary_transition(np.array([-5.0, -4.0, -3.0, -2.0]), np.array([-8.0, -7.0, -6.0, -5.0]))
    log_observations = log_observations[:, indices]
    crf = CRFDecoder(torch.tensor(log_transitions))
    result = crf.viterbi_decode(log_observations[None]).squeeze(0).numpy()
    return indices[result]

def get_rolls(midi, subbeat_count=4, drums=1, melody=1, others=1, tracks=None):
    n_subbeat, downbeat_bins, boundaries, subbeat_time = prepare_quantization(midi, subbeat_count)
    piano_rolls = [get_piano_roll(ins, boundaries, False, ignore_drums=True) for ins in midi.instruments]
    onset_rolls = [get_piano_roll(ins, boundaries, True, ignore_drums=True) for ins in midi.instruments]
    drum_rolls = [get_piano_roll(ins, boundaries, True, ignore_drums=False, ignore_non_drums=True) for ins in midi.instruments]
    rolls = []
    ins_names = []
    # collect all drum tracks first
    for j, ins in enumerate(midi.instruments):
        if (ins.is_drum):
            if (drums == 0 or (tracks is not None and j not in tracks)):
                continue
            roll = np.concatenate((onset_rolls[j], piano_rolls[j], drum_rolls[j]), axis=-1)
            rolls.append(roll)
            ins_names.append('drums:%d' % j)
    if (len(rolls) > 1):
        rolls = [np.max(rolls, axis=0)]
        ins_names = ['drums:-1']
    for j, ins in enumerate(midi.instruments):
        if (ins.is_drum):
            continue
        if ('mel' in ins.name.lower() or 'vocal' in ins.name.lower()):
            if (melody == 0 or (tracks is not None and j not in tracks)):
                continue
            ins_name = 'melody'
        else:
            ins_name = pretty_midi.program_to_instrument_name(ins.program) + '(%d)' % ins.program
            if (others == 0 or (tracks is not None and j not in tracks)):
                continue
        roll = np.concatenate((onset_rolls[j], piano_rolls[j], drum_rolls[j]), axis=-1)
        rolls.append(roll)
        ins_names.append('%s:%d' % (ins_name, j))
        # visualized_preds, _ = model.inference(roll.astype(np.float32))
        # plt.figure(figsize=(26, 6))
        # plt.imshow(np.concatenate((piano_rolls[j][:, ::-1], np.repeat(visualized_preds, 16, axis=1)), axis=1).T, interpolation='nearest')
        # plt.title(os.path.basename(midi_path))
        # plt.show()
    if (len(rolls) == 0):
        print('No track!')
        return None
    # print('Tracks: %d' % (len(rolls)))
    rolls = np.stack(rolls, axis=0)
    return rolls, (n_subbeat, downbeat_bins, boundaries, subbeat_time)

def model_eval_downbeat(model, midi_path, subbeat_count=4, drums=1, melody=1, others=1, tracks=None):
    try:
        midi = pretty_midi.PrettyMIDI(midi_path)
    except:
        print('Midi load failed: %s' % midi_path)
        return None
    # print('Evaluating:', midi_path)
    rolls, (n_subbeat, downbeat_bins, boundaries, subbeat_time) = get_rolls(midi, subbeat_count=subbeat_count, drums=drums, melody=melody, others=others, tracks=tracks)
    original_length = rolls.shape[1]
    log_pred, _ = model.inference_function('inference_song', np.pad(rolls.astype(np.float32), ((0, 0), (MAX_OFFSET, MAX_OFFSET), (0, 0))))
    log_pred = log_pred[MAX_OFFSET - model.net.remap_offset:]
    log_downbeat_pred = logsumexp(log_pred[:, 4:] if log_pred.shape[1] >= 4 else log_pred[:, 2:], axis=1)
    max_values = maximum_filter1d(log_downbeat_pred[::subbeat_count], size=5, axis=0)
    downbeat_pred = max_values == log_downbeat_pred[::subbeat_count]
    downbeat_gt = np.zeros(original_length // subbeat_count)
    downbeat_gt[downbeat_bins[downbeat_bins < original_length] // subbeat_count] = 1
    return eval_beat_result(downbeat_pred, downbeat_gt)

def model_eval(model, midi_path, subbeat_count=4, drums=1, melody=1, others=1, visualize=True, tracks=None, raw_output=False):
    try:
        midi = pretty_midi.PrettyMIDI(midi_path)
    except:
        print('Midi load failed: %s' % midi_path)
        return None
    # print('Evaluating:', midi_path)
    rolls, (n_subbeat, downbeat_bins, boundaries, subbeat_time) = get_rolls(midi, subbeat_count=subbeat_count, drums=drums, melody=melody, others=others, tracks=tracks)
    original_length = rolls.shape[1]
    log_pred, log_conf = model.inference_function('inference_song', np.pad(rolls.astype(np.float32), ((0, 0), (MAX_OFFSET, MAX_OFFSET), (0, 0))))
    log_pred = log_pred[MAX_OFFSET - model.net.remap_offset:MAX_OFFSET - model.net.remap_offset + original_length]
    if (visualize):
        pred_viz = np.log(log_pred)
        pred_viz = np.cumsum(pred_viz[:, ::-1], axis=-1)[:, ::-1]
        visualized_preds = pred_viz[:, ::-1]
        plt.figure(figsize=(26, 6))
        plt.imshow(np.concatenate((rolls.max(axis=0)[:, ::-1], np.repeat(visualized_preds, 16, axis=1), ), axis=1).T)
        plt.title(os.path.basename(midi_path) + ' final')
        plt.show()
    if raw_output:
        pred = np.exp(log_pred)
        if (pred.shape[1] < 9):
            pred = np.pad(pred, ((0, 0), (0, 9 - pred.shape[1])))
        outputs = [pretty_midi.Instrument(program=0, is_drum=True, name='Layers%d' % i) for i in range(8)]
        pred = np.cumsum(pred[:, ::-1], axis=-1)[:, ::-1]
        current_downbeat_offset = -1
        for i in range(len(pred)):
            offset_bin_count_beat[i % 4] += pred[i, 2]
            if (pred[i, 4] > 0.5):
                current_downbeat_offset = 0
            elif (current_downbeat_offset >= 0):
                current_downbeat_offset += 1
            if (0 <= current_downbeat_offset < 16):
                offset_bin_count_downbeat[current_downbeat_offset] += pred[i, 4]
            if (0 <= i < len(subbeat_time) - 1):
                for j in range(8):
                    outputs[j].notes.append(pretty_midi.Note(velocity=int(pred[i, j + 1] * 127), pitch=[42, 44, 46, 37, 36, 49, 49, 49][j], start=subbeat_time[i], end=subbeat_time[i + 1]))
        for output in outputs:
            midi.instruments.append(output)
        evaluation = None
    else:
        # metrical structure eval
        log_final_pred = np.zeros((log_pred.shape[0], 5))
        log_final_pred[:, 1:] = log_pred[:, -4:]
        log_final_pred[:, 0] = logsumexp(log_pred[:, :-4], axis=1)
        # log_final_pred = np.log(uniform_filter1d(np.exp(log_final_pred), size=5, axis=0))
        used_downbeats = downbeat_bins[downbeat_bins < len(log_final_pred)]
        log_downbeat_pred = log_final_pred[used_downbeats]
        result = decode(log_downbeat_pred)
        gt_midi_path = 'annotation/%s_gt.mid' % os.path.basename(midi_path)
        if (os.path.exists(gt_midi_path)):
            evaluation = evaluate_result(result, gt_midi_path, downbeat_bins, subbeat_count, 4)
            # print(('%s:\t' % gt_midi_path) + '\t'.join(str(x) for x in evaluation))
        else:
            evaluation = None
        output = pretty_midi.Instrument(program=0, is_drum=True, name='Layers')
        for i, pred in enumerate(result):
            for k in range(pred):
                onset_time = subbeat_time[downbeat_bins[i]]
                output.notes.append(pretty_midi.Note(velocity=100, pitch=40 + k, start=onset_time, end=onset_time + 0.5))

        midi.instruments.append(output)
    if not (os.path.exists('output/%s' % model.save_name)):
        os.mkdir('output/%s' % model.save_name)
    midi.write('output/%s/%s_drums%d_mel%d_others%d_%s.mid' % (model.save_name, os.path.basename(midi_path), drums, melody, others, 'raw' if raw_output else 'crf'))
    return evaluation
def get_split(data_file, split):

    f = open('./data/%s.split.txt.names' % data_file, 'r')
    tokens = [line.strip().split(',') for line in f.readlines() if line.strip() != '']
    f.close()
    if (split == 'train'):
        return tokens[0]
    elif (split == 'val'):
        return tokens[1]
    elif (split == 'test'):
        return tokens[2]
    else:
        raise Exception('No such split')

def evaluate_lmd(model, count, drums=1, melody=1, others=1, visualize=False):
    f = open('data/lmd_matched_usable_midi.txt', 'r')
    lines = [line.strip() for line in f.readlines() if line.strip() != '']
    f.close()
    np.random.seed(6172)
    np.random.shuffle(lines)
    lines = lines[:count]
    for line in lines:
        model_eval(model, os.path.join(LMD_MATCHED_FOLDER, line), drums=drums, melody=melody, others=others, visualize=visualize)



def main():
    # model = NetworkInterface(TCNClassifier(384, 256, 7, 9, 0.1, remap_offset=12),
    #     'tcn_downbeat_unsupervised_v2.25_fix_1024_context_7_tcn.cp', load_checkpoint=False)
    # model = NetworkInterface(TCNClassifier(384, 256, 6, 9, 0.1, remap_offset=10),
    #     'tcn_downbeat_unsupervised_v2.25_fix_1024_context_6_tcn', load_checkpoint=False)
    # model = NetworkInterface(TCNClassifier(384, 256, 6, 9, 0.1, remap_offset=1),
    #     'tcn_downbeat_unsupervised_v2.25_fix_1024_context_6_tcn_dynamic_resolution.cp', load_checkpoint=False)
    # model = NetworkInterface(TCNClassifier(384, 256, 6, 9, 0.1, remap_offset=12),
    #     'tcn_downbeat_unsupervised_v2.3_fix_1024_context_6_tcn_crf.cp', load_checkpoint=False)
    for use_consistency_loss in [True, False]:
        model = NetworkInterface(TCNClassifier(384, 256, 6, 9, 0.1, remap_offset=11 if use_consistency_loss else 21),
            'tcn_downbeat_unsupervised_v3.0_1024_context_6_tcn' + ('' if use_consistency_loss else '_noc'), load_checkpoint=False)
        # evaluate_lmd(model, 9999)
        # exit(0)
        # model_eval(model, R'E:\Dataset\lmd_matched\L\C\N\TRLCNWM128F423BB63\7596e59dea60afab6bbc7207aca8bd8c.mid')
        model_name = 'Unsupervised' if use_consistency_loss else 'W/o consistency'
        split_files = get_split('rwc_multitrack_hierarchy_v6_supervised', 'test')
        print(evaluations_to_latex(f'{model_name}\n (mel. only)',
            [model_eval(model, os.path.join(RWC_DATASET_PATH, 'AIST.RWC-MDB-P-2001.SMF_SYNC', file),
                        drums=0, melody=1, others=0, visualize=False) for file in split_files]))
        print(evaluations_to_latex(model_name,
            [model_eval(model, os.path.join(RWC_DATASET_PATH, 'AIST.RWC-MDB-P-2001.SMF_SYNC', file),
                        drums=1, melody=1, others=1, visualize=False) for file in split_files]))
        print(evaluations_to_latex(f'{model_name}\n(mel. only)',
            [model_eval(model, R'input/POP909-%d.mid' % (i + 1), tracks=[0]) for i in range(5)]))
        print(evaluations_to_latex(model_name,
            [model_eval(model, R'input/POP909-%d.mid' % (i + 1)) for i in range(5)]))

def main_downbeat():
    models = {'Supervised': NetworkInterface(TCNClassifierSupervised(384, 256, 6, 5, 0.1),
        'tcn_downbeat_supervised_v1.1', load_checkpoint=False),
              'Unsupervised': NetworkInterface(TCNClassifier(384, 256, 6, 9, 0.1, remap_offset=11),
        'tcn_downbeat_unsupervised_v3.0_1024_context_6_tcn', load_checkpoint=False),
              'W/o consistency': NetworkInterface(TCNClassifier(384, 256, 6, 9, 0.1, remap_offset=21),
        'tcn_downbeat_unsupervised_v3.0_1024_context_6_tcn_noc', load_checkpoint=False)}
    for model_name, model in models.items():
        for mel_only in [False, True]:
            results = []
            f = open('data/rwc_downbeat_eval_indices.txt', 'r')
            lines = [line.strip() for line in f.readlines() if line.strip() != '']
            f.close()
            for line in lines:
                result = model_eval_downbeat(model, os.path.join(RWC_DATASET_PATH, 'AIST.RWC-MDB-P-2001.SMF_SYNC', line),
                                                   drums=0 if mel_only else 1, melody=1, others=0 if mel_only else 1)
                results.append(result)
            print(model_name, mel_only, np.mean(results), np.std(results))


if __name__ == '__main__':
    model = NetworkInterface(TCNClassifier(384, 256, 6, 9, 0.1, remap_offset=11),
                             'tcn_downbeat_unsupervised_v3.0_1024_context_6_tcn', load_checkpoint=False)
    assert(model.finalized)
    if (len(sys.argv) != 2):
        print('Usage:', 'tcn_downbeat_eval.py', 'midi_path.midi')
        exit(0)
    midi_path = sys.argv[1]
    model_eval(model, midi_path, visualize=True, raw_output=True)
    model_eval(model, midi_path, visualize=True)

