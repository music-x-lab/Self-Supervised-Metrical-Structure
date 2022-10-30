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


def hierarchical_viterbi_decode(novelty_score):
    log_observations = torch.cumsum(torch.log(torch.tensor(novelty_score)), dim=1)
    log_observations = torch.cat([torch.zeros(log_observations.shape[0], 1), log_observations], dim=1) * 4
    # log_transitions = get_log_transitions(4)
    log_transitions, indices = get_ternary_transition(np.array([-2.6, -2.0, -1.4, -0.8]), np.array([-5.0, -4.2, -3.4, -2.6]))
    log_observations = log_observations[:, indices]
    crf = CRFDecoder(torch.tensor(log_transitions))
    result = crf.viterbi_decode(log_observations[None]).squeeze(0).numpy()
    return indices[result]

def joint_analyze(file_name, subbeat_count=4, save_type='midi', load_type='none', data_aug=False, drums=1, melody=1, others=1, tracks=None):
    try:
        midi = pretty_midi.PrettyMIDI(file_name)
    except:
        print('Midi load failed: %s' % file_name)
        return None
    n_subbeat, downbeat_bins, boundaries, subbeat_time = prepare_quantization(midi, subbeat_count)
    output = pretty_midi.Instrument(program=0, is_drum=True, name='Layers')
    piano_rolls = [get_piano_roll(ins, boundaries, False, ignore_drums=False) for ins in midi.instruments]
    context_length = subbeat_count * 4
    novelty_scores = []
    n_layers = 4
    for k in range(n_layers):
        way_mean_ssm = []
        for way in [0, 1]:
            ssms = []
            for j, ins in enumerate(midi.instruments):
                if (ins.is_drum):
                    if (drums == 0 or (tracks is not None and j not in tracks)):
                        continue
                elif ('mel' in ins.name.lower() or 'vocal' in ins.name.lower()):
                    if (melody == 0 or (tracks is not None and j not in tracks)):
                        continue
                else:
                    if (others == 0 or (tracks is not None and j not in tracks)):
                        continue
                features = []
                for i, downbeat_bin in enumerate(downbeat_bins):
                    feature = piano_rolls[j][downbeat_bin:downbeat_bin + context_length] if way == 1 else \
                        piano_rolls[j][downbeat_bin - context_length:downbeat_bin]
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
                if (ssm.any()):
                    ssms.append(ssm)
            mean_ssm = np.mean(np.stack(ssms, axis=0), axis=0)
            mean_ssm = (mean_ssm - mean_ssm.mean()) / np.std(mean_ssm)
            way_mean_ssm.append(mean_ssm)
        novelty_score = np.abs(way_mean_ssm[1] - way_mean_ssm[0]).mean(axis=0)
        novelty_scores.append(novelty_score)
        context_length *= 2
        # plt.plot(np.arange(len(novelty_score)), novelty_score)
        # plt.show()
    novelty_scores = np.stack(novelty_scores, axis=1)
    result = hierarchical_viterbi_decode(novelty_scores)
    gt_file_path = 'annotation/%s_gt.mid' % os.path.basename(file_name)
    if (load_type == 'midi'):
        # assert (save_type == 'numpy')
        midi_gt = pretty_midi.PrettyMIDI(gt_file_path)
        n_subbeat_gt, downbeat_bins_gt, boundaries_gt, subbeat_time_gt = prepare_quantization(midi_gt, subbeat_count)
        notes = [[np.searchsorted(boundaries_gt, note.start), note.pitch] for note in midi_gt.instruments[0].notes]
        note_dict = np.zeros(max(n_subbeat, n_subbeat_gt), dtype=np.int16)
        print('before:', result)
        for note in notes:
            if (note[1] >= 40):
                note_dict[note[0]] = max(note_dict[note[0]], note[1] - 39)
        for i in range(len(result)):
            result[i] = note_dict[downbeat_bins[i]]
        print('after:', result)


    if (save_type == 'midi'):
        for i, pred in enumerate(result):
            for k in range(pred):
                onset_time = subbeat_time[downbeat_bins[i]]
                output.notes.append(pretty_midi.Note(velocity=100, pitch=40 + k, start=onset_time, end=onset_time + 0.5))

        midi.instruments.append(output)
        midi.write('output/%s_hierarchical_dp.mid' % os.path.basename(file_name))
    elif (save_type == 'f1'):
        if (os.path.exists(gt_file_path)):
            eval_result = evaluate_result(result, gt_file_path, downbeat_bins, subbeat_count, n_layers)
            print(('%s:\t' % file_name) + '\t'.join(
                str(x) for x in eval_result))
            return eval_result
    elif (save_type == 'numpy'):
        output_context_length = 64
        context_length = output_context_length
        results = []
        onset_rolls = [get_piano_roll(ins, boundaries, True, ignore_drums=False) for ins in midi.instruments]
        beat_bins = np.searchsorted(boundaries, midi.get_beats())
        k_min, k_max = 0, 0
        if (data_aug):
            k_min, k_max = -1, 1
        for k in range(k_min, k_max + 1):
            for beat_bin in beat_bins:
                context_left = beat_bin - context_length
                context_right = beat_bin + context_length
                js = np.argwhere(downbeat_bins == beat_bin)
                if (context_left >= 0 and context_right < n_subbeat):
                    for i, ins in enumerate(midi.instruments):
                        if (np.any(piano_rolls[i][context_left: context_right])):
                            if (ins.is_drum):
                                roll = onset_rolls[i][context_left: context_right] * 2
                            else:
                                roll = piano_rolls[i][context_left: context_right] + onset_rolls[i][context_left: context_right]
                            target = result[js[0]] + 1 if len(js) > 0 else 0.0
                            target = np.clip(target - k, 0, n_layers + 1)
                            roll = roll.reshape(output_context_length * 2, -1, roll.shape[-1]).max(axis=1)
                            results.append(np.concatenate([
                                np.array([target, ins.is_drum, ins.program], dtype=np.uint8),
                                roll.reshape(-1).astype(np.uint8)
                            ]))
            context_length *= 2
        results = np.stack(results, axis=0)
        return results
    else:
        raise NotImplementedError()

def generate_labels(start_id, end_id, name, load_type='none'):
    train_data = []
    test_data = []
    for i in range(start_id, end_id):
        result = joint_analyze(os.path.join(LMD_MATCHED_FOLDER,
                                            R'E:\Dataset\RWC\AIST.RWC-MDB-P-2001.SMF_SYNC\RM-P%03d.SMF_SYNC.MID' % (
                                                        i + 1)),
                               save_type='numpy', load_type=load_type, data_aug=True, subbeat_count=8)
        if (result is not None):
            if (i % 10 == 0):
                test_data.append(result)
            else:
                train_data.append(result)
    np.savez_compressed('data/%s.npz' % name, np.concatenate(train_data, axis=0))

def eval_beat_result(predicted_result, gt_result):
    if (len(predicted_result) >= len(gt_result)):
        predicted_result = predicted_result[:len(gt_result)]
    else:
        predicted_result = np.pad(predicted_result, (0, len(gt_result) - len(predicted_result)))
    return sklearn.metrics.f1_score(gt_result > 0, predicted_result > 0)

def evaluate_result(predicted_result, gt_midi_path, downbeat_bins, subbeat_count, n_layers):
    midi_gt = pretty_midi.PrettyMIDI(gt_midi_path)
    n_subbeat_gt, downbeat_bins_gt, boundaries_gt, subbeat_time_gt = prepare_quantization(midi_gt, subbeat_count)
    notes = [[np.searchsorted(boundaries_gt, note.start), note.pitch] for note in midi_gt.instruments[0].notes]
    note_dict = np.zeros(n_subbeat_gt, dtype=np.int16)
    gt_result = np.zeros(len(downbeat_bins_gt), dtype=np.int16)
    if (len(predicted_result) >= len(gt_result)):
        predicted_result = predicted_result[:len(gt_result)]
    else:
        predicted_result = np.pad(predicted_result, (0, len(gt_result) - len(predicted_result)))
    for note in notes:
        if (note[1] >= 40):
            note_dict[note[0]] = max(note_dict[note[0]], note[1] - 39)
    for i in range(len(gt_result)):
        if (downbeat_bins[i] < n_subbeat_gt):
            gt_result[i] = note_dict[downbeat_bins[i]]
    return [sklearn.metrics.f1_score(gt_result >= i, predicted_result >= i) for i in range(1, n_layers + 1)]


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


def oracle_analyze(file_name, subbeat_count=4):
    midi = pretty_midi.PrettyMIDI(file_name)
    n_subbeat, downbeat_bins, boundaries, subbeat_time = prepare_quantization(midi, subbeat_count)
    template = np.array([4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0])
    results = []
    for i in range(16):
        template = np.roll(template, 1)
        pred = template[np.arange(len(downbeat_bins)) % 16]
        gt_file_path = 'annotation/%s_gt.mid' % os.path.basename(file_name)
        results.append(evaluate_result(pred, gt_file_path, downbeat_bins, subbeat_count, 4))
    # print(results)
    return np.max(results, axis=0)


def evaluations_to_latex(set_name, evaluations):
    mean_evaluations = np.mean(evaluations, axis=0)
    std_evaluations = np.std(evaluations, axis=0)
    if ('\n' in set_name):
        set_name = R'\begin{tabular}[c]{@{}l@{}}%s\end{tabular}' % (set_name.replace('\n', '\\\\ '))
    return(set_name + '\t' + '\t'.join(R'& \begin{tabular}[c]{@{}r@{}}%.4f \\ $\pm$%.4f\end{tabular}' % (mean_evaluation, std_evaluation)
                     for (mean_evaluation, std_evaluation) in zip(mean_evaluations, std_evaluations)) + R' \\ \midrule')

if __name__ == '__main__':
    # generate_labels(23, 40, 'hierarchical_dp_analysis_rwc_test_v3', 'none')
    # generate_labels(0, 33, 'hierarchical_gt_rwc_train_v6', 'midi')

    split_files = get_split('rwc_multitrack_hierarchy_v6_supervised', 'test')
    print(evaluations_to_latex('Oracle',
        [oracle_analyze(os.path.join(RWC_DATASET_PATH, 'AIST.RWC-MDB-P-2001.SMF_SYNC', file)) for file in split_files]))
    print(evaluations_to_latex('Rule',
        [joint_analyze(os.path.join(RWC_DATASET_PATH, 'AIST.RWC-MDB-P-2001.SMF_SYNC', file), save_type='f1',
                    drums=1, melody=1, others=1) for file in split_files]))
    print(evaluations_to_latex('Rule\n (mel. only)',
        [joint_analyze(os.path.join(RWC_DATASET_PATH, 'AIST.RWC-MDB-P-2001.SMF_SYNC', file), save_type='f1',
                    drums=0, melody=1, others=0) for file in split_files]))
    print(evaluations_to_latex('Rule\n (no drums)',
        [joint_analyze(os.path.join(RWC_DATASET_PATH, 'AIST.RWC-MDB-P-2001.SMF_SYNC', file), save_type='f1',
                    drums=0, melody=1, others=1) for file in split_files]))

    print('\n\nPOP909\n\n')
    print(evaluations_to_latex('Oracle', [oracle_analyze(R'input/POP909-%d.mid' % (i + 1)) for i in range(5)]))
    print(evaluations_to_latex('Rule', [joint_analyze(R'input/POP909-%d.mid' % (i + 1), save_type='f1') for i in range(5)]))
    print(evaluations_to_latex('Rule\n(mel. only)', [joint_analyze(R'input/POP909-%d.mid' % (i + 1), tracks=[0], save_type='f1') for i in range(5)]))
    joint_analyze(R'E:\Dataset\RWC\AIST.RWC-MDB-P-2001.SMF_SYNC\RM-P051.SMF_SYNC.MID', save_type='f1')
    exit(0)
