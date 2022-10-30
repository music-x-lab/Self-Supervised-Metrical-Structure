from simple_tcn import TCNClassifier, NetworkInterface, N_MIDI_PITCH, CONTEXT_LENGTH
import numpy as np
from midi_structure import get_piano_roll, prepare_quantization, evaluate_result, evaluations_to_latex, get_split
import pretty_midi
import os
from settings import LMD_MATCHED_FOLDER, RWC_DATASET_PATH
import matplotlib.pyplot as plt
import torch
from crf import CRFDecoder
from metrical_crf import get_ternary_transition
from scipy.ndimage.filters import uniform_filter1d

def decode(log_observations):
    log_observations = torch.tensor(log_observations)
    # log_transitions = get_log_transitions(4)
    log_transitions, indices = get_ternary_transition(np.array([-5.0, -4.0, -3.0, -2.0]), np.array([-8.0, -7.0, -6.0, -5.0]))
    log_observations = log_observations[:, indices]
    crf = CRFDecoder(torch.tensor(log_transitions))
    result = crf.viterbi_decode(log_observations[None]).squeeze(0).numpy()
    return indices[result]


def model_eval(model, midi_path, subbeat_count=4, drums=1, melody=1, others=1, visualize=True, tracks=None, crf=True):
    # print('Evaluating:', midi_path)
    try:
        midi = pretty_midi.PrettyMIDI(midi_path)
    except:
        print('Midi load failed: %s' % midi_path)
        return None
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
        # visualized_preds = np.cumsum(visualized_preds[:, ::-1], axis=1)
        # plt.figure(figsize=(26, 6))
        # plt.imshow(np.concatenate((piano_rolls[j][:, ::-1], np.repeat(visualized_preds, 16, axis=1)), axis=1).T, interpolation='nearest')
        # plt.title(os.path.basename(midi_path))
        # plt.show()
    if (len(rolls) == 0):
        print('No track!')
        return None
    # print('Tracks: %d' % (len(rolls)))
    rolls = np.stack(rolls, axis=0)
    log_final_pred, log_conf = model.inference_function('inference_song', rolls.astype(np.float32), return_log_prob=True)
    log_final_pred = np.log(uniform_filter1d(np.exp(log_final_pred), size=5, axis=0))
    used_downbeats = downbeat_bins[downbeat_bins < len(log_final_pred)]
    log_downbeat_pred = log_final_pred[used_downbeats]
    if (crf == True):
        result = decode(log_downbeat_pred)
    else:
        result = np.argmax(log_downbeat_pred, axis=-1)
    if (visualize):
        onehot_result = np.eye(5)[result]
        final_pred = np.exp(log_final_pred)
        visualized_preds = np.cumsum(final_pred[:, ::-1], axis=1)
        visualized_result = np.zeros((final_pred.shape[0], 5))
        visualized_result[used_downbeats] = onehot_result
        visualized_result = np.cumsum(visualized_result[:, ::-1], axis=1)
        plt.figure(figsize=(26, 6))
        plt.imshow(np.concatenate((rolls.max(axis=0)[:, ::-1], np.repeat(visualized_preds, 16, axis=1), np.repeat(visualized_result, 16, axis=1)), axis=1).T)
        plt.title(os.path.basename(midi_path) + ' final')
        plt.show()
    gt_midi_path = 'annotation/%s_gt.mid' % os.path.basename(midi_path)
    if (os.path.exists(gt_midi_path)):
        evaluation = evaluate_result(result, gt_midi_path, downbeat_bins, subbeat_count, 4)
        # print(('%s:\t' % gt_midi_path) + '\t'.join(str(x) for x in evaluation))
    else:
        evaluation = None
    # print(conf)
    output = pretty_midi.Instrument(program=0, is_drum=True, name='Layers')
    for i, pred in enumerate(result):
        for k in range(pred):
            onset_time = subbeat_time[downbeat_bins[i]]
            output.notes.append(pretty_midi.Note(velocity=100, pitch=40 + k, start=onset_time, end=onset_time + 0.5))

    midi.instruments.append(output)
    if not (os.path.exists('output/%s' % model.save_name)):
        os.mkdir('output/%s' % model.save_name)
    midi.write('output/%s/%s_crf.mid' % (model.save_name, os.path.basename(midi_path)))
    np.savetxt('output/%s/%s_conf.txt' % (model.save_name, os.path.basename(midi_path)), log_conf)
    f = open('output/%s/%s_conf_ins.txt' % (model.save_name, os.path.basename(midi_path)), 'w')
    f.write(','.join(ins_names))
    f.close()
    return evaluation


def evaluate_lmd(model, count):
    f = open('data/lmd_matched_usable_midi.txt', 'r')
    lines = [line.strip() for line in f.readlines() if line.strip() != '']
    f.close()
    np.random.seed(6172)
    np.random.shuffle(lines)
    lines = lines[:count]
    for line in lines:
        model_eval(model, os.path.join(LMD_MATCHED_FOLDER, line), visualize=False)


def main():
    model = NetworkInterface(TCNClassifier(384, 256, 6, 5, 0.1),
        'simple_tcn_v2.2_fixed_shift', load_checkpoint=False)
    # evaluate_lmd(model, 9999)
    # exit(0)
    # model_eval(model, R'E:\Dataset\lmd_matched\L\C\N\TRLCNWM128F423BB63\7596e59dea60afab6bbc7207aca8bd8c.mid')
    print(evaluations_to_latex('Proposed\n(mel. only)',
        [model_eval(model, R'input/POP909-%d.mid' % (i + 1), tracks=[0]) for i in range(5)]))
    print(evaluations_to_latex('Proposed',
        [model_eval(model, R'input/POP909-%d.mid' % (i + 1)) for i in range(5)]))

    split_files = get_split('rwc_multitrack_hierarchy_v6_supervised', 'test')
    print(evaluations_to_latex('Proposed\nw/o CRF',
        [model_eval(model, os.path.join(RWC_DATASET_PATH, 'AIST.RWC-MDB-P-2001.SMF_SYNC', file),
                    drums=1, melody=1, others=1, visualize=False, crf=False) for file in split_files]))
    print(evaluations_to_latex('Proposed\n(mel. only)',
        [model_eval(model, os.path.join(RWC_DATASET_PATH, 'AIST.RWC-MDB-P-2001.SMF_SYNC', file),
                    drums=0, melody=1, others=0, visualize=False) for file in split_files]))
    print(evaluations_to_latex('Proposed\n(no drums)',
        [model_eval(model, os.path.join(RWC_DATASET_PATH, 'AIST.RWC-MDB-P-2001.SMF_SYNC', file),
                    drums=0, melody=1, others=1, visualize=False) for file in split_files]))
    print(evaluations_to_latex('Proposed',
        [model_eval(model, os.path.join(RWC_DATASET_PATH, 'AIST.RWC-MDB-P-2001.SMF_SYNC', file),
                    drums=1, melody=1, others=1, visualize=False) for file in split_files]))
    exit(0)
    evaluations = []
    print(model.save_name)
    for file in split_files:
        evaluation = model_eval(model, os.path.join(RWC_DATASET_PATH, 'AIST.RWC-MDB-P-2001.SMF_SYNC', file),
                   drums=1, melody=1, others=1)
        if (evaluation is not None):
            evaluations.append(evaluation)
    if (len(evaluations) > 0):
        mean_evaluation = np.mean(evaluations, axis=0)
        print('Mean evaluation:\t' + '\t'.join('%.4f' % x for x in mean_evaluation))

if __name__ == '__main__':
    main()

