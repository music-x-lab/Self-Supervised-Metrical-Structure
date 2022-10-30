import torch.nn as nn
import torch
import torch.nn.functional as F
from mir.nn.data_storage import FramedRAMDataStorage
from mir.nn.data_provider import DataProvider, default_collate, data_type_fix
from mir.nn.train import NetworkBehavior, NetworkInterface
from mir.nn.data_provider import FramedDataProvider, data_type_fix
from modules.vae import Reparameterizer
from crf import CRFDecoder
from metrical_crf import get_binary_transition
import numpy as np
from simple_tcn import TCN

N_MIDI_PITCH = 128
SHIFT_LOW = -5
SHIFT_HIGH = 6

CONTEXT_LENGTH = 1024


class TCNClassifier(NetworkBehavior):

    def __init__(self, in_channels, hidden_dim, n_layers, n_classes, dropout, max_seq_length=8192, use_consistency_loss=True, remap_offset=None):
        super().__init__()
        self.use_consistency_loss = use_consistency_loss
        self.max_seq_length = max_seq_length
        self.tcn = TCN(in_channels, hidden_dim, n_layers, dropout)
        self.linear = nn.Linear(hidden_dim, n_classes)
        self.confidence_linear = nn.Linear(hidden_dim, 1)
        self.onset_counter = nn.Linear(hidden_dim, 5)
        self.n_classes = n_classes
        self.transition, self.z_to_layer = get_binary_transition(
            np.array([-np.inf, -np.inf, -np.inf, -np.inf, -15.0, -15.0, -15.0, -15.0]),
            np.array([-np.inf, -np.inf, -np.inf, -np.inf, -20.0, -20.0, -20.0, -20.0]),
        )
        self.crf = CRFDecoder(torch.tensor(self.transition), transition_as_parameter=True)
        self.remap_offset = remap_offset

    def forward(self, x):
        h = self.tcn(x)
        return self.linear(h), self.confidence_linear(h), self.onset_counter(h)

    def log_prob(self, log_pred1, conf1, log_pred2, conf2):
        log_alpha = torch.log_softmax(torch.cat([conf1, conf2], dim=-1), dim=-1)
        log_prob = torch.logsumexp(
            torch.stack([
                log_alpha[:, :, None, 0] + log_pred1,
                log_alpha[:, :, None, 1] + log_pred2
            ], dim=-1), dim=-1
        )
        return log_prob

    def loss(self, x1, x2p, x2, y, downbeat_bins, delta_scale):
        batch_size, seq_length, _ = y.shape
        pred1, conf1, onset_result1 = self(x1)
        pred2, conf2, _ = self(x2)
        log_pred1 = F.log_softmax(pred1, dim=-1)
        log_pred2 = F.log_softmax(pred2, dim=-1)
        log_prob = self.log_prob(log_pred1, conf1, log_pred2, conf2)
        log_obs = log_prob[:, :, self.z_to_layer]
        log_obs12 = (log_pred1 + log_pred2)[:, :, self.z_to_layer]
        if self.use_consistency_loss:
            return self.crf.neg_log_likelihood_batch(torch.cat([log_obs, log_obs12], dim=0)).mean() * 2
        else:
            return self.crf.neg_log_likelihood_batch(log_obs).mean()

    def inference_song(self, xs):
        logits, conf, _ = self(xs)
        # print('Fix me: inverted conf')
        # conf = -conf
        log_alpha = torch.log_softmax(conf, dim=0)
        log_prob = torch.logsumexp(
            log_alpha + F.log_softmax(logits, dim=-1), dim=0
        )
        return log_prob.cpu().numpy(), log_alpha.squeeze(-1).cpu().numpy()

class HierarchicalDataProvider(DataProvider):

    def __init__(self, file_name, subrange, shift_low, shift_high, context_length):
        super().__init__(True, default_collate)
        self.num_workers = 0
        self.length = np.load(file_name + '.length.npy')
        self.start = np.concatenate((np.zeros(1, dtype=int), np.cumsum(self.length)), axis=0)
        self.data = np.load(file_name + '.npy')
        self.subrange = [x for x in subrange if self.length[x] >= context_length]
        self.shift_low = shift_low
        self.shift_high = shift_high
        self.valid_song_count = len(self.subrange)
        self.context_length = context_length
        # print(f'valid: {self.subrange} for {context_length}', flush=True)

    def init_worker(self, worker_id, is_training_set):
        pass

    def get_length(self):
        return self.valid_song_count * 10 * (self.shift_high - self.shift_low + 1)

    def pitch_shift(self, raw_data, start, end, shift, arr, return_labels):
        pad_left = max(-start, 0)
        pad_right = max(end - len(raw_data), 0)
        data_labels = np.pad(raw_data[start + pad_left:end - pad_right], ((pad_left, pad_right), (0, 0)))
        data = data_labels[:, 2:]
        labels = data_labels[:, :2]
        # labels = raw_data[:, 0] * 2  # create 2 internal metrical layers
        # labels[::2] = np.maximum(labels[::2], 1)
        # labels[8:-8] = np.maximum(labels[8:-8], np.logical_and(labels[16:] == 4, labels[:-16] == 4) * 3)
        # labels = np.pad(labels[start + pad_left:end - pad_right], ((pad_left, pad_right),))
        result = None
        if (np.all(data == 0)):
            result = np.zeros((data.shape[0], data.shape[1] * 3))
        else:
            for i in arr:
                new_data = np.bitwise_and(data, 3 << (int(i) * 2))
                if (np.any(new_data)):
                    onset = np.bitwise_and(new_data, 1 << (int(i) * 2)) != 0
                    roll = np.bitwise_and(new_data, 1 << (int(i) * 2 + 1)) != 0
                    # print('onset:', onset.sum(), 'roll:', roll.sum())
                    if not (np.any(roll)):  # drum roll
                        result_drum = onset
                        result_roll = np.zeros_like(roll)
                        result_onset = np.zeros_like(roll)
                    else:  # augmentation for non-drum tracks
                        result_drum = np.zeros_like(roll)
                        result_roll = np.roll(roll, shift, axis=-1)
                        result_onset = np.roll(onset, shift, axis=-1)
                    result = np.concatenate((result_onset, result_roll, result_drum), axis=-1)
                    break
        if (return_labels):
            downbeat_bins = np.where(data_labels[:, 0] == 2)[0]
            # retain 32 downbeat bins, pad if necessary
            desired_downbeat_count = 32
            if (len(downbeat_bins) > desired_downbeat_count):
                clip_start = np.random.randint(len(downbeat_bins) - desired_downbeat_count)
                downbeat_bins = downbeat_bins[clip_start: clip_start + desired_downbeat_count]
            elif (len(downbeat_bins) < desired_downbeat_count):
                # todo: better padding
                downbeat_bins = np.pad(downbeat_bins, ((0, desired_downbeat_count - len(downbeat_bins)),), mode='reflect')

            return result.astype(np.float32), labels.astype(np.int64), downbeat_bins
        else:
            return result.astype(np.float32)

    def get_sample(self, id):
        shift = id % (self.shift_high - self.shift_low + 1) + self.shift_low
        raw_id = id // (self.shift_high - self.shift_low + 1) % self.valid_song_count
        shift2 = np.random.randint(self.shift_low, self.shift_high + 1)
        song_id = self.subrange[raw_id]
        # print(f'{song_id}, len={self.length[song_id]}', flush=True)
        data = self.data[self.start[song_id]:self.start[song_id] + self.length[song_id]]
        id = np.random.randint(len(data) - self.context_length + 1) // 2 * 2
        arr = np.arange(32)
        np.random.shuffle(arr)
        return (self.pitch_shift(data, id, id + self.context_length, shift, arr, False),
            self.pitch_shift(data, id, id + self.context_length, shift, arr[::-1], False),
            *self.pitch_shift(data, id, id + self.context_length, shift2, arr[::-1], True),
            shift2 - shift)

def get_providers(data_file):
    f = open('./data/%s.split.txt' % data_file, 'r')
    tokens = [line.strip().split(',') for line in f.readlines() if line.strip() != '']
    f.close()
    train_indices = np.array([int(id) for id in tokens[0]])
    val_indices = np.array([int(id) for id in tokens[1]])
    print('%s: Using %d samples to train' % (data_file, len(train_indices)))
    print('%s: Using %d samples to val' % (data_file, len(val_indices)))
    train_provider = HierarchicalDataProvider('data/%s' % data_file, train_indices, SHIFT_LOW,
                                              SHIFT_HIGH, CONTEXT_LENGTH)
    val_provider = HierarchicalDataProvider('data/%s' % data_file, val_indices, SHIFT_LOW, SHIFT_HIGH,
                                            CONTEXT_LENGTH)
    return train_provider, val_provider

class JointProvider(DataProvider):

    def __init__(self, provider1, provider2):
        super().__init__(True, default_collate)
        self.num_workers = 0
        self.provider1 = provider1
        self.provider2 = provider2
        self.length1 = self.provider1.get_length()
        self.length2 = self.provider2.get_length()

    def init_worker(self, worker_id, is_training_set):
        self.provider1.init_worker(worker_id, is_training_set)
        self.provider2.init_worker(worker_id, is_training_set)

    def get_length(self):
        return self.length1 + self.length2

    def get_sample(self, id):
        if (id >= self.length1):
            return self.provider2.get_sample(id - self.length1)
        else:
            return self.provider1.get_sample(id)

if __name__ == '__main__':
    model_name = 'tcn_downbeat_unsupervised_v3.0_1024_context_6_tcn'
    # train_provider, val_provider = get_providers('rwc_multitrack_hierarchy_v5_semi')
    train_provider, val_provider = get_providers('lmd_multitrack_hierarchy_v7_unsupervised')
    trainer = NetworkInterface(TCNClassifier(384, 256, 6, 9, 0.5, use_consistency_loss='noc' not in model_name),
        model_name, load_checkpoint=True)
    trainer.train_supervised(train_provider,
                             val_provider,
                             batch_size=16,
                             learning_rates_dict={1e-4: 10},
                             round_per_print=100,
                             round_per_val=500,
                             round_per_save=1000)