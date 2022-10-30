import torch.nn as nn
import torch
import torch.nn.functional as F
from mir.nn.data_storage import FramedRAMDataStorage
from mir.nn.data_provider import DataProvider, default_collate, data_type_fix
from mir.nn.train import NetworkBehavior, NetworkInterface
from mir.nn.data_provider import FramedDataProvider, data_type_fix
from modules.vae import Reparameterizer
from crf import CRFDecoder
import numpy as np
from scipy.ndimage.filters import maximum_filter1d

N_MIDI_PITCH = 128
SHIFT_LOW = -12
SHIFT_HIGH = 12

CONTEXT_LENGTH = 512

class TCNBlock(nn.Module):

    def __init__(self, in_channels, out_channels, dilation, dropout):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, (3, ), padding=(dilation, ), dilation=(dilation, )),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, (3, ), padding=(dilation, ), dilation=(dilation, )),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
        if (self.in_channels != self.out_channels):
            self.linear = nn.Conv1d(self.in_channels, self.out_channels, (1, ))

    def forward(self, x):
        if (self.in_channels != self.out_channels):
            return self.layers(x) + self.linear(x)
        return self.layers(x) + x

class TCN(nn.Module):

    def __init__(self, in_channels, out_channels, n_layers, dropout):
        super().__init__()
        layers = []
        dilation = 1
        for i in range(n_layers):
            layers.append(TCNBlock(in_channels if i == 0 else out_channels, out_channels, dilation, dropout))
            dilation *= 2
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x.transpose(1, 2)).transpose(1, 2)

class TCNClassifier(NetworkBehavior):

    def __init__(self, in_channels, hidden_dim, n_layers, n_classes, dropout):
        super().__init__()
        self.tcn = TCN(in_channels, hidden_dim, n_layers, dropout)
        self.linear = nn.Linear(hidden_dim, n_classes)
        self.confidence_linear = nn.Linear(hidden_dim, 1)
        self.n_classes = n_classes

    def forward(self, x):
        h = self.tcn(x)
        return self.linear(h), self.confidence_linear(h)

    def log_prob(self, logits1, conf1, logits2, conf2):
        log_alpha = torch.log_softmax(torch.cat([conf1, conf2], dim=-1), dim=-1)
        log_prob = torch.logsumexp(
            torch.stack([
                log_alpha[:, :, None, 0] + F.log_softmax(logits1, dim=-1),
                log_alpha[:, :, None, 1] + F.log_softmax(logits2, dim=-1)
            ], dim=-1), dim=-1
        )
        return log_prob

    def loss(self, x1, x2, y, downbeat_bins):
        pred1, conf1 = self(x1)
        pred2, conf2 = self(x2)
        log_prob = self.log_prob(pred1,
                                 conf1,
                                 pred2,
                                 conf2)
        y_hierarchy = y[:, :, 1].contiguous()
        return F.nll_loss(log_prob.view(-1, log_prob.shape[-1]), y_hierarchy.view(-1), ignore_index=-1)

    def inference_song(self, xs, return_log_prob=False):
        logits, conf = self(xs)
        log_alpha = torch.log_softmax(conf, dim=0)
        log_prob = torch.logsumexp(
            log_alpha + F.log_softmax(logits, dim=-1), dim=0
        )
        if (return_log_prob):
            return log_prob.cpu().numpy(), conf.squeeze(-1).cpu().numpy()
        else:
            return torch.exp(log_prob).cpu().numpy(), conf.squeeze(-1).cpu().numpy()

    def inference(self, x):
        logits, conf = self(x[None])
        return F.softmax(logits, dim=-1).squeeze(0).cpu().numpy(), conf.squeeze(0).cpu().numpy()


class HierarchicalDataProvider(DataProvider):

    def __init__(self, file_name, subrange, shift_low, shift_high, context_length, samples_per_song):
        super().__init__(True, default_collate)
        self.num_workers = 0
        self.length = np.load(file_name + '.length.npy')
        self.start = np.concatenate((np.zeros(1, dtype=int), np.cumsum(self.length)), axis=0)
        self.data = np.load(file_name + '.npy')
        self.subrange = subrange
        self.shift_low = shift_low
        self.shift_high = shift_high
        self.valid_song_count = len(subrange)
        self.context_length = context_length
        self.samples_per_song = samples_per_song

    def init_worker(self, worker_id, is_training_set):
        pass # np.random.seed(worker_id + 1)

    def get_length(self):
        return self.valid_song_count * self.samples_per_song * (self.shift_high - self.shift_low + 1)

    def pitch_shift(self, raw_data, start, end, shift, arr, return_labels):
        pad_left = max(-start, 0)
        pad_right = max(end - len(raw_data), 0)
        data_labels = np.pad(raw_data[start + pad_left:end - pad_right], ((pad_left, pad_right), (0, 0)))
        data = data_labels[:, 2:]
        labels = data_labels[:, :2].astype(np.int64)
        labels[:, 1] = maximum_filter1d(labels[:, 1], size=5)
        result = None
        if (np.all(data == 0)):
            result = np.zeros((data.shape[0], data.shape[1] * 3))
        else:
            for i in arr:
                new_data = np.bitwise_and(data, 3 << (int(i) * 2))
                if (np.any(new_data)):
                    onset = np.bitwise_and(new_data, 1 << (int(i) * 2)) != 0
                    roll = np.bitwise_and(new_data, 1 << (int(i) * 2 + 1)) != 0
                    if not (np.any(roll)):  # drum roll
                        result_drum = onset
                        result_roll = np.zeros_like(roll)
                        result_onset = np.zeros_like(roll)
                    else:  # augmentation for non-drum tracks
                        result_drum = np.zeros_like(roll)
                        def shift_roll(roll, shift):
                            if (shift > 0):
                                return np.pad(roll[:, :-shift], ((0, 0), (shift, 0)))
                            elif (shift < 0):
                                return np.pad(roll[:, -shift:], ((0, 0), (0, -shift)))
                            return roll
                        result_roll = shift_roll(roll, shift)
                        result_onset = shift_roll(onset, shift)
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

            return result.astype(np.float32), labels, downbeat_bins
        else:
            return result.astype(np.float32)

    def get_sample(self, id):
        shift = id % (self.shift_high - self.shift_low + 1) + self.shift_low
        raw_id = id // (self.shift_high - self.shift_low + 1) % self.valid_song_count
        shift2 = np.random.randint(self.shift_low, self.shift_high + 1)
        song_id = self.subrange[raw_id]
        data = self.data[self.start[song_id]:self.start[song_id] + self.length[song_id]]
        id = np.random.randint(len(data) - self.context_length)
        arr = np.arange(32)
        np.random.shuffle(arr)
        return (self.pitch_shift(data, id, id + self.context_length, shift, arr, False),
            *self.pitch_shift(data, id, id + self.context_length, shift2, arr[::-1], True))

def get_providers(data_file, use_pitch_shift):
    f = open('./data/%s.split.txt' % data_file, 'r')
    tokens = [line.strip().split(',') for line in f.readlines() if line.strip() != '']
    f.close()
    train_indices = np.array([int(id) for id in tokens[0]])
    val_indices = np.array([int(id) for id in tokens[1]])
    print('%s: Using %d samples to train' % (data_file, len(train_indices)))
    print('%s: Using %d samples to val' % (data_file, len(val_indices)))
    train_provider = HierarchicalDataProvider('data/%s' % data_file, train_indices, SHIFT_LOW if use_pitch_shift else 0,
                                              SHIFT_HIGH if use_pitch_shift else 0, CONTEXT_LENGTH, 5 if use_pitch_shift else 130)
    val_provider = HierarchicalDataProvider('data/%s' % data_file, val_indices, 0, 0, CONTEXT_LENGTH, 5)
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
    np.random.seed(0)
    torch.manual_seed(0)
    model_name = 'simple_tcn_v2.2_fixed_shift'
    use_pitch_shift = 'no_pitch_shift' not in model_name
    train_provider, val_provider = get_providers('rwc_multitrack_hierarchy_v8_supervised_fix_onset', use_pitch_shift)
    trainer = NetworkInterface(TCNClassifier(384, 256, 6, 5, 0.5),
        model_name, load_checkpoint=True)
    trainer.train_supervised(train_provider,
                             val_provider,
                             batch_size=16,
                             learning_rates_dict={1e-4: 100},
                             round_per_print=100,
                             round_per_val=500,
                             round_per_save=1000)

