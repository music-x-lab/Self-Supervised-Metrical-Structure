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
from simple_tcn import TCN, TCNBlock
from mir.nn.data_decorator import NoPitchShifter
from metrical_crf import get_binary_transition

TRAIN_LENGTH = 1024

class TCNClassifier(NetworkBehavior):

    def __init__(self, in_channels, hidden_dim, n_layers, n_classes, dropout, max_seq_length=8192, remap_offset=None):
        super().__init__()
        self.max_seq_length = max_seq_length
        self.pre_tcn = nn.Sequential(
            TCNBlock(in_channels, hidden_dim, 1, dropout),
            nn.MaxPool1d(2),
            TCNBlock(hidden_dim, hidden_dim, 1, dropout),
            nn.MaxPool1d(2),
            TCNBlock(hidden_dim, hidden_dim, 1, dropout),
            nn.MaxPool1d(2),
        )
        self.tcn = TCN(hidden_dim, hidden_dim, n_layers, dropout)
        self.linear = nn.Linear(hidden_dim, n_classes)
        self.n_classes = n_classes
        self.transition, self.z_to_layer = get_binary_transition(
            np.array([-np.inf, -np.inf, -np.inf, -np.inf, -15.0, -15.0, -15.0, -15.0]),
            np.array([-np.inf, -np.inf, -np.inf, -np.inf, -20.0, -20.0, -20.0, -20.0]),
        )
        self.crf = CRFDecoder(torch.tensor(self.transition), transition_as_parameter=True)
        self.remap_offset = remap_offset

    def forward(self, x):
        h = self.pre_tcn(x.transpose(1, 2)).transpose(1, 2)
        h = self.tcn(h)
        return F.log_softmax(self.linear(h), dim=-1)

    def loss(self, x):
        batch_size, seq_length, _ = x.shape
        x = x.view(batch_size, seq_length * 8, -1)
        log_prob = self(x)
        log_obs = log_prob[:, :, self.z_to_layer]
        return self.crf.neg_log_likelihood_batch(log_obs).mean()

    def inference(self, x):
        seq_length, _ = x.shape
        x = x.view(1, seq_length * 8, -1)
        log_prob = self(x)
        return log_prob.squeeze(0).cpu().numpy()


if __name__ == '__main__':
    model_name = 'tcn_audio_metrical_v1.0_crf'
    # train_provider, val_provider = get_providers('rwc_multitrack_hierarchy_v5_semi')
    storage_x = FramedRAMDataStorage('osu_beat_aligned_stft_div_4_8')
    song_count = storage_x.get_length()
    is_valid = np.ones(song_count,dtype=np.bool)
    for i in range(song_count):
        if (i % 5 == 0):
            is_valid[i]=False
    train_indices=np.arange(song_count)[is_valid]
    val_indices=np.arange(song_count)[np.bitwise_not(is_valid)]
    trainer = NetworkInterface(TCNClassifier(513, 256, 6, 9, 0.5),
        model_name, load_checkpoint=True)
    train_provider=FramedDataProvider(train_sample_length=TRAIN_LENGTH,shift_low=0,shift_high=0,sample_step=2)
    train_provider.link(storage_x, NoPitchShifter(), subrange=train_indices)
    val_provider=FramedDataProvider(train_sample_length=TRAIN_LENGTH,shift_low=0,shift_high=0,sample_step=2)
    val_provider.link(storage_x, NoPitchShifter(), subrange=val_indices)

    trainer.train_supervised(train_provider,
                             val_provider,
                             batch_size=16,
                             learning_rates_dict={1e-4: 2000},
                             round_per_print=100,
                             round_per_val=500,
                             round_per_save=1000)