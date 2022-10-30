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
from tcn_downbeat_unsupervised import get_providers

N_MIDI_PITCH = 128
SHIFT_LOW = -5
SHIFT_HIGH = 6

CONTEXT_LENGTH = 1024


class TCNClassifierSupervised(NetworkBehavior):

    def __init__(self, in_channels, hidden_dim, n_layers, n_classes, dropout, max_seq_length=8192):
        super().__init__()
        self.max_seq_length = max_seq_length
        self.tcn = TCN(in_channels, hidden_dim, n_layers, dropout)
        self.linear = nn.Linear(hidden_dim, n_classes)
        self.confidence_linear = nn.Linear(hidden_dim, 1)
        self.n_classes = n_classes
        self.remap_offset = 0

    def forward(self, x):
        h = self.tcn(x)
        return self.linear(h), self.confidence_linear(h)

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
        pred1, conf1 = self(x1)
        pred2, conf2 = self(x2)
        log_pred1 = F.log_softmax(pred1, dim=-1)
        log_pred2 = F.log_softmax(pred2, dim=-1)
        log_prob = self.log_prob(log_pred1, conf1, log_pred2, conf2)
        pred = log_prob.view(-1, log_prob.shape[-1])
        y = y[:, :, 0].byte()
        return F.nll_loss(pred, y.view(-1))

    def inference_song(self, xs):
        logits, conf = self(xs)
        # print('Fix me: inverted conf')
        # conf = -conf
        log_alpha = torch.log_softmax(conf, dim=0)
        log_prob = torch.logsumexp(
            log_alpha + F.log_softmax(logits, dim=-1), dim=0
        )
        return log_prob.cpu().numpy(), log_alpha.squeeze(-1).cpu().numpy()

if __name__ == '__main__':
    model_name = 'tcn_downbeat_supervised_v1.2'
    # train_provider, val_provider = get_providers('rwc_multitrack_hierarchy_v5_semi')
    train_provider, val_provider = get_providers('lmd_multitrack_hierarchy_v7_unsupervised')
    trainer = NetworkInterface(TCNClassifierSupervised(384, 256, 6, 3, 0.5),
        model_name, load_checkpoint=True)
    trainer.train_supervised(train_provider,
                             val_provider,
                             batch_size=16,
                             learning_rates_dict={1e-4: 5},
                             round_per_print=100,
                             round_per_val=500,
                             round_per_save=1000)