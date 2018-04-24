import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable


class Encoder(nn.Module):
    def __init__(self, n_vocab, d_emb=150, dropout=0.1, d_rnn=300,
                 n_layers=1, d_out=300):
        super().__init__()

        self.emb = nn.Embedding(n_vocab, d_emb)
        self.drop = nn.Dropout(dropout)
        self.rnn = nn.GRU(d_emb, d_rnn, n_layers, dropout=dropout)
        self.enc_mu = nn.Linear(d_rnn, d_out)
        self.enc_log_sigma = nn.Linear(d_rnn, d_out)

    def forward(self, x):
        '''x: (batch_size, max_len) of longs'''
        pass

    def _sample_z(self, h):
        mu, log_sigma = self.enc_mu(h), self.enc_log_sigma(h)
        sigma = torch.exp(log_sigma)

        std = torch.from_numpy(
            np.random.normal(0, 1, size=sigma.size())
        ).float()

        # Reparameterization trick
        return mu + sigma * Variable(std, requires_grad=False)


class RNN_VAE(nn.Module):
    def __init__(self, unk_idx=0, pad_idx=1, start_idx=2, eos_idx=3,
                 max_len=15):
        super().__init__()

        self.unk_idx = unk_idx
        self.pad_idx = pad_idx
        self.start_idx = start_idx
        self.eos_idx = eos_idx

        self.max_len = max_len

        self._input_linear = nn.Linear(10, 10)
        self.middle_linear = nn.Linear(10, 10)
        self.output_linear = nn.Linear(10, 10)


if __name__ == '__main__':
    model = RNN_VAE()
    print(len(list(model.parameters())))
