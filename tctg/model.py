import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        '''x: (batch_size, max_len) of longs'''
        pass


class RNN_VAE(nn.Module):
    def __init__(self, unk_idx=0, pad_idx=1, start_idx=2, eos_idx=3,
                 max_len=15):
        super().__init__()

        self.unk_idx = unk_idx
        self.pad_idx = pad_idx
        self.start_idx = start_idx
        self.eos_idx = eos_idx

        self.max_len = max_len

        self.input_linear = nn.Linear(10, 10)
        self.middle_linear = nn.Linear(10, 10)
        self.output_linear = nn.Linear(10, 10)


if __name__ == '__main__':
    model = RNN_VAE()
