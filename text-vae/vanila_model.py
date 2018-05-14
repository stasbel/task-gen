import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import assert_check


class RnnVae(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.n_len = kwargs['n_len']
        self.d_h = kwargs['d_h']
        self.d_z = kwargs['d_z']
        self.d_c = kwargs['d_c']
        self.p_word_dropout = kwargs['p_word_dropout']
        self.freeze_embeddings = kwargs['freeze_embeddings']
        x_vocab = kwargs['x_vocab']
        self.unk = x_vocab.stoi['<unk>']
        self.pad = x_vocab.stoi['<pad>']
        self.bos = x_vocab.stoi['<bos>']
        self.eos = x_vocab.stoi['<eos>']
        self.n_vocab = len(x_vocab)

        # Word embeddings layer
        if x_vocab is None:
            self.d_emb = self.d_h
            self.x_emb = nn.Embedding(self.n_vocab, self.d_emb, self.pad)
        else:
            self.d_emb = x_vocab.vectors.size(1)
            self.x_emb = nn.Embedding(self.n_vocab, self.d_emb, self.pad)

            # Set pretrained embeddings
            self.x_emb.weight.data.copy_(x_vocab.vectors)

            if self.freeze_embeddings:
                self.x_emb.weight.requires_grad = False

        # Encoder
        self.encoder_rnn = nn.GRU(self.d_emb, self.d_h)
        self.q_mu = nn.Linear(self.d_h, self.d_z)
        self.q_logvar = nn.Linear(self.d_h, self.d_z)

        # Decoder
        self.decoder_rnn = nn.GRU(
            self.d_emb + self.d_z + self.d_c,
            self.d_z + self.d_c
        )
        self.decoder_fc = nn.Linear(self.d_z + self.d_c, self.n_vocab)

        # Discriminator
        self.conv3 = nn.Conv2d(1, 100, (3, self.d_emb))
        self.conv4 = nn.Conv2d(1, 100, (4, self.d_emb))
        self.conv5 = nn.Conv2d(1, 100, (5, self.d_emb))
        self.disc_fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(300, self.d_c)
        )

        # Grouping the model's parameters
        self.encoder = nn.ModuleList([
            self.encoder_rnn,
            self.q_mu,
            self.q_logvar
        ])
        self.decoder = nn.ModuleList([
            self.decoder_rnn,
            self.decoder_fc
        ])
        self.vae = nn.ModuleList([
            self.x_emb,
            self.encoder,
            self.decoder
        ])
        self.discriminator = nn.ModuleList([
            self.conv3,
            self.conv4,
            self.conv5,
            self.disc_fc
        ])

    def forward(self, x):
        """Do the VAE forward step with prior c

        :param x: (n_batch, n_len) of longs, input sentence x
        :return: float, kl term component of loss
        :return: float, recon component of loss
        """

        # Input check
        assert_check(x, [-1, self.n_len], torch.long)

        # Entering train mode
        self.train()

        # Encoder: x -> z, kl_loss
        z, kl_loss = self.forward_encoder(x)

        # Code: x -> c
        c = self.sample_c_prior(x.size(0), x.device)

        # Decoder: x, z, c -> recon_loss
        recon_loss = self.forward_decoder(x, z, c)

        # Output check
        assert_check(kl_loss, [], torch.float, x.device)
        assert_check(recon_loss, [], torch.float, kl_loss.device)

        return kl_loss, recon_loss

    def forward_encoder(self, x, do_emb=True):
        """Encoder step, emulating z ~ E(x) = q_E(z|x)

        :param x: (n_batch, n_len) of longs or (n_batch, n_len, d_emb) of
        floats, input sentence x
        :param do_emb: whether do embedding for x or not
        :return: (n_batch, d_z) of floats, sample of latent vector z
        :return: float, kl term component of loss
        """

        # Input check
        if do_emb:
            assert_check(x, [-1, self.n_len], torch.long)
        else:
            assert_check(x, [-1, self.n_len, self.d_emb], torch.float)

        # Emb (n_batch, n_len, d_emb)
        if do_emb:
            x_emb = self.x_emb(x)
        else:
            x_emb = x

        # RNN
        _, h = self.encoder_rnn(x_emb.t(), None)  # (1, n_batch, d_h)

        # Forward to latent
        h = h.squeeze()  # (n_batch, d_h)
        mu, logvar = self.q_mu(h), self.q_logvar(h)  # (n_batch, d_z)

        # Reparameterization trick: z = mu + std * eps; eps ~ N(0, I)
        eps = torch.randn_like(mu)
        z = mu + torch.exp(logvar / 2) * eps

        # KL term loss
        kl_loss = 0.5 * (
                logvar.exp() + mu ** 2 - 1 - logvar
        ).sum(1).mean()  # 0

        # Output check
        assert_check(z, [x.size(0), self.d_z], torch.float, x.device)
        assert_check(kl_loss, [], torch.float, z.device)

        return z, kl_loss

    def forward_decoder(self, x, z, c):
        """Decoder step, emulating x ~ G(z, c)

        :param x: (n_batch, n_len) of longs, input sentence x
        :param z: (n_batch, d_z) of floats, latent vector z
        :param c: (n_batch, d_c) of floats, code c
        :return: float, recon component of loss
        """

        # Input check
        assert_check(x, [-1, self.n_len], torch.long)
        assert_check(z, [-1, self.d_z], torch.float, x.device)
        assert_check(c, [-1, self.d_c], torch.float, z.device)

        # Init
        h_init = torch.cat([
            z.unsqueeze(0),
            c.unsqueeze(0)
        ], 2)  # (1, n_batch, d_z + d_c)

        # Inputs
        x_drop = self.word_dropout(x)  # (n_batch, n_len)
        x_emb = self.x_emb(x_drop.t())  # (n_len, n_batch, d_emb)
        x_emb = torch.cat([
            x_emb,
            h_init.repeat(x_emb.shape[0], 1, 1)
        ], 2)  # (n_len, n_batch, d_emb + d_z + d_c)

        # Rnn step
        outputs, _ = self.decoder_rnn(x_emb,
                                      h_init)  # (n_len, n_batch, d_z + d_c)
        n_len, n_batch, _ = outputs.shape  # (n_len, n_batch)

        # FC to vocab
        y = self.decoder_fc(
            outputs.view(n_len * n_batch, -1)
        ).view(n_len, n_batch, -1)  # (n_len, n_batch, n_vocab)

        # Loss
        recon_loss = F.cross_entropy(
            y.view(-1, y.size(2)),
            F.pad(x.t()[1:], (0, 0, 0, 1), 'constant', self.pad).view(-1)
        )  # 0

        # Output check
        assert_check(recon_loss, [], torch.float, x.device)

        return recon_loss

    def forward_discriminator(self, x, do_emb=True):
        """Discriminator step, emulating c ~ D(x)

        :param x: (n_batch, n_len) of longs or (n_batch, n_len, d_emb) of
        floats, input sentence x
        :param do_emb: whether do embedding for x or not
        :return: (n_batch, d_c) of floats, sample of code c
        """

        # Input check
        if do_emb:
            assert_check(x, [-1, self.n_len], torch.long)
        else:
            assert_check(x, [-1, self.n_len, self.d_emb], torch.float)

        # Emb (n_batch, n_len, d_emb)
        if do_emb:
            x_emb = self.x_emb(x)
        else:
            x_emb = x

        # CNN + FC
        x_emb = x_emb.unsqueeze(1)  # (n_batch, 1, n_len, d_emb)
        x3 = F.relu(self.conv3(x_emb)).squeeze()  # (n_batch, 100, n_len/3)
        x4 = F.relu(self.conv4(x_emb)).squeeze()  # (n_batch, 100, n_len/3)
        x5 = F.relu(self.conv5(x_emb)).squeeze()  # (n_batch, 100, n_len/3)
        x3 = F.max_pool1d(x3, x3.size(2)).squeeze()  # (n_batch, 100)
        x4 = F.max_pool1d(x4, x4.size(2)).squeeze()  # (n_batch, 100)
        x5 = F.max_pool1d(x5, x5.size(2)).squeeze()  # (n_batch, 100)
        x_comb = torch.cat([x3, x4, x5], dim=1)  # (n_batch, 300)
        c = self.disc_fc(x_comb)  # (n_batch, d_c)

        # Output check
        assert_check(c, [x.size(0), self.d_c], torch.float, x.device)

        return c

    def word_dropout(self, x):
        """
        Do word dropout: with prob `self.p_word_dropout`, set the word to
        `self.unk`, as initial Bowman et al. (2014) paper proposed.

        :param x: (n_batch, n_len) of longs, input sentence x
        :return: (n_batch, n_len) of longs, x with drops
        """

        # Input check
        assert_check(x, [-1, self.n_len], torch.long)

        # Apply dropout mask
        mask = x.new_tensor(
            np.random.binomial(
                n=1,
                p=self.p_word_dropout,
                size=tuple(x.shape)
            ),
            dtype=torch.uint8
        )
        x_drop = x.clone()
        x_drop[mask] = self.unk

        # Output check
        assert_check(x_drop, [x.size(0), self.n_len], torch.long, x.device)

        return x_drop

    def sample_z_prior(self, n_batch, device):
        """Sampling z ~ p(z) = N(0, I)

        :param n_batch: number of batches
        :param device: device to run
        :return: (n_batch, d_z) of floats, sample of latent z
        """

        # Input check
        assert isinstance(n_batch, int) and n_batch > 0
        assert isinstance(device, torch.device)

        # Sampling
        z = torch.randn((n_batch, self.d_z), device=device)  # (n_batch, d_z)

        # Output check
        assert_check(z, [n_batch, self.d_z], torch.float, device)

        return z

    def sample_c_prior(self, n_batch, device):
        """Sampling prior, emulating c ~ P(c)

        :param n_batch: number of batches
        :param device: device to run
        :return: (n_batch, d_c) of floats, sample of code c
        """

        # Input check
        assert isinstance(n_batch, int) and n_batch > 0
        assert isinstance(device, torch.device)

        # Sampling
        c = torch.tensor(
            np.random.multinomial(
                n=1,
                pvals=np.ones(self.d_c) / self.d_c,
                size=n_batch
            ),
            dtype=torch.float,
            device=device
        )  # (n_batch, d_c)

        # Output check
        assert_check(c, [n_batch, self.d_c], torch.float, device)

        return c

    def sample_sentence(self, z=None, c=None, temp=1.0):
        """Generating single sentence in eval mode

        :param z: (n_batch, d_z) of floats, latent vector z / None
        :param c: (n_batch, d_c) of floats, code c / None
        :param temp: temperature of softmax
        :param device: device to run
        :return: 1-dim tensor of longs: generated sent word ids
        """

        # Input check
        if z is not None:
            assert_check(z, [-1, self.d_z], torch.float)
        if c is not None:
            assert_check(c, [-1, self.d_c], torch.float)
        assert isinstance(temp, float) and 0 < temp <= 1

        # Enable evaluating mode
        self.eval()

        # Initial values
        device = self.x_emb.weight.device
        w = torch.tensor(self.bos, device=device)  # 0
        if z is None:
            z = self.sample_z_prior(1, device)  # (1, d_z)
        if c is None:
            c = self.sample_c_prior(1, device)  # (1, d_c)
        z, c = z.view(1, 1, -1), c.view(1, 1, -1)  # (1, 1, d_z/d_c)
        h = torch.cat([z, c], dim=2)  # (1, 1, d_z + d_c)

        # Generating cycle, word by word
        outputs = [self.bos]
        for i in range(self.n_len - 1):
            # Init
            x_emb = self.x_emb(w).view(1, 1, -1)  # (1, 1, d_emb)
            x_emb = torch.cat([x_emb, z, c], 2)  # (1, 1, d_emb + d_z + d_c)

            # Step
            o, h = self.decoder_rnn(x_emb, h)  # (1, 1, d_z + d_c)
            y = self.decoder_fc(o).view(-1)  # n_vocab
            y = F.softmax(y / temp, dim=0)  # n_vocab

            # Generating
            w = torch.multinomial(y, 1)[0]  # 0
            outputs.append(w.item())

            # Eos guard
            if outputs[-1] == self.eos:
                break

        # Whole sentence
        x = torch.tensor(outputs, device=device)

        # Back to train
        self.train()

        # Output check
        assert_check(x, [-1], torch.long, device)

        return x

    def sample_soft_embed(self, z=None, c=None, temp=1.0):
        """Generating single soft sample x

        :param z: (n_batch, d_z) of floats, latent vector z / None
        :param c: (n_batch, d_c) of floats, code c / None
        :param temp: temperature of softmax
        :param device: device to run
        :return: (n_len, d_emb) of floats, sampled soft x
        """

        # Input check
        if z is not None:
            assert_check(z, [-1, self.d_z], torch.float)
        if c is not None:
            assert_check(c, [-1, self.d_c], torch.float)
        assert isinstance(temp, float) and 0 < temp <= 1

        # Enable evaluating mode
        self.eval()

        # Initial values
        device = self.x_emb.weight.device
        emb = self.x_emb(torch.tensor(self.bos, device=device))  # d_emb
        if z is None:
            z = self.sample_z_prior(1, device)  # (1, d_z)
        if c is None:
            c = self.sample_c_prior(1, device)  # (1, d_c)
        z, c = z.view(1, 1, -1), c.view(1, 1, -1)  # (1, 1, d_z/d_c)
        h = torch.cat([z, c], dim=2)  # (1, 1, d_z + d_c)

        # Generating cycle, word by word
        outputs = [emb]
        for i in range(self.n_len - 1):
            # Init
            x_emb = emb.view(1, 1, -1)
            x_emb = torch.cat([x_emb, z, c], 2)  # (1, 1, d_emb + d_z + d_c)

            # Step
            o, h = self.decoder_rnn(x_emb, h)
            y = self.decoder_fc(o).view(-1)  # n_vocab
            y = F.softmax(y / temp, dim=0)  # n_vocab

            # Emb
            emb = y.unsqueeze(0) @ self.x_emb.weight  # d_emb
            outputs.append(emb)

        # Whole sequence
        x = torch.cat(outputs, dim=0)  # (n_len, d_emb)

        # Back to train
        self.train()

        # Output check
        assert_check(x, [-1, self.d_emb], torch.float, device)

        return x
