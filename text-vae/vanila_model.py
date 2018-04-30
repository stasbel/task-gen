from itertools import chain

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class RnnVae(nn.Module):
    def __init__(
            self,
            args,
            X_vocab
    ):
        super().__init__()

        self.unk = X_vocab.stoi['<unk>']
        self.pad = X_vocab.stoi['<pad>']
        self.bos = X_vocab.stoi['<bos>']
        self.eos = X_vocab.stoi['<eos>']

        self.n_len = args.n_len
        self.n_vocab = len(X_vocab)

        self.d_h = args.d_h
        self.d_z = args.d_z
        self.d_c = args.d_c

        self.p_word_dropout = args.p_word_dropout

        """
        Word embeddings layer
        """
        if X_vocab is None:
            self.d_emb = self.d_h
            self.X_emb = nn.Embedding(self.n_vocab, self.d_emb, self.pad)
        else:
            self.d_emb = X_vocab.vectors.size(1)
            self.X_emb = nn.Embedding(self.n_vocab, self.d_emb, self.pad)

            # Set pretrained embeddings
            self.X_emb.weight.data.copy_(X_vocab.vectors)

            if args.freeze_embeddings:
                self.X_emb.weight.requires_grad = False

        """
        Encoder is GRU with FC layers connected to last hidden unit
        """
        self.encoder = nn.GRU(self.d_emb, self.d_h)
        self.q_mu = nn.Linear(self.d_h, self.d_z)
        self.q_logvar = nn.Linear(self.d_h, self.d_z)

        """
        Decoder is GRU with `z` and `c` appended at its inputs
        """
        self.decoder = nn.GRU(
            self.d_emb + self.d_z + self.d_c,
            self.d_z + self.d_c
        )
        self.decoder_fc = nn.Linear(self.d_z + self.d_c, self.n_vocab)

        """
        Discriminator is CNN as in Kim, 2014
        """
        self.conv3 = nn.Conv2d(1, 100, (3, self.d_emb))
        self.conv4 = nn.Conv2d(1, 100, (4, self.d_emb))
        self.conv5 = nn.Conv2d(1, 100, (5, self.d_emb))

        self.disc_fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(300, 2)
        )

        self.discriminator = nn.ModuleList([
            self.conv3, self.conv4, self.conv5, self.disc_fc
        ])

        """
        Grouping the model's parameters: separating encoder, decoder, 
        and discriminator
        """
        self.encoder_params = chain(
            self.encoder.parameters(), self.q_mu.parameters(),
            self.q_logvar.parameters()
        )

        self.decoder_params = chain(
            self.decoder.parameters(), self.decoder_fc.parameters()
        )

        self.vae_params = chain(
            self.X_emb.parameters(), self.encoder_params,
            self.decoder_params
        )
        self.vae_params = filter(lambda p: p.requires_grad, self.vae_params)

        self.discriminator_params = filter(lambda p: p.requires_grad,
                                           self.discriminator.parameters())

    def forward_encoder(self, x):
        """Encoder step, emulating z ~ E(x) = q_E(z|x)

        :param x: (n_batch, n_len) of longs, input sentence x
        :return: (n_batch, d_z) of floats, sample of latent vector z
        """

        # Input check
        assert len(x.shape) == 2 and x.shape[1] == self.n_len
        assert x.dtype == torch.long

        # Embedding + RNN
        x_emb = self.X_emb(x.transpose(0, 1))  # (n_len, n_batch, d_emb)
        _, h = self.encoder(x_emb, None)  # (1, n_batch, d_h)

        # Forward to latent
        h = h.view(-1, self.d_h)  # (n_batch, d_h)
        mu, logvar = self.q_mu(h), self.q_logvar(h)  # (n_batch, d_z)

        # Reparameterization trick: z = mu + std * eps; eps ~ N(0, I)
        eps = torch.randn_like(mu)
        z = mu + torch.exp(logvar / 2) * eps

        # Output check
        assert len(z.shape) == 2 and z.shape[1] == self.d_z
        assert z.dtype == torch.float
        assert x.shape[0] == z.shape[0]
        assert x.device == z.device

        return z

    def sample_z_prior(self, n_batch):
        """
        Sample z ~ p(z) = N(0, I)
        """
        z = Variable(torch.randn(n_batch, self.d_z))
        z = z.cuda() if self.gpu else z
        return z

    def sample_c_prior(self, x):
        """Sampling prior, emulating c ~ P(c)

        :param x: (n_batch, n_len) of longs, input sentence x
        :return: (n_batch, d_c) of floats, sample of code c
        """

        # Input check
        assert len(x.shape) == 2 and x.shape[1] == self.n_len
        assert x.dtype == torch.long

        # Sampling
        c = x.new_tensor(
            np.random.multinomial(
                n=1,
                pvals=np.ones(self.d_c) / self.d_c,
                size=x.shape[0]
            ),
            dtype=torch.float
        )

        # Output check
        assert len(c.shape) == 2 and c.shape[1] == self.d_c
        assert c.dtype == torch.float
        assert x.shape[0] == c.shape[0]
        assert x.device == c.device

        return c

    def forward_decoder(self, x, z, c):
        """Decoder step, emulating ???

        :param x: (n_batch, n_len) of longs, input sentence x
        :param z: (n_batch, d_z) of floats, latent vector z
        :param c: (n_batch, d_c) of floats, code c
        :return:
        """

        # Input check
        assert len(x.shape) == 2 and x.shape[1] == self.n_len
        assert x.dtype == torch.long
        assert len(z.shape) == 2 and z.shape[1] == self.d_z
        assert z.dtype == torch.float
        assert x.shape[0] == z.shape[0]
        assert x.device == z.device
        assert len(c.shape) == 2 and c.shape[1] == self.d_c
        assert c.dtype == torch.float
        assert x.shape[0] == c.shape[0]
        assert x.device == c.device

        print('KEK')

        x_drop = self.word_dropout(x)

        h_init = torch.cat(
            [z.unsqueeze(0), c.unsqueeze(0)], 2
        )  # (1, n_batch, d_z + d_c)
        x_emb = self.X_emb(x_drop.transpose(0, 1))  # (n_len, n_batch, d_emb)


        assert False, 'OK'

        # Forward
        seq_len = x_drop.size(0)

        # 1 x n_batch x (z_dim+c_dim)
        init_h = torch.cat([z.unsqueeze(0), c.unsqueeze(0)], dim=2)

        inputs_emb = self.X_emb(x_drop)  # seq_len x n_batch x emb_dim
        inputs_emb = torch.cat([inputs_emb, init_h.repeat(seq_len, 1, 1)], 2)

        outputs, _ = self.decoder(inputs_emb, init_h)
        seq_len, n_batch, _ = outputs.size()

        outputs = outputs.view(seq_len * n_batch, -1)
        y = self.decoder_fc(outputs)
        y = y.view(seq_len, n_batch, self.n_vocab)

        return y

    def forward_discriminator(self, inputs):
        """
        Inputs is batch of sentences: n_batch x seq_len
        """
        inputs = self.X_emb(inputs)
        return self.forward_discriminator_embed(inputs)

    def forward_discriminator_embed(self, inputs):
        """
        Inputs must be embeddings: n_batch x seq_len x emb_dim
        """
        inputs = inputs.unsqueeze(1)  # n_batch x 1 x seq_len x emb_dim

        x3 = F.relu(self.conv3(inputs)).squeeze()
        x4 = F.relu(self.conv4(inputs)).squeeze()
        x5 = F.relu(self.conv5(inputs)).squeeze()

        # Max-over-time-pool
        x3 = F.max_pool1d(x3, x3.size(2)).squeeze()
        x4 = F.max_pool1d(x4, x4.size(2)).squeeze()
        x5 = F.max_pool1d(x5, x5.size(2)).squeeze()

        x = torch.cat([x3, x4, x5], dim=1)

        y = self.disc_fc(x)

        return y

    def forward(self, x, use_c_prior=True):
        """

        :param x: n_batch x n_len of longs
        :type x: torch.Tensor
        :param use_c_prior:
        :return:
        """

        # Input check
        # ...

        self.train()

        # n_batch = x.size(0)
        # x = x.transpose(1, 0)  # n_len x n_batch

        # # sentence: '<start> I want to fly <eos>'
        # # enc_inputs: '<start> I want to fly <eos>'
        # # dec_inputs: '<start> I want to fly <eos>'
        # # dec_targets: 'I want to fly <eos> <pad>'
        # enc_inputs = x
        # dec_inputs = x
        # pad_words = torch.tensor([self.pad], dtype=torch.long) \
        #     .repeat(1, n_batch).to(self.device)
        # dec_targets = torch.cat([x[1:], pad_words], dim=0)

        # Encoder: x -> z
        z = self.forward_encoder(x)

        # Code
        c = self.sample_c_prior(x) if use_c_prior \
            else self.forward_discriminator(x)

        # Decoder: sentence -> y
        y = self.forward_decoder(x, z, c)

        assert False, 'OK'

        recon_loss = F.cross_entropy(
            y.view(-1, self.n_vocab), dec_targets.view(-1), size_average=True
        )
        kl_loss = torch.mean(
            0.5 * torch.sum(torch.exp(logvar) + mu ** 2 - 1 - logvar, 1))

        return recon_loss, kl_loss

    def generate_sentences(self, n_batch):
        """
        Generate sentences of (n_batch x max_sent_len)
        """
        samples = []

        for _ in range(n_batch):
            z = self.sample_z_prior(1)
            c = self.sample_c_prior(1)
            samples.append(self.sample_sentence(z, c, raw=True))

        X_gen = torch.cat(samples, dim=0)

        return X_gen

    def sample_sentence(self, z, c, raw=False, temp=1):
        """
        Sample single sentence from p(x|z,c) according to given temperature.
        `raw = True` means this returns sentence as in dataset which is useful
        to train discriminator. `False` means that this will return list of
        `word_idx` which is useful for evaluation.
        """
        self.eval()

        word = torch.LongTensor([self.bos])
        word = word.cuda() if self.gpu else word
        word = Variable(word)  # '<start>'

        z, c = z.view(1, 1, -1), c.view(1, 1, -1)

        h = torch.cat([z, c], dim=2)

        if not isinstance(h, Variable):
            h = Variable(h)

        outputs = []

        if raw:
            outputs.append(self.bos)

        for i in range(self.n_len):
            emb = self.X_emb(word).view(1, 1, -1)
            emb = torch.cat([emb, z, c], 2)

            output, h = self.decoder(emb, h)
            y = self.decoder_fc(output).view(-1)
            y = F.softmax(y / temp, dim=0)

            idx = torch.multinomial(y)

            word = Variable(torch.LongTensor([int(idx)]))
            word = word.cuda() if self.gpu else word

            idx = int(idx)

            if not raw and idx == self.eos:
                break

            outputs.append(idx)

        # Back to default state: train
        self.train()

        if raw:
            outputs = Variable(torch.LongTensor(outputs)).unsqueeze(0)
            return outputs.cuda() if self.gpu else outputs
        else:
            return outputs

    def generate_soft_embed(self, n_batch, temp=1):
        """
        Generate soft embeddings of (n_batch x emb_dim) along with target z
        and c for each row (n_batch x {z_dim, c_dim})
        """
        samples = []
        targets_c = []
        targets_z = []

        for _ in range(n_batch):
            z = self.sample_z_prior(1)
            c = self.sample_c_prior(1)

            samples.append(self.sample_soft_embed(z, c, temp=1))
            targets_z.append(z)
            targets_c.append(c)

        X_gen = torch.cat(samples, dim=0)
        targets_z = torch.cat(targets_z, dim=0)
        _, targets_c = torch.cat(targets_c, dim=0).max(dim=1)

        return X_gen, targets_z, targets_c

    def sample_soft_embed(self, z, c, temp=1):
        """
        Sample single soft embedded sentence from p(x|z,c) and temperature.
        Soft embeddings are calculated as weighted average of word_emb
        according to p(x|z,c).
        """
        self.eval()

        z, c = z.view(1, 1, -1), c.view(1, 1, -1)

        word = torch.LongTensor([self.bos])
        word = word.cuda() if self.gpu else word
        word = Variable(word)  # '<start>'
        emb = self.X_emb(word).view(1, 1, -1)
        emb = torch.cat([emb, z, c], 2)

        h = torch.cat([z, c], dim=2)

        if not isinstance(h, Variable):
            h = Variable(h)

        outputs = [self.X_emb(word).view(1, -1)]

        for i in range(self.n_len):
            output, h = self.decoder(emb, h)
            o = self.decoder_fc(output).view(-1)

            # Sample softmax with temperature
            y = F.softmax(o / temp, dim=0)

            # Take expectation of embedding given output prob -> soft embedding
            # <y, w> = 1 x n_vocab * n_vocab x emb_dim
            emb = y.unsqueeze(0) @ self.X_emb.weight
            emb = emb.view(1, 1, -1)

            # Save resulting soft embedding
            outputs.append(emb.view(1, -1))

            # Append with z and c for the next input
            emb = torch.cat([emb, z, c], 2)

        # 1 x 16 x emb_dim
        outputs = torch.cat(outputs, dim=0).unsqueeze(0)

        # Back to default state: train
        self.train()

        return outputs.cuda() if self.gpu else outputs

    def word_dropout(self, x):
        """
        Do word dropout: with prob `self.p_word_dropout`, set the word to
        `self.unk`, as initial Bowman et al. (2014) paper told.

        :param x: (n_batch, n_len) of longs, input sentence x
        :return: (n_batch, n_len) of longs, x with drops
        """

        # Input check
        assert len(x.shape) == 2 and x.shape[1] == self.n_len
        assert x.dtype == torch.long

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
        assert len(x_drop.shape) == 2 and x_drop.shape[1] == self.n_len
        assert x_drop.dtype == torch.long
        assert x.shape[0] == x_drop.shape[0]
        assert x.device == x_drop.device

        return x_drop
