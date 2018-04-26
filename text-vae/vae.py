import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch import nn
from torch.autograd import Variable
from torchvision import transforms
from tqdm import tqdm

# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
torch.cuda.set_device(2)


class Encoder(nn.Module):
    def __init__(self, d_in, h, d_out):
        super().__init__()

        self.linear1 = nn.Linear(d_in, h)
        self.linear2 = nn.Linear(h, d_out)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        return F.relu(self.linear2(x))


class Decoder(nn.Module):
    def __init__(self, d_in, h, d_out):
        super().__init__()

        self.linear1 = nn.Linear(d_in, h)
        self.linear2 = nn.Linear(h, d_out)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        return F.relu(self.linear2(x))


class VAE(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.enc_mu = nn.Linear(100, 8)
        self.enc_log_sigma = nn.Linear(100, 8)

    def forward(self, x):
        h = self.encoder(x)
        z = self._sample_z(h)
        return self.decoder(z)

    def _sample_z(self, h):
        mu, sigma = self.enc_mu(h), torch.exp(self.enc_log_sigma(h))

        std = torch.from_numpy(
            np.random.normal(0, 1, size=sigma.size())
        ).float().cuda()

        self.z_mu, self.z_sigma = mu, sigma

        return mu + sigma * Variable(std, requires_grad=False)


def latent_loss(z_mu, z_sigma):
    mu_sq, sigma_sq = z_mu ** 2, z_sigma ** 2
    return 0.5 * torch.mean(mu_sq + sigma_sq - torch.log(sigma_sq) - 1)


if __name__ == '__main__':
    batch_size = 32
    d_input = 28 * 28
    n_epoch = 5

    transform = transforms.Compose([transforms.ToTensor()])
    mnist = torchvision.datasets.MNIST('./',
                                       download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(mnist, batch_size=batch_size,
                                             shuffle=True, num_workers=2,
                                             drop_last=True)

    encoder = Encoder(d_input, 100, 100)
    decoder = Decoder(8, 100, d_input)
    model = VAE(encoder, decoder)
    model.cuda()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in tqdm(range(n_epoch)):
        l1_loss, l2_loss, total_loss = 0, 0, 0
        for i, (x, _) in enumerate(dataloader):
            # data
            x = Variable(x.resize_(batch_size, d_input).cuda())

            # forward pass
            xh = model(x)

            # loss
            l1, l2 = criterion(xh, x), latent_loss(model.z_mu, model.z_sigma)
            loss = l1 + l2

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # metrics
            l1_loss += l1.data[0]
            l2_loss += l2.data[0]
            total_loss += loss.data[0]

        l1_loss /= (len(dataloader) * batch_size)
        l2_loss /= (len(dataloader) * batch_size)
        total_loss /= (len(dataloader) * batch_size)
        print(f'\nepoch={epoch} l1={l1_loss} l2={l2_loss} loss={total_loss}')
