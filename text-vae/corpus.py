import abc

from torchtext import data, datasets
from torchtext.vocab import GloVe

__all__ = ['SSTCorpus']


class Corpus(abc.ABC):
    """Handle both preprocessing and batching, also store vocabs"""

    @abc.abstractmethod
    def vocab(self, name):
        pass

    @abc.abstractmethod
    def make_batcher(self, mode, split, batch_size):
        pass


class SSTCorpus(Corpus):
    def __init__(self, args, device):
        self.n_batch = args.n_batch
        self.n_len = args.n_len
        self.n_vocab = args.n_vocab
        self.d_emb = args.d_emb
        self.device = device

        self._load_data()

    def vocab(self, name):
        return getattr(self, name).vocab

    def make_batcher(self, mode, split, n_batch=None, device=None):
        n_batch = n_batch or self.n_batch
        device = device or self.device
        b_iter = self._make_b_iter(split, n_batch, device)

        if mode == 'unlabeled':
            for batch in b_iter:
                if batch.batch_size == n_batch:
                    yield batch.text
        elif mode == 'labeled':
            for batch in b_iter:
                if batch.batch_size == n_batch:
                    yield batch.text, batch.label
        else:
            raise ValueError(
                "Invalid mode, should be one of the ('unlabeled', 'labeled')"
            )

    def _load_data(self):
        self.X = data.ReversibleField(
            init_token='<bos>',
            eos_token='<eos>',
            fix_length=self.n_len,
            lower=True,
            # commented to enable reversiblity
            # tokenize='spacy',
            unk_token='<unk>',
            batch_first=True
        )
        self.y = data.ReversibleField(
            sequential=False,
            unk_token=None,
            batch_first=True
        )

        def filter_pred(e):
            return len(e.text) <= self.n_len and e.label != 'neutral'

        self.train, self.val, self.test = datasets.SST.splits(
            self.X,
            self.y,
            fine_grained=False,
            train_subtrees=False,
            filter_pred=filter_pred
        )

        self.X.build_vocab(self.train,
                           max_size=self.n_vocab,
                           vectors=GloVe('6B', dim=self.d_emb))
        self.y.build_vocab(self.train)

    def _make_b_iter(self, split, n_batch, device):
        return data.BucketIterator(
            self._choose_split(split),
            n_batch,
            train=(split == 'train'),
            device=device
        )

    def _choose_split(self, split):
        if split == 'train':
            return self.train
        elif split == 'val':
            return self.val
        elif split == 'test':
            return self.test
        else:
            raise ValueError(
                "Invalid split, should be one of the ('train', 'val', 'test')"
            )
