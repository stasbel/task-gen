import abc

from torchtext import data, datasets
from torchtext.vocab import GloVe

__all__ = ['SSTCorpus']


class Corpus(abc.ABC):
    @abc.abstractmethod
    def unlabeled(self, split, batch_size):
        pass

    @abc.abstractmethod
    def labeled(self, split, batch_size):
        pass


class SSTCorpus(Corpus):
    def __init__(self, n_batch=32, n_len=15, n_vocab=10000, d_emb=50):
        self.n_batch = n_batch
        self.n_len = n_len
        self.n_vocab = n_vocab
        self.d_emb = d_emb

        self._load_data()

    def unlabeled(self, split='train', n_batch=None):
        for batch in self._make_b_iter(split, n_batch):
            yield batch.text

    def labeled(self, split='train', n_batch=None):
        for batch in self._make_b_iter(split, n_batch):
            yield batch.text, batch.label

    def _load_data(self):
        self.text = data.Field(
            init_token='<start>',
            eos_token='<eos>',
            fix_length=self.n_len,
            lower=True,
            tokenize='spacy',
            batch_first=True
        )
        self.label = data.Field(
            sequential=False,
            unk_token=None,
            batch_first=True
        )

        def filter_pred(e):
            return len(e.text) <= self.n_len and e.label != 'neutral'

        self.train, self.val, self.test = datasets.SST.splits(
            self.text,
            self.label,
            fine_grained=False,
            train_subtrees=False,
            filter_pred=filter_pred
        )

        self.text.build_vocab(self.train,
                              max_size=self.n_vocab,
                              vectors=GloVe('6B', dim=self.d_emb))
        self.label.build_vocab(self.train)

    def _make_b_iter(self, split, n_batch):
        return data.BucketIterator(
            self._choose_split(split),
            n_batch or self.n_batch,
            repeat=False,
            train=(split == 'train'),
            device=-1
        )

    def _choose_split(self, split):
        if split == 'train':
            return self.train
        elif split == 'val':
            return self.val
        elif split == 'test':
            return self.test
        else:
            raise ValueError('Invalid split, should be one of the '
                             '(\'train\', \'val\', \'test\')')
