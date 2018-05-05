import abc

from torchtext import data, datasets
from torchtext.vocab import GloVe

__all__ = ['SSTCorpus']


class Corpus(abc.ABC):
    """Handle both preprocessing and batching, also store vocabs"""

    @abc.abstractmethod
    def vocab(self, name):
        """Gets vocab instance by var name"""
        pass

    @abc.abstractmethod
    def batcher(self, mode, split, **kwargs):
        """Make batcher generator"""
        pass

    @abc.abstractmethod
    def reverse(self, example, name):
        """Reverse given example to human-readable form"""
        pass


class SSTCorpus(Corpus):
    def __init__(self, **kwargs):
        self.n_batch = kwargs['n_batch']
        self.n_len = kwargs['n_len']
        self.n_vocab = kwargs['n_vocab']
        self.d_emb = kwargs['d_emb']
        self.device = kwargs['device']

        self._load_data()

    def vocab(self, name):
        vocab = getattr(self, name).vocab
        # vocab.vectors = vocab.vectors.to(self.device)
        return vocab

    def batcher(self, mode, split, *, n_batch=None, device=None, n_iter=None):
        n_batch = n_batch or self.n_batch
        device = device or self.device
        split_instance = self._choose_split(split)
        n_iter = n_iter or len(split_instance)
        b_iter = data.BucketIterator(
            split_instance,
            n_batch,
            train=(split == 'train'),
            device=device  # also explicitly due to torchtext bug
        )

        if mode == 'unlabeled':
            i_iter = 0
            for batch in b_iter:
                if batch.batch_size == n_batch:
                    i_iter += 1
                    yield batch.text.to(device)
                if i_iter == n_iter:
                    break
        elif mode == 'labeled':
            i_iter = 0
            for batch in b_iter:
                if batch.batch_size == n_batch:
                    i_iter += 1
                    yield batch.text.to(device), batch.label.to(device)
                if i_iter == n_iter:
                    break
        else:
            raise ValueError(
                "Invalid mode, should be one of the ('unlabeled', 'labeled')"
            )

    def reverse(self, example, name='x'):
        # sent = ' '.join(corpus.vocab('x').itos[w] \
        #                 for w in model.sample_sentence(device=device) \
        #                 if w > 3)

        if name == 'x':
            return self.x.reverse(example.unsqueeze(0))[0]
        elif name == 'y':
            return self.y.itos[example]
        else:
            raise ValueError(
                "Invalid name, should be one of the ('x', 'y')"
            )

    def _load_data(self):
        self.x = data.ReversibleField(
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
            self.x,
            self.y,
            fine_grained=False,
            train_subtrees=False,
            filter_pred=filter_pred
        )

        self.x.build_vocab(self.train,
                           max_size=self.n_vocab,
                           vectors=GloVe('6B', dim=self.d_emb))
        self.y.build_vocab(self.train)

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
