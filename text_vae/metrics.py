from functools import lru_cache

import torch
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


class Evaluator:
    def __init__(self, corpus, n_ref, sample_params=None,
                 blue_span=(2, 5), blue_smooth='epsilon'):
        self.corpus = corpus
        self.n_ref = n_ref
        self.sample_params = sample_params or {}

        # BLEU
        self.blue_weights = [
            (i, np.array([1 / i] * i + [0] * (blue_span[1] - i)))
            for i in range(blue_span[0], blue_span[1] + 1)
        ]
        if blue_smooth == 'epsilon':
            # Adds epsilon to zero counts
            self.blue_smooth = SmoothingFunction().method1
        else:
            self.blue_smooth = SmoothingFunction().method0

        # Preload some modes, it may require some time...
        for mode in ('train', 'val', 'test'):
            self._get_reference(mode)

    def bleu(self, model, n_hypot, split):
        """Calculating similarity metric, higher is better"""
        references = self._get_reference(split)
        hypotheses = self._get_hypotheses(model, n_hypot)

        result = {}
        for i, w in self.blue_weights:
            result[f'{i}-gram'] = np.mean([
                sentence_bleu(references, h,
                              weights=w, smoothing_function=self.blue_smooth)
                for h in hypotheses
            ])
        return result

    def self_bleu(self, model=None, n_hypot=None, split=None):
        """Calculating diversity metric, lower is better"""

        if model is not None and n_hypot is not None:
            hypotheses = self._get_hypotheses(model, n_hypot or self.n_ref)
        else:
            hypotheses = self._get_reference(split)

        result = {}
        for i, w in self.blue_weights:
            result[f'{i}-gram'] = np.mean([
                sentence_bleu(hypotheses[:j] + hypotheses[j + 1:],
                              hypotheses[j],
                              weights=w, smoothing_function=self.blue_smooth)
                for j in range(len(hypotheses))
            ])
        return result

    def perplexity(self, model, split):
        ppl = []
        batcher = self.corpus.batcher(split, 'unlabeled')
        for x in batcher:
            ppl.append(model.perplexity(x, use_c_prior=True))
        return torch.stack(ppl).mean().item()

    @lru_cache(maxsize=None)
    def _get_reference(self, split):
        batcher = self.corpus.batcher(split, 'unlabeled',
                                      n_batch=1, device=torch.device('cpu'))

        vocab = self.corpus.vocab('x')
        result = []
        for x in batcher:
            if len(result) == self.n_ref:
                break

            result.append([vocab.itos[i] for i in x[0]
                           if i != vocab.stoi['<pad>']])

        return result

    def _get_hypotheses(self, model, n_hypot):
        vocab = self.corpus.vocab('x')
        hypotheses = [
            [vocab.itos[i] for i in sent if i != vocab.stoi['<pad>']]
            for sent in model.sample_sentence(n_hypot,
                                              **self.sample_params)[2]
        ]
        return hypotheses
