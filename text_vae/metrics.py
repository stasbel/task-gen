import numpy as np

from functools import lru_cache
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


class Evaluator:
    def __init__(self, corpus,
                 blue_span=(2, 5), blue_smooth='epsilon'):
        self.corpus = corpus

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

    def bleu(self, model, hypot_size, mode='val'):
        """Calculating similarity metric, higher is better"""
        references = self._get_reference(mode)
        hypotheses = [
            word_tokenize(sent)
            for sent in self.corpus.reverse(
                model.sample_sentence(hypot_size)[-1]
            )
        ]

        result = {}
        for i, w in self.blue_weights:
            result[f'{i}-gram'] = np.mean([
                sentence_bleu(references, h,
                              weights=w, smoothing_function=self.blue_smooth)
                for h in hypotheses
            ])
        return result

    def self_bleu(self, model, hypot_size):
        """Calculating diversity metric, lower is better"""
        hypotheses = [
            word_tokenize(sent)
            for sent in self.corpus.reverse(
                model.sample_sentence(hypot_size)[-1]
            )
        ]

        result = {}
        for i, w in self.blue_weights:
            result[f'{i}-gram'] = np.mean([
                sentence_bleu(hypotheses[:j] + hypotheses[j + 1:],
                              hypotheses[j],
                              weights=w, smoothing_function=self.blue_smooth)
                for j in range(len(hypotheses))
            ])
        return result

    @lru_cache(maxsize=None)
    def _get_reference(self, mode):
        return [word_tokenize(d['text']) for d in self.corpus.raw(mode)]
