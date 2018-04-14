import abc
from collections import Counter
from typing import Sequence, List

import nltk
from entity import linearize_tags
from lazy import lazy
from rake_nltk import Rake
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm
from utils import monitor_apply_gen


class StagsExtractor(abc.ABC):
    @abc.abstractmethod
    def __call__(self, text: str) -> List[str]:
        """Extract list of stags from `text`."""
        pass


class RakeStagsExtractor(StagsExtractor):
    def __init__(self, score_lb=2):
        self.score_lb = score_lb
        self.r = Rake()

    def __call__(self, text):
        self.r.extract_keywords_from_text(text)
        return [t for s, t in self.r.get_ranked_phrases_with_scores()
                if s >= self.score_lb]


class StagsCollector(BaseEstimator, TransformerMixin):
    def __init__(self, total_up, *, monitor=None, do_tqdm=False,
                 extractor=RakeStagsExtractor()):
        self.total_up = total_up
        self.monitor = monitor
        self.do_tdqm = do_tqdm
        self.extractor = extractor

    def fit(self, X: Sequence[str]) -> 'StagsCollector':
        stags = []
        cum_stags = Counter()

        def f(s):
            s_stags = (linearize_tags(self.extractor(s)) or '').split()
            # s_stags = [' '.join(i_stags.split('-')) for i_stags in s_stags]
            stags.append(s_stags)
            cum_stags.update(Counter(s_stags))

        for _ in monitor_apply_gen(f, X, self.monitor, self.do_tdqm): pass

        self._stags = stags
        self._top_stags = [t for t, _ in cum_stags.most_common(self.total_up)]
        return self

    def transform(self, _: Sequence[str]) -> List[str]:
        good_tags = set(self._top_stags)
        if self.do_tdqm:
            stags_it = tqdm(self._stags)
        else:
            stags_it = self._stags

        return [' '.join(tag for tag in i_stags if tag in good_tags)
                for i_stags in stags_it]
