import abc
import re
from typing import List

import entity
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from utils import monitor_apply_gen


class RawPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, monitor=None, do_tqdm=False):
        self.monitor = monitor
        self.do_tqdm = do_tqdm

    def fit(self, X):
        return self

    def transform(self, X):
        data = {'text': [], 'tags': []}

        def f(x):
            if 'text' not in x:
                return

            text = entity.Text(x['text'])
            if not text.is_valid:
                return

            lin_tags = entity.linearize_tags(x.get('tags', []))

            data['text'].append(text.norm)
            data['tags'].append(lin_tags)

        for _ in monitor_apply_gen(f, X, self.monitor, self.do_tqdm): pass

        return pd.DataFrame.from_dict(data)[['text', 'tags']].fillna('NaN')
