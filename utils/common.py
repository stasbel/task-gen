import re
import pandas as pd

from IPython.display import display


__all__ = ['examine_df']


def _calc_stats(df, k):
    return [
        (df.head(k), 'head'),
        (df.tail(k), 'tail'),
        (pd.DataFrame({n: [str(t), df[n].isnull().any()] for n, t in df.dtypes.iteritems()}, 
                     columns=df.dtypes.index.tolist(), index=['type', 'has_null']), 'meta'),
        (df.describe(include='all'), 'math')
    ]


def examine_df(df, *, k=5, exclude=None):
    print(f'shape={df.shape}')
    k = min(k, len(df))
    
    if isinstance(exclude, list):
        exclude = set(exclude)
    elif isinstance(exclude, str):
        exclude = set(word for word in re.split('[^a-zA-Z]', exclude.strip().lower()) if len(word))
    else:
        exclude = set()
    
    stats, keys = [], []
    for stat, key in _calc_stats(df, k):
        if key not in exclude:
            stats.append(stat)
            keys.append(key)

    df = pd.concat(stats, axis=0, keys=keys)
    df.columns = pd.MultiIndex.from_tuples(list(zip(df.columns, range(len(df.columns)))), 
                                           names=['name', 'pos'])

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        display(df)
