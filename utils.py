from IPython.display import display

import pandas as pd


def examine_df(df, k=4):
    print(f'shape={df.shape}')
    k = min(k, len(df))
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        display(pd.concat([
            df.head(k),
            df.tail(k),
            pd.DataFrame({n: [str(t)] for n, t in df.dtypes.iteritems()}, 
                         columns=df.dtypes.index.tolist(), index=['type']), 
            df.describe(include='all')
        ], axis=0, keys=['head', 'tail', 'meta', 'math']))
