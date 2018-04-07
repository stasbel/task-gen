import json
from functools import wraps

from hyperdash import Experiment
from tqdm import tqdm


__all__ = ['monitor', 'monitor_apply_gen']


def _api_key_getter():
    return json.load(open('hd.json', 'r'))['api_key']


def _new_exp(name, supress_output):
    exp = Experiment(name, capture_io=False, api_key_getter=_api_key_getter)

    # SUPER-hacky, but it's work (needed to supress hd output)
    if supress_output:
        exp._hd.out_buf.write = lambda _: _

    return exp


class _Log:
    def __init__(self, exp, log_step, log_total):
        self.exp = exp
        self.log_step = log_step
        self.log_total = log_total

        self.param, self.metric = self.exp.param, self.exp.metric

    def is_step(self, i):
        return i % self.log_step == 0 \
               or (self.log_total is not None and i == self.log_total - 1)

    def istep(self, i):
        if self.is_step(i):
            self.metric('i_step', i)

    def imetric(self, i, *args, **kwargs):
        self.istep(i)
        self.metric(*args, **kwargs)


class monitor:
    def __init__(self, name, * ,supress_output=True,
                 log_step=1, log_total=None):
        self.name = name
        self.supress_output = supress_output
        self.log_step = log_step
        self.log_total = log_total

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            exp = _new_exp(self.name, self.supress_output)
            kwargs['log'] = _Log(exp, self.log_step, self.log_total)
            result = func(*args, **kwargs)
            exp.end()
            return result

        return wrapper


def monitor_apply_gen(f, X, monitor=None, do_tqdm=False):
    if hasattr(X, '__len__') and monitor is not None:
        monitor.log_total = len(X)

    if do_tqdm:
        X = tqdm(X)

    if monitor is None:
        def apply():
            return (f(x) for x in X)
    else:
        @monitor
        def apply(*, log):
            for i, x in enumerate(X):
                yield f(x)
                log.istep(i)

    return apply()
