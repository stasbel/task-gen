import json

from tqdm import tqdm
from functools import wraps
from hyperdash import Experiment


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


def monitor(name, supress_output=True, log_step=1, log_max=None):
    def decorate(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            exp = _new_exp(name, supress_output)
            kwargs['log'] = _Log(exp, log_step, log_max)
            result = func(*args, **kwargs)
            exp.end()
            return result
        return wrapper
    return decorate


def monitor_run_gen(f, X, monitor):
    if monitor is None:
        def run():
            return (f(x) for x in tqdm(X))
    else:
        @monitor
        def run(*, log):
            for i, x in enumerate(tqdm(X)):
                yield f(x)
                log.istep(i)

    return run()
