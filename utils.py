import json

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


def monitor(name, supress_output=True):
    def decorate(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            exp = _new_exp(name, supress_output)
            kwargs['exp'] = exp
            result = func(*args, **kwargs)
            exp.end()
            return result
        return wrapper
    return decorate
