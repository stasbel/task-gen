from functools import wraps
from hyperdash import Experiment


def hyperdash(name):
    def decorate(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            exp = Experiment(name, capture_io=False)
            kwargs['exp'] = exp
            result = func(*args, **kwargs)
            exp.end()
            return result
        return wrapper
    return decorate
