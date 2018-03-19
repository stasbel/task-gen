from functools import wraps
from hyperdash import Experiment


def hyperdash(name):
    def decorate(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            exp = Experiment(name, capture_io=False,
                             api_key_getter=lambda: 'mB3wTK1XXivrCFp4HpnX/KDUFT/0az3+W8BhLSF+Vdg=')
            kwargs['exp'] = exp
            result = func(*args, **kwargs)
            exp.end()
            return result
        return wrapper
    return decorate
