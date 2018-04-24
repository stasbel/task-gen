import numpy as np
import torch

print(f'cuda available={torch.cuda.is_available()}')
print(f'cuda version={torch.version.cuda}')
print(f'cuddn version={torch.backends.cudnn.version()}')

print('kek')

print(np.zeros(10))
