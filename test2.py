import sys
sys.path.insert(0,'.')
print('before', flush=True)
from datasets import load_dataset
print('after', flush=True)