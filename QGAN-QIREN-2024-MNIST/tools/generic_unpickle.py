import pickletools

with open('c:/Users/Adminb/OneDrive/Documents/Projects/qgan/QINR-QGAN/QGAN-QIREN-2024-MNIST/results_analysis/relaxed_structures.pkl', 'rb') as f:
    data = f.read()

# We will yield all the strings sequentially
mg_mn_counts = []

# Let's just do a naive regex search over the binary data! It's a text protocol!
# But it's binary data so we can just look for instances.
import re
# PyMatgen dumps as dicts sometimes, or maybe we can just find 'Mg' and 'Mn'

# Let's unpickle using a completely generic unpickler
import pickle
import sys

class IgnorantUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Return a dynamically created class
        class Dummy:
            def __init__(self, *args, **kwargs): pass
            def __setstate__(self, state):
                self.__dict__.update(state)
        return Dummy

with open('c:/Users/Adminb/OneDrive/Documents/Projects/qgan/QINR-QGAN/QGAN-QIREN-2024-MNIST/results_analysis/relaxed_structures.pkl', 'rb') as f:
    cache = IgnorantUnpickler(f).load()

print('Cache keys:', cache.keys())
print('Length of q_structs:', len(cache['q_structs']))
print('Type of first st:', type(cache['q_structs'][0]))
q_st = cache['q_structs'][0]
if hasattr(q_st, '__dict__'):
    print('Struct keys:', q_st.__dict__.keys())
