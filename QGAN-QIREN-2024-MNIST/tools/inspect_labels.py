import numpy as np
import pickle
import os

def load_dataset():
    db_path = 'data/mgmno_100_aug.pickle'
    with open(db_path, 'rb') as f:
        dataset = pickle.load(f)
    real_crystals = dataset['data']
    real_labels = dataset['cond']
    return real_crystals, real_labels

def make_labels(all_labels, n):
    flat = np.array(all_labels, dtype=np.float32).reshape(len(all_labels), -1)
    reps = (n // len(flat)) + 1
    tiled = np.tile(flat, (reps, 1))[:n]
    return tiled.astype(np.float32)

real_crystals, real_labels = load_dataset()
np.random.seed(0)
labels_gen = make_labels(real_labels, 4800)

print(labels_gen[0])
print(labels_gen.shape)

# Let's see if we can decode the label!
