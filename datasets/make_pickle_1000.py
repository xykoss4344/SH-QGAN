import numpy as np
import pickle

THRESHOLD = 0.4

def make_label(image):
    mg_occ = (np.sum(image[2:10, :],  axis=1) > THRESHOLD).astype('float32').reshape(8,  1)
    mn_occ = (np.sum(image[10:18, :], axis=1) > THRESHOLD).astype('float32').reshape(8,  1)
    o_occ  = (np.sum(image[18:, :],   axis=1) > THRESHOLD).astype('float32').reshape(12, 1)
    return np.vstack([mg_occ, mn_occ, o_occ])

data = np.load('mgmno_1000.npy', allow_pickle=True)
print(f"Loaded mgmno_1000.npy: shape={data.shape}")

output = [(img, make_label(img)) for img in data if img.shape == (30, 3)]
print(f"Valid samples: {len(output)}")
print(f"  image shape: {output[0][0].shape}")
print(f"  label shape: {output[0][1].shape}")

with open('mgmno_1000.pickle', 'wb') as f:
    pickle.dump(output, f)
print("Saved mgmno_1000.pickle")
