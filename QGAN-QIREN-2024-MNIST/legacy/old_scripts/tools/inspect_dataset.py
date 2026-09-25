import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import numpy as np

pickle_path = r'c:\Users\Jeremy\Documents\Projects\qgan\Composition-Conditioned-Crystal-GAN\Composition_Conditioned_Crystal_GAN\preparing_dataset\mgmno_100.pickle'
try:
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    print(f"Data type: {type(data)}")
    if hasattr(data, 'shape'):
        print(f"Data shape: {data.shape}")
    elif isinstance(data, list):
         print(f"Data length: {len(data)}")
         print(f"First element: {data[0]}")
    # Inspect content of first item
    first_item = data[0]
    print(f"First item type: {type(first_item)}")
    if isinstance(first_item, (tuple, list)):
        print(f"Item structure: {len(first_item)} elements")
        for i, el in enumerate(first_item):
            if hasattr(el, 'shape'):
                print(f"  Element {i} shape: {el.shape}")
            else:
                 print(f"  Element {i}: {el}")
except Exception as e:
    print(f"Error: {e}")
