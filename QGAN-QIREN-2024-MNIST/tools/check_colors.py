from PIL import Image
import numpy as np
from collections import Counter

img = Image.open(r'c:\Users\Adminb\OneDrive\Documents\Projects\qgan\QINR-QGAN\QGAN-QIREN-2024-MNIST\results_analysis\phase_diagram.png').convert('RGB')
data = np.array(img)

# Get top 10 colors
colors = [tuple(c) for c in data.reshape(-1, 3)]
counter = Counter(colors)
for color, count in counter.most_common(10):
    print(f'Color RGBA {color}: {count} pixels')
