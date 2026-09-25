from PIL import Image
import numpy as np

img = Image.open(r'c:\Users\Adminb\OneDrive\Documents\Projects\qgan\QINR-QGAN\QGAN-QIREN-2024-MNIST\results_analysis\phase_diagram.png').convert('RGBA')
data = np.array(img)

r, g, b, a = data.T
print('Total pixels:', data.size // 4)
print('Transparent pixels (a == 0):', np.sum(a == 0))
print('Dark bg pixels (#161b22):', np.sum((r == 22) & (g == 27) & (b == 34)))
print('White background pixels (#ffffff):', np.sum((r == 255) & (g == 255) & (b == 255) & (a == 255)))
print('White text/pixels:', np.sum((r > 200) & (g > 200) & (b > 200)))
print('Black text/pixels:', np.sum((r < 50) & (g < 50) & (b < 50)))
