from PIL import Image
import numpy as np
import os

img_path = r'c:\Users\Adminb\OneDrive\Documents\Projects\qgan\QINR-QGAN\QGAN-QIREN-2024-MNIST\results_analysis\phase_diagram.png'

print(f'Converting {img_path} ...')
img = Image.open(img_path).convert('RGBA')
data = np.array(img)

# Original dark mode colors: BG #161b22 = (22, 27, 34), SPINE #444c56 = (68, 76, 86)
r, g, b, a = data.T
dark_bg = (r == 22) & (g == 27) & (b == 34)
data[..., :3][dark_bg.T] = [255, 255, 255]  # BG -> White

# Convert text / ticks (which is white/light gray) to black
white_txt = (r > 200) & (g > 200) & (b > 200)
data[..., :3][white_txt.T] = [0, 0, 0]

# Change spines lines #444c56 to black
spines = (r == 68) & (g == 76) & (b == 86)
data[..., :3][spines.T] = [0, 0, 0]

# Save back as a white academic plot
img_out = Image.fromarray(data)
img_out.save(img_path)
print('Done.')
