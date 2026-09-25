import pickletools

with open('c:/Users/Adminb/OneDrive/Documents/Projects/qgan/QINR-QGAN/QGAN-QIREN-2024-MNIST/results_analysis/relaxed_structures.pkl', 'rb') as f:
    data = f.read()

with open('c:/Users/Adminb/OneDrive/Documents/Projects/qgan/QINR-QGAN/QGAN-QIREN-2024-MNIST/results_analysis/pickle_dis.txt', 'w') as f:
    pickletools.dis(data, out=f)
