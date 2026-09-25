import nbformat

nb = nbformat.read('QGAN_Evaluation.ipynb', as_version=4)

API_KEY = 'hDXXoVT3jdqAXiIan4F9QK9J1Bc7gAGA'

for cell in nb.cells:
    if 'MP_API_KEY' in cell.source:
        # Replace any broken version of the line with the correct one
        lines = cell.source.splitlines()
        new_lines = []
        for line in lines:
            if 'MP_API_KEY = os.environ' in line:
                line = f"MP_API_KEY = os.environ.get('MP_API_KEY', '{API_KEY}')"
            new_lines.append(line)
        cell.source = '\n'.join(new_lines)
        print("Fixed cell source:")
        print(next(l for l in new_lines if 'MP_API_KEY' in l))

nbformat.write(nb, 'QGAN_Evaluation.ipynb')
print("Notebook saved successfully.")
