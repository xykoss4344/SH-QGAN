import os, pickle
import numpy as np

with open('results_eval_ablation/ablation_report.pkl', 'rb') as f:
    class SafeUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == 'torch.storage' and name == '_load_from_bytes':
                import torch, io
                return lambda b: torch.load(io.BytesIO(b), map_location='cpu', weights_only=False)
            try: return super().find_class(module, name)
            except: 
                class Mock: pass
                return Mock
    cache = SafeUnpickler(f).load()

def get_counts(eh):
    valid = np.array([e for e in eh if e is not None and not np.isnan(e) and 0 <= e < 3.0])
    return [len(valid[valid <= 0.08]), len(valid[(valid > 0.08) & (valid <= 0.12)]), len(valid[(valid > 0.12) & (valid <= 2.0)])]

c_eh = get_counts(cache['cls_ehull'])
a_eh = get_counts(cache['abl_ehull'])
q_eh = get_counts(cache['q_ehull'])

with open('plot_ablation.py', 'r') as f: content = f.read()

content = content.replace('["MIC Valid%", "Stable%\\n(< 0.1 eV/at)", "Near-stable%\\n(0.1-0.5 eV/at)", "Metastable%\\n(0.5-2.0 eV/at)"]', 
                          '["MIC Valid%", "Stable%\\n(<= 80 meV/at)", "Near-stable%\\n(80-120 meV/at)", "Metastable%\\n(120-2000 meV/at)"]')

content = content.replace('c_counts = [40, 5, 20, 14]', f'c_counts = [40, {c_eh[0]}, {c_eh[1]}, {c_eh[2]}]')
content = content.replace('a_counts = [123, 56, 60, 7]', f'a_counts = [123, {a_eh[0]}, {a_eh[1]}, {a_eh[2]}]')
content = content.replace('q_counts = [283, 14, 260, 9]', f'q_counts = [283, {q_eh[0]}, {q_eh[1]}, {q_eh[2]}]')

with open('plot_ablation.py', 'w') as f: f.write(content)

print("plot_ablation.py successfully updated with exact data from ablation cache.")
