import pickle
import pprint
import types
from importlib.abc import MetaPathFinder, Loader
import sys

class MockObj:
    def __init__(self, *args, **kwargs): pass
    def __getattr__(self, name): return MockObj()
    def __call__(self, *args, **kwargs): return MockObj()
    def __setstate__(self, state):
        self.__dict__.update(state)

class MockLoader(Loader):
    def exec_module(self, module): pass
    def create_module(self, spec):
        m = types.ModuleType(spec.name)
        m.__path__ = []
        setattr(m, 'Structure', MockObj)
        return m

class MockMetaFinder(MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if any(fullname.startswith(x) for x in ['pymatgen', 'chgnet', 'mace']): 
            import importlib.util
            return importlib.util.spec_from_loader(fullname, MockLoader())
        return None
sys.meta_path.insert(0, MockMetaFinder())

with open('c:/Users/Adminb/OneDrive/Documents/Projects/qgan/QINR-QGAN/QGAN-QIREN-2024-MNIST/results_analysis/relaxed_structures.pkl', 'rb') as f:
    cache = pickle.load(f)
    st = cache['q_structs'][0]
    # print the raw state
    for k, v in st.__dict__.items():
        print(f"{k}: {type(v)}")
        if k == 'sites':
            print(f"  First 3 sites: {v[:3]}")
            for site in v[:3]:
                if hasattr(site, '__dict__'):
                    print("    ", site.__dict__)
