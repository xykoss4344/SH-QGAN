import pickle, types, sys
from importlib.abc import MetaPathFinder, Loader

class SmartMock:
    def __init__(self, *args, **kwargs): pass
    def __getattr__(self, name): return SmartMock()
    def __call__(self, *args, **kwargs): return SmartMock()
    def __setstate__(self, state):
        self.__dict__.update(state)
        # If this is a Structure, it has a 'sites' list or 'species'
        if 'sites' in state:
            print('Found a structure! Keys:', state.keys())
            print('First site type:', type(state['sites'][0]))
            if hasattr(state['sites'][0], '__dict__'):
                print('First site dict keys:', state['sites'][0].__dict__.keys())
                if 'species' in state['sites'][0].__dict__:
                    sp = state['sites'][0].__dict__['species']
                    print('Species:', type(sp))
                    if hasattr(sp, '__dict__'):
                        print('Species dict:', sp.__dict__)
            sys.exit(0)

class SmartLoader(Loader):
    def exec_module(self, module): pass
    def create_module(self, spec):
        m = types.ModuleType(spec.name)
        m.__path__ = []
        setattr(m, 'Structure', SmartMock)
        setattr(m, 'PeriodicSite', SmartMock)
        setattr(m, 'Composition', SmartMock)
        setattr(m, 'Element', SmartMock)
        setattr(m, 'Specie', SmartMock)
        setattr(m, 'Lattice', SmartMock)
        return m

class SmartFinder(MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if any(fullname.startswith(x) for x in ['pymatgen', 'chgnet', 'mace']):
            import importlib.util
            return importlib.util.spec_from_loader(fullname, SmartLoader())
        return None

sys.meta_path.insert(0, SmartFinder())

with open('c:/Users/Adminb/OneDrive/Documents/Projects/qgan/QINR-QGAN/QGAN-QIREN-2024-MNIST/results_analysis/relaxed_structures.pkl', 'rb') as f:
    pickle.Unpickler(f).load()
