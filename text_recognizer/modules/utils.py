import yaml
from yaml.loader import SafeLoader


def yaml_loader(filename):
    with open(filename) as f:
        data = yaml.load(f, Loader=SafeLoader)
        return data

class DictObj:
    def __init__(self, in_dict:dict):
        assert isinstance(in_dict, dict)
        for key, val in in_dict.items():
            if isinstance(val, (list, tuple)):
                setattr(self, key, [DictObj(x) if isinstance(x, dict) else x for x in val])
            else:
                setattr(self, key, DictObj(val) if isinstance(val, dict) else val)