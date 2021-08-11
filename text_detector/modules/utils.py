import yaml
from yaml.loader import SafeLoader


def yaml_loader(filename):
    with open(filename) as f:
        data = yaml.load(f, Loader=SafeLoader)
        return data
