import string
import torch

from .modules.model_utils import CTCLabelConverter, AttnLabelConverter
from .modules.utils import yaml_loader, DictObj
from .model import Model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_star(config_file):
    cfg = yaml_loader(config_file)
    obj_cfg = DictObj(cfg)

    """ vocab / character number configuration """
    if obj_cfg.sensitive:
        obj_cfg.character = string.printable[:-6]

    if 'CTC' in obj_cfg.Prediction:
        converter = CTCLabelConverter(obj_cfg.character)
    else:
        converter = AttnLabelConverter(obj_cfg.character)

    obj_cfg.num_class = len(converter.character)

    net = Model(obj_cfg)
    net = torch.nn.DataParallel(net).to(device)
    net.load_state_dict(torch.load(obj_cfg.model, map_location=device))
    return obj_cfg, net
