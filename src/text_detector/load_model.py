import torch
import torch.backends.cudnn as cudnn

from collections import OrderedDict
from .modules.utils import yaml_loader, create_model_for_provider
from .modules.craft import CRAFT


def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


def load_craft(config_file):
    cfg = yaml_loader(config_file)
    net = CRAFT()

    print('Loading weights from checkpoint (' + cfg['model'] + ')')
    if cfg['cuda']:
        net.load_state_dict(copyStateDict(torch.load(cfg['model'])))
    else:
        net.load_state_dict(copyStateDict(
            torch.load(cfg['model'], map_location='cpu')))

    if cfg['cuda']:
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False

    net.eval()
    return cfg, net


def load_craft_onnx(config_file):
    cfg = yaml_loader(config_file)
    device = "CUDAExecutionProvider" if torch.cuda.is_available() else "CPUExecutionProvider"
    print('Loading weights from checkpoint (' + cfg['model_onnx'] + ')')
    net = create_model_for_provider(cfg['model_onnx'], device)
    return cfg, net
