import yaml
import multiprocessing
import onnxruntime as rt

from onnxruntime import InferenceSession, get_all_providers
from yaml.loader import SafeLoader


def yaml_loader(filename):
    with open(filename) as f:
        data = yaml.load(f, Loader=SafeLoader)
        return data


def create_model_for_provider(model_path: str, provider: str) -> InferenceSession:
    """Return inference session for ONNX model with specific provider."""
    assert provider in get_all_providers(
    ), f"provider {provider} not found, {get_all_providers()}"

    sess_options = rt.SessionOptions()
    sess_options.intra_op_num_threads = multiprocessing.cpu_count()
    sess_options.execution_mode = rt.ExecutionMode.ORT_PARALLEL
    sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL

    return InferenceSession(model_path, sess_options, providers=[provider])
