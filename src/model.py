"""Model Class."""
from abc import ABC, abstractmethod
from .text_detector.load_model import load_craft, load_craft_onnx
from .text_recognizer.load_model import load_star


class BaseModel(ABC):
    """Abstract base class for Receipt OCR models."""

    def __init__(self, detector_cfg, detector_model, recognizer_cfg, recognizer_model):
        """Init model config.

        Args:
            detector_cfg: config file for text detector
            recognizer_cfg: config file for text recognizer
        """
        self._cfg_detector, self._detector = self._load_detector(detector_cfg, detector_model)
        self._cfg_recognizer, self._recognizer, self._converter = self._load_recognizer(
            recognizer_cfg, recognizer_model
        )

    @property
    def cfg_detector(self):
        return self._cfg_detector

    @property
    def detector(self):
        return self._detector

    @property
    def cfg_recognizer(self):
        return self._cfg_recognizer

    @property
    def recognizer(self):
        return self._recognizer

    @property
    def converter(self):
        return self._converter

    @abstractmethod
    def _load_detector(self):
        """Return CRAFT model."""

    @abstractmethod
    def _load_recognizer(self):
        """Return STAR model."""


class DefaultModel(BaseModel):
    """Default implementation of Receipt OCR models."""

    def _load_detector(self, detector_cfg, detector_model):
        return load_craft(detector_cfg, detector_model)

    def _load_recognizer(self, recognizer_cfg, recognizer_model):
        return load_star(recognizer_cfg, recognizer_model)


class ONNXModel(DefaultModel):
    """ONNX Model."""

    def _load_detector(self, detector_cfg, detector_model):
        return load_craft_onnx(detector_cfg, detector_model)
