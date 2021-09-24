from abc import ABC, abstractmethod
import torch
from text_detector.load_model import load_craft, load_craft_onnx
from text_recognizer.load_model import load_star


class ReceiptOCR_BaseModel(ABC):
    """Abstract base class for Receipt OCR models."""

    def __init__(self, detector_cfg, recognizer_cfg):
        self._cfg_detector, self._detector = self._load_detector(detector_cfg)
        self._cfg_recognizer, self._recognizer, self._converter = self._load_recognizer(
            recognizer_cfg)

    @abstractmethod
    def _load_detector(self, detector_cfg):
        """Return CRAFT model."""

    @abstractmethod
    def _load_recognizer(self, recognizer_cfg):
        """Return STAR model."""

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


class ReceiptOCR_DefaultModel(ReceiptOCR_BaseModel):
    """Default implementation of Receipt OCR models."""

    def _load_detector(self, detector_cfg):
        return load_craft(detector_cfg)

    def _load_recognizer(self, recognizer_cfg):
        return load_star(recognizer_cfg)


class ReceiptOCR_ONNXModel(ReceiptOCR_DefaultModel):
    def _load_detector(self, detector_cfg):
        return load_craft_onnx(detector_cfg)
