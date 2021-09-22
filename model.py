from text_detector.load_model import load_craft
from text_recognizer.load_model import load_star


class ReceiptOCR_Model():
    def __init__(self, detector_cfg, recognizer_cfg):
        self._cfg_detector, self._detector = load_craft(detector_cfg)
        self._cfg_recognizer, self._recognizer, self._converter = load_star(recognizer_cfg)

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
