from text_detector.load_model import load_craft

class ReceiptOCR_Model():
    def __init__(self, detector_cfg):
        self._cfg_detector, self._detector = load_craft(detector_cfg)

    @property
    def cfg_detector(self): return self._cfg_detector

    @property
    def detector(self): return self._detector
