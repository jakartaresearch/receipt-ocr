from model import ReceiptOCR_Model
from text_detector.infer import inference as infer_detector

class ReceiptOCR_Engine():
    def __init__(self, receipt_ocr_model: ReceiptOCR_Model):
        if not isinstance(receipt_ocr_model, ReceiptOCR_Model):
            raise TypeError
        self._model: ReceiptOCR_Model = receipt_ocr_model
    
    def infer(self, image_path):
        output = infer_detector(self._model.cfg_detector, self._model.detector, image_path)
        return output