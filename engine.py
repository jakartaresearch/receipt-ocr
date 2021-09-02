import cv2
from model import ReceiptOCR_Model
from text_detector.infer import inference as infer_detector


class ReceiptOCR_Engine():
    def __init__(self, receipt_ocr_model: ReceiptOCR_Model):
        if not isinstance(receipt_ocr_model, ReceiptOCR_Model):
            raise TypeError
        self._model: ReceiptOCR_Model = receipt_ocr_model

    def inference_detector(self):
        output = infer_detector(self._model.cfg_detector,
                                self._model.detector, self._img_pth)
        self._out_detector = output

    def predict(self, image_path):
        self._img_pth = image_path
        self._img = cv2.imread(image_path)

        self.inference_detector()
        self.get_img_from_bb()

    def get_img_from_bb(self):
        imgs = []
        for bb in self._out_detector:
            imgs.append(self.crop_img(self._img, bb))
        self._imgs = imgs

    def crop_img(self, img, bb):
        x1, y1 = bb[0]
        x2, y2 = bb[1]
        x3, y3 = bb[2]
        x4, y4 = bb[3]

        top_left_x = min([x1, x2, x3, x4])
        top_left_y = min([y1, y2, y3, y4])
        bot_right_x = max([x1, x2, x3, x4])
        bot_right_y = max([y1, y2, y3, y4])

        cropped_image = img[int(top_left_y):int(
            bot_right_y)+1, int(top_left_x):int(bot_right_x)+1]
        return cropped_image
