from abc import ABC, abstractmethod
import time
import math
import cv2
from model import ReceiptOCR_DefaultModel
from text_detector.infer import inference as infer_detector
from text_recognizer.infer import data_preparation, inference as infer_recognizer


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts))
        else:
            print('%r  %2.2f s' % (method.__name__, (te - ts)))
        return result
    return timed


class ReceiptOCR_BaseEngine(ABC):
    def __init__(self, receipt_ocr_model: ReceiptOCR_DefaultModel):
        if not isinstance(receipt_ocr_model, ReceiptOCR_DefaultModel):
            raise TypeError
        self._model: ReceiptOCR_Model = receipt_ocr_model

    @abstractmethod
    def inference_detector(self):
        pass

    @abstractmethod
    def inference_recognizer(self):
        pass

    @abstractmethod
    def predict(self):
        pass


class ReceiptOCR_DefaultEngine(ReceiptOCR_BaseEngine):
    @timeit
    def inference_detector(self):
        output = infer_detector(self._model.cfg_detector,
                                self._model.detector, self._img_pth)
        self._out_detector = output

    @timeit
    def inference_recognizer(self):
        data_loader = data_preparation(
            opt=self._model.cfg_recognizer, list_data=self._imgs)
        pred, conf_score = infer_recognizer(opt=self._model.cfg_recognizer, model=self._model.recognizer,
                                            converter=self._model.converter, data_loader=data_loader)
        output = list(zip(pred, conf_score, self._coords))
        output = filter(lambda x: x[1] > 0.5, output)
        self.raw_output = sorted(output, key=lambda x: x[2][0])
        self.result = self.combine_entity()

    @timeit
    def predict(self, image_path):
        self._img_pth = image_path
        self._img = cv2.imread(image_path)

        self.inference_detector()
        self.get_img_from_bb()
        self.inference_recognizer()

    def get_img_from_bb(self):
        imgs, coords = [], []
        for bb in self._out_detector:
            cropped_img, coord = self.crop_img(self._img, bb)
            imgs.append(cropped_img)
            coords.append(coord)
        self._imgs = imgs
        self._coords = coords

    def crop_img(self, img, bb):
        x1, y1 = bb[0]
        x2, y2 = bb[1]
        x3, y3 = bb[2]
        x4, y4 = bb[3]

        top_left_x = math.ceil(min([x1, x2, x3, x4]))
        top_left_y = math.ceil(min([y1, y2, y3, y4]))
        bot_right_x = math.ceil(max([x1, x2, x3, x4]))
        bot_right_y = math.ceil(max([y1, y2, y3, y4]))
        coord = (top_left_y, bot_right_y, top_left_x, bot_right_x)

        cropped_image = img[top_left_y:bot_right_y+1, top_left_x:bot_right_x+1]
        return cropped_image, coord

    def combine_entity(self):
        THRES = 20
        output, all_entity, entity = [], [], []
        entity.append(self.raw_output[0])

        for idx in range(len(self.raw_output)-1):
            diff = abs(self.raw_output[idx][2][0] -
                       self.raw_output[idx+1][2][0])
            if diff < THRES:
                entity.append(self.raw_output[idx+1])
            else:
                all_entity.append(entity)
                entity = []
                entity.append(self.raw_output[idx+1])

        # Sorting entity by coordinates
        for idx in range(len(all_entity)):
            all_entity[idx] = sorted(
                all_entity[idx], key=lambda x: (x[2][3], x[2][1], x[2][2]))

        # Concatenate Entity
        for entity in all_entity:
            tmp = [x[0] for x in entity]
            output.append(' '.join(tmp))
        return output


class ReceiptOCR_ONNXEngine(ReceiptOCR_DefaultEngine):
    @timeit
    def inference_detector(self):
        output = infer_detector(self._model.cfg_detector, self._model.detector,
                                self._img_pth, onnx=True)
        self._out_detector = output
