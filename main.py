import base64
import numpy as np
import io
import warnings
warnings.filterwarnings("ignore")

from PIL import Image
from fastapi import FastAPI
from pydantic import BaseModel
from src.model import ReceiptOCR_ONNXModel
from src.engine import ReceiptOCR_ONNXEngine


app = FastAPI()

detector_cfg = 'configs/craft_config.yaml'
recognizer_cfg = 'configs/star_config.yaml'
model = ReceiptOCR_ONNXModel(detector_cfg, recognizer_cfg)
engine = ReceiptOCR_ONNXEngine(model)


class Item(BaseModel):      
    image: str


@app.get("/")
def read_root():
    return {'message': 'API is running...'}


@app.post("/ocr/predict/")
def predict(item: Item):
    item = item.dict()
    img_bytes = base64.b64decode(item['image'].encode('utf-8'))
    image = Image.open(io.BytesIO(img_bytes))
    image = np.array(image)
    
    engine.predict(image)
    return engine.result