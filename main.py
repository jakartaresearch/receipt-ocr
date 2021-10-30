"""Script for Fast API Endpoint."""
import base64
import io
import warnings
import numpy as np
from fastapi import FastAPI
from PIL import Image
from pydantic import BaseModel
from src.engine import ONNXEngine
from src.model import ONNXModel

warnings.filterwarnings("ignore")


app = FastAPI()

detector_cfg = "configs/craft_config.yaml"
detector_model = "models/text_detector/craft.onnx"
recognizer_cfg = "configs/star_config.yaml"
recognizer_model = "models/text_recognizer/TPS-ResNet-BiLSTM-Attn-case-sensitive.pth"

model = ONNXModel(detector_cfg, detector_model, recognizer_cfg, recognizer_model)
engine = ONNXEngine(model)


class Item(BaseModel):
    image: str


@app.get("/")
def read_root():
    return {"message": "API is running..."}


@app.post("/ocr/predict")
def predict(item: Item):
    item = item.dict()
    img_bytes = base64.b64decode(item["image"].encode("utf-8"))
    image = Image.open(io.BytesIO(img_bytes))
    image = np.array(image)

    engine.predict(image)
    return engine.result
