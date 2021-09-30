FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7
WORKDIR /receipt-ocr
COPY configs /receipt-ocr/configs
COPY models /receipt-ocr/models
COPY src /receipt-ocr/src
COPY main.py /receipt-ocr
COPY requirements.txt /receipt-ocr

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 -y
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 80
CMD gunicorn -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:80