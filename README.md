# Optical Character Recognition for Receipt

## Sample Results
Input Image             |  Output
:----------------------:|:----------------------:
<img src="./data/tes.jpg" width="300" title="sample-input">  |  <img src="./data/sample_output.jpg" width="300" title="sample-output">

## References

| Title                                                                                   | Author           | Year | Github | Paper | Download Model|
| ----------------------------------------------------------------------------------------| ---------------- | ---- | --------- | ----- |  -------- | 
| Character Region Awareness for Text Detection                                           | Clova AI Research, NAVER Corp.| 2019 | https://github.com/clovaai/CRAFT-pytorch | https://arxiv.org/abs/1904.01941 | [craft_mlt_25k.pth](https://drive.google.com/file/d/1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ/view)|
| What Is Wrong With Scene Text Recognition Model Comparisons? Dataset and Model Analysis | Clova AI Research, NAVER Corp.| 2019 | https://github.com/clovaai/deep-text-recognition-benchmark | https://arxiv.org/abs/1904.01906 | [TPS-ResNet-BiLSTM-Attn-case-sensitive.pth](https://www.dropbox.com/sh/j3xmli4di1zuv3s/AAArdcPgz7UFxIHUuKNOeKv_a?dl=0) |

## Folder structure
```
.
├─ configs               
|  ├─ craft_config.yaml  
|  └─ star_config.yaml   
├─ data
|  ├─ sample_output.jpg  
|  └─ tes.jpg
├─ notebooks                          
|  ├─ export_onnx_model.ipynb         
|  ├─ inference_default_engine.ipynb  
|  ├─ inference_onnx_engine.ipynb     
|  └─ test_api.ipynb                  
├─ src                                                               
|  ├─ text_detector                                         
|  │  ├─ basenet                                           
|  │  │  ├─ __init__.py                           
|  │  │  └─ vgg16_bn.py                           
|  │  ├─ modules                                              
|  │  │  ├─ __init__.py                           
|  │  │  ├─ craft.py                              
|  │  │  ├─ craft_utils.py                        
|  │  │  ├─ imgproc.py                            
|  │  │  ├─ refinenet.py                          
|  │  │  └─ utils.py                              
|  │  ├─ __init__.py                              
|  │  ├─ infer.py                                 
|  │  └─ load_model.py                            
|  ├─ text_recognizer                                           
|  │  ├─ modules                                              
|  │  │  ├─ dataset.py                            
|  │  │  ├─ feature_extraction.py                 
|  │  │  ├─ model.py                              
|  │  │  ├─ model_utils.py                        
|  │  │  ├─ prediction.py                         
|  │  │  ├─ sequence_modeling.py                  
|  │  │  ├─ transformation.py                     
|  │  │  └─ utils.py                              
|  │  ├─ __init__.py                              
|  │  ├─ infer.py                                 
|  │  └─ load_model.py                            
|  ├─ __init__.py                                 
|  ├─ engine.py                                   
|  └─ model.py                                    
├─ .gitignore
├─ CONTRIBUTING.md
├─ Dockerfile
├─ environment.yaml
├─ LICENSE
├─ main.py
├─ pyproject.toml
├─ README.md
├─ requirements.txt
├─ setup.cfg
```

## Model Preparation
You need to create "models" folder to store this:
- detector_model = "models/text_detector/craft_mlt_25k.pth"
- recognizer_model = "models/text_recognizer/TPS-ResNet-BiLSTM-Attn-case-sensitive.pth"

Download all of pretrained models from "References" section

## Requirements
You can setup the environment using conda or pip
```
pip install -r requirements.txt
```
or
```
conda env create -f environment.yaml
```

## Container
```
docker build -t receipt-ocr .
docker run -d --name receipt-ocr-service -p 80:80 receipt-ocr
docker start receipt-ocr-service
docker stop receipt-ocr-service
```

## How to contribute?
Check the docs [here](CONTRIBUTING.md)

## Creator
[![](https://github.com/andreaschandra/git-assets/blob/master/pictures/ruben.png)](https://github.com/rubentea16)