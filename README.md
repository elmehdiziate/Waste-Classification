# Waste Classification on WaRP-C

  
> Group coursework  **EEEM068 Applied Machine Learning**,  University of Surrey.
 

**Classification, detection, segmentation, explainability, and self-supervised learning** on the [WaRP — Waste Recycling Plant](https://www.kaggle.com/datasets/parohod/warp-waste-recycling-plant-dataset) dataset.

  
### Clone and install

 ```bash

git  clone  https://github.com/elmehdiziate/Waste-Classification.git
cd  Waste-Classification
Download data: `python download_data.py`

```
  
## Repository Structure

  

```

Waste-Classification/

├── Dataset/
│ ├── data_config.json
├── Models/
│ ├── CNN.py
│ ├── ResNet50.py / ResNet50_Optimised.py
│ ├── efficientnet.py / efficientnetv2m.py
│ ├── ConvNeXT.py / ConvNeXT_Optimised.py
│ ├── ViT_B.py
│ ├── swin.py
│ ├── MobileViT.py
│ ├── EfficientVit.py
│ ├── UNet.py
│ ├── FastRcnn.py
│ ├── SAM2WaRP.py
│ ├── Open_vocabulary.py
│ └── weights/
├── Notebook/
│ ├── Classification/ # CNN, ResNet, EfficientNet, ConvNeXt, ViT, Swin, MobileViT, EfficientViT
│ ├── Detection/ # R-CNN, RetinaNet, RT-DETR, YOLOv26, SAM2
│ ├── Segmentation/ # U-Net, SegFormer, Mask2Former, SAM2, YOLO-seg
│ ├── Explainability/ # Grad-CAM, dashboard
│ ├── Data_Notebooks/ # EDA, preprocessing walkthrough
│ ├── ssl/ # Swin + DINO self-supervised
│ └── openvoc/ # Open-vocabulary detection
├── Pipeline_/
│ ├── eda.py # EDAModule
│ ├── preprocessor.py # Datasets, transforms, samplers, dataloaders
│ └── focal_loss.py
├── download_data.py
└── README.md
```
---
