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

## Contribution Table

| Team Member | Main Work Completed | Methods / Models Applied | Evidence File Path(s) |
|---|---|---|---|
| El Mehdi Ziate | Worked across data preparation, classification, detection, segmentation, explainability, open-vocabulary learning, and self-supervised learning, and the dashboard. | EDA, preprocessing pipeline, CNN baseline, Swin Transformer, EfficientViT, Faster R-CNN, SAM2 detection, SAM2 segmentation, Grad-CAM explainability, dashboard explainability, open-vocabulary recognition, DINO / MIM self-supervised learning. | `Notebook/Data_Notebooks/EDA.ipynb` <br> `Notebook/Data_Notebooks/Preprocessing.ipynb` <br> `Notebook/Data_Notebooks/How_To_Use_Pipeline.ipynb` <br> `Pipeline_/eda.py` <br> `Pipeline_/preprocessor.py` <br> `Notebook/Classification/CNN_Baseline.ipynb` <br> `Notebook/Classification/Swin/Swin.ipynb` <br> `Notebook/Classification/Swin/Swin_optimised.ipynb` <br> `Notebook/Classification/EfficientViT.ipynb` <br> `Notebook/Detection/RCNN.ipynb`  <br> `Notebook/Detection/SAM2_Detection.ipynb` <br> `Notebook/Segmentation/SAM2_Segmentation.ipynb`  <br> `Notebook/Explainability/gradcam_explainability.ipynb` <br> `Notebook/Explainability/extended_explainability.ipynb` <br> `Notebook/Explainability/dashboard.ipynb` <br> `Notebook/openvoc/open_voc.ipynb` <br> `Notebook/ssl/Swin_DINO_SSL.ipynb` <br> `Models/CNN.py` <br> `Models/swin.py` <br> `Models/EfficientVit.py` <br> `Models/FastRcnn.py` <br> `Models/SAM2WaRP.py` <br> `Models/Open_vocabulary.py` <br> `Logs/El_Mehdi_Logs/EfficientViT_ElMehdiZiate.pdf` <br> `Logs/El_Mehdi_Logs/FasterRCNN_ElMehdiZiate.pdf` <br> `Logs/El_Mehdi_Logs/SAM2_LoRA_ElMehdiZiate.pdf` <br> `Logs/El_Mehdi_Logs/Swin_DINO_MIM_SSL_training_log.pdf` <br> `Logs/El_Mehdi_Logs/Swin_ElMehdiZiate.pdf` |
| Scott Lewis | Worked on classification, detection and segmentation experiments. | ResNet50, optimised ResNet50, ConvNeXT, optimised ConvNeXT, RetinaNet with ResNet50 backbone, and SegFormer. | `Notebook/Classification/Resnest/ResNet50_Baseline.ipynb` <br> `Notebook/Classification/Resnest/ResNet50_Optimised.ipynb` <br> `Notebook/Classification/Convnext/ConvNeXT_Baseline.ipynb` <br> `Notebook/Classification/Convnext/ConvNeXT_Optimised.ipynb` <br> `Notebook/Detection/RetinaNet_ResNet50_Baseline.ipynb` <br> `Models/ResNet50.py` <br> `Models/ResNet50_Optimised.py` <br> `Models/ConvNeXT.py` <br> `Models/ConvNeXT_Optimised.py` <br> `Logs/Scott_experiment_logs.txt` <br>  `Notebook/Segmentation/SegFormer#U2011B1_HuggingFace.ipynb` |
| Mohamed Fahmi Ahmed | Worked on classification, detection, and segmentation experiments. | EfficientNet, EfficientNet-V2M, MixUp augmentation, simple augmentation, YOLOv26 detection, YOLOv26 segmentation. | `Notebook/Classification/EfficientNet/efficientnet_with_mixup.ipynb` <br> `Notebook/Classification/EfficientNet/efficientnet_with_simple_augmentation.ipynb` <br> `Notebook/Classification/EfficientNet/efficientnetv2m_only_paper_setup_and_parameter.ipynb` <br> `Notebook/Detection/yolo26.ipynb` <br> `Notebook/Segmentation/yolo26_seg_v2.ipynb` <br> `Models/efficientnet.py` <br> `Models/efficientnetv2m.py` <br> `Logs/EfficientNet_MohamedFahmi.pdf` <br> `Logs/YOLO26_MohamedFahmi.pdf` |
| Umme-Yusrah Sumtally | Worked on classification , Detection and segmentation experiments. | MobileViT classification, RT-DETR Detection and Mask2Former segmentation. | `Notebook/Classification/Mobile_vit/MobileViT_fixed.ipynb` <br> `Notebook/Classification/Mobile_vit/MobileViT_model.ipynb` <br> `Notebook/Classification/Mobile_vit/MobileViT_trial(5).ipynb` <br> `Notebook/Segmentation/Mask2Former/Mask2Former_Visuals.ipynb` <br> `Notebook/Segmentation/Mask2Former/Mask2Former_YS_Exp1.ipynb` <br> `Notebook/Segmentation/Mask2Former/Mask2Former_YS_Exp2.ipynb` <br> `Notebook/Segmentation/Mask2Former/Mask2Former_YS_Experiment3.ipynb` <br> `Models/MobileViT.py` <br> `Logs/Yusrah - MobileViT - TRAINING EXPERIMENT LOG.pdf` <br> `Logs/Yusrah Mask2Former Experiments.pdf` <br> `Notebook/Detection/RTDETR_with_Visuals.ipynb` |
| Sayed Omar Aabid | Worked on transformer-based classification experiments and U-Net for Segmentation. | Vision Transformer ViT-B/16 and optimised ViT. | `Notebook/Classification/VIT/VIT.ipynb` <br> `Notebook/Classification/VIT/ViT_optimised.ipynb` <br> `Models/ViT_B.py` <br> `Notebook/Segmentation/UNet_WarpS.ipynb` <br> `Models/UNet.py`   | 