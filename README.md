**Title:**

# Custom Object Detection with Detectron2 and Transfer Learning

**README.md Content:**

---

# Custom Object Detection with Detectron2 and Transfer Learning

## Overview

This repository demonstrates how to train a custom object detector using [Detectron2](https://github.com/facebookresearch/detectron2) and transfer learning from pretrained models. By leveraging pretrained weights, we can efficiently train a high-performance object detection model on a custom dataset with reduced training time and improved accuracy.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Inference](#inference)
- [Evaluation](#evaluation)
- [Results](#results)
- [Configuration](#configuration)
- [Acknowledgments](#acknowledgments)
- [License](#license)

## Introduction

Object detection is a crucial task in computer vision, enabling systems to identify and locate objects within images or videos. This project utilizes Detectron2—a robust and flexible object detection library by Facebook AI Research—to train a custom object detector. By applying transfer learning from pretrained models like Faster R-CNN, we can adapt the model to our specific dataset efficiently.

## Features

- **Transfer Learning**: Utilize pretrained models to improve training efficiency.
- **Custom Training**: Train on your own dataset in COCO format.
- **Evaluation Metrics**: Assess model performance using COCO Evaluator.
- **Visualization Tools**: Visualize training samples and inference results.

## Installation

### Prerequisites

- Python 3.6 or higher
- PyTorch (compatible with your CUDA version)
- CUDA Toolkit (if using GPU acceleration)

### Steps

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name
   ```

2. **Install Dependencies**

   Install the required Python packages:

   ```bash
   pip install torch torchvision
   pip install opencv-python numpy pyyaml==5.1
   pip install 'git+https://github.com/facebookresearch/detectron2.git'
   ```

   Ensure that your `torch` and `torchvision` versions are compatible with your CUDA installation.

## Dataset Preparation

1. **Download or Prepare Your Dataset**

   Replace the dataset download command in the code with your dataset:

   ```bash
   curl -L "your-dataset-link" > dataset.zip
   unzip dataset.zip
   rm dataset.zip
   ```

2. **Register the Dataset**

   Use Detectron2's `register_coco_instances` to register your dataset:

   ```python
   from detectron2.data.datasets import register_coco_instances

   register_coco_instances("my_dataset_train", {}, "path/to/train/annotations.json", "path/to/train/images")
   register_coco_instances("my_dataset_test", {}, "path/to/test/annotations.json", "path/to/test/images")
   ```

3. **Visualize the Data (Optional)**

   ```python
   from detectron2.utils.visualizer import Visualizer
   from detectron2.data import MetadataCatalog, DatasetCatalog
   import random
   import cv2

   dataset_dicts = DatasetCatalog.get("my_dataset_train")
   metadata = MetadataCatalog.get("my_dataset_train")

   for d in random.sample(dataset_dicts, 3):
       img = cv2.imread(d["file_name"])
       visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.5)
       vis = visualizer.draw_dataset_dict(d)
       cv2.imshow('Sample', vis.get_image()[:, :, ::-1])
       cv2.waitKey(0)
   ```

## Training

1. **Configure the Model**

   ```python
   from detectron2.config import get_cfg
   from detectron2 import model_zoo

   cfg = get_cfg()
   cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
   cfg.DATASETS.TRAIN = ("my_dataset_train",)
   cfg.DATASETS.TEST = ("my_dataset_test",)
   cfg.DATALOADER.NUM_WORKERS = 2
   cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
   cfg.SOLVER.IMS_PER_BATCH = 2
   cfg.SOLVER.BASE_LR = 0.0025
   cfg.SOLVER.MAX_ITER = 500
   cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
   cfg.MODEL.ROI_HEADS.NUM_CLASSES = <number_of_classes>
   ```

   Replace `<number_of_classes>` with the number of classes in your dataset.

2. **Start Training**

   ```python
   from detectron2.engine import DefaultTrainer
   import os

   os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
   trainer = DefaultTrainer(cfg)
   trainer.resume_or_load(resume=False)
   trainer.train()
   ```

## Inference

1. **Set Up the Predictor**

   ```python
   from detectron2.engine import DefaultPredictor

   cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
   cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set the testing threshold
   predictor = DefaultPredictor(cfg)
   ```

2. **Run Inference on Images**

   ```python
   import glob
   from detectron2.utils.visualizer import ColorMode

   for image_path in glob.glob('path/to/test/images/*.jpg'):
       im = cv2.imread(image_path)
       outputs = predictor(im)
       v = Visualizer(im[:, :, ::-1],
                      metadata=MetadataCatalog.get("my_dataset_test"),
                      scale=0.8,
                      instance_mode=ColorMode.SEGMENTATION)
       out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
       cv2.imshow('Inference', out.get_image()[:, :, ::-1])
       cv2.waitKey(0)
   ```

## Evaluation

1. **Evaluate the Model**

   ```python
   from detectron2.evaluation import COCOEvaluator, inference_on_dataset
   from detectron2.data import build_detection_test_loader

   evaluator = COCOEvaluator("my_dataset_test", cfg, False, output_dir="./output/")
   val_loader = build_detection_test_loader(cfg, "my_dataset_test")
   inference_on_dataset(trainer.model, val_loader, evaluator)
   ```

2. **View Results**

   Evaluation metrics such as mAP (mean Average Precision) will be displayed in the console and saved in the `output` directory.

## Results

Include sample images and evaluation metrics to showcase the model's performance. You can display images with bounding boxes and class labels to visualize detection results.

## Configuration

Save the configuration for future reference:

```python
with open('config.yml', 'w') as f:
    f.write(cfg.dump())
```

## Acknowledgments

- **[Detectron2](https://github.com/facebookresearch/detectron2)**: A next-generation library that provides state-of-the-art detection and segmentation algorithms.
- **[PyTorch](https://pytorch.org/)**: An open-source machine learning library for Python.
- **[COCO Dataset](https://cocodataset.org/#home)**: A large-scale object detection, segmentation, and captioning dataset.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Feel free to customize the README.md content to better suit your project's specifics, such as adding more detailed instructions, adjusting code snippets, or including additional sections like FAQs or Troubleshooting.
