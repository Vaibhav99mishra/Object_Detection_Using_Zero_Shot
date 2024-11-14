# Object_Detection_Using_Zero_Shot

YOLO-World is a CNN-based zero-shot object detection model, designed to offer fast, efficient object detection without prior training on specific classes. Unlike traditional models that require training, YOLO-World enables users to detect objects with just a defined list of target classes. This repository provides a guide for running YOLO-World on sample image and video data with tools for visualization, confidence adjustment, and area filtering.

## Requirements

- **Hardware**: GPU access is recommended for faster processing.
- **Software**: Python with `inference` and `supervision` packages.

## Setup

### Step 1: Configure GPU

Ensure GPU access by running:
```bash
!nvidia-smi
```

If GPU is not available, set it in **Notebook settings** > **Hardware accelerator** > **GPU**.

### Step 2: Define Project Directory

To manage datasets, images, and models, define a `HOME` constant:
```python
import os
HOME = os.getcwd()
```

### Step 3: Install Required Packages

Install the necessary libraries:
```bash
!pip install -q inference-gpu[yolo-world]==0.9.13
!pip install -q supervision==0.19.0rc3
```

### Step 4: Import Libraries

```python
import cv2
import supervision as sv
from tqdm import tqdm
from inference.models import YOLOWorld
```

## Running YOLO-World Zero-Shot Detection

### 1. Download Example Data

```bash
!wget -P {HOME} -q https://media.roboflow.com/notebooks/examples/dog.jpeg
!wget -P {HOME} -q https://media.roboflow.com/supervision/cookbooks/yellow-filling.mp4
```

Define paths:
```python
SOURCE_IMAGE_PATH = f"{HOME}/dog.jpeg"
SOURCE_VIDEO_PATH = f"{HOME}/yellow-filling.mp4"
```

### 2. Initialize Model

Load the YOLO-World model in one of its three versions (S, M, L):
```python
model = YOLOWorld(model_id="yolo_world/l")
```

### 3. Set Target Classes

Define the list of target classes:
```python
classes = ["person", "backpack", "dog", "eye", "nose", "ear", "tongue"]
model.set_classes(classes)
```

### 4. Run Object Detection on Image

```python
image = cv2.imread(SOURCE_IMAGE_PATH)
results = model.infer(image)
detections = sv.Detections.from_inference(results)
```

Visualize results:
```python
BOUNDING_BOX_ANNOTATOR = sv.BoundingBoxAnnotator(thickness=2)
LABEL_ANNOTATOR = sv.LabelAnnotator(text_thickness=2, text_scale=1, text_color=sv.Color.BLACK)

annotated_image = image.copy()
annotated_image = BOUNDING_BOX_ANNOTATOR.annotate(annotated_image, detections)
annotated_image = LABEL_ANNOTATOR.annotate(annotated_image, detections)
sv.plot_image(annotated_image, (10, 10))
```

### 5. Adjust Confidence Level

Lower the confidence threshold for classes outside the COCO dataset:
```python
results = model.infer(image, confidence=0.003)
```

Add confidence to labels:
```python
labels = [f"{classes[class_id]} {confidence:0.3f}" for class_id, confidence in zip(detections.class_id, detections.confidence)]
```

### 6. Apply Non-Max Suppression (NMS)

To remove duplicate detections:
```python
detections = sv.Detections.from_inference(results).with_nms(threshold=0.1)
```

### 7. Process Video Frame-by-Frame

Generate video frames:
```python
generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)
frame = next(generator)
```

Set target class and run detection on frames:
```python
classes = ["yellow filling"]
model.set_classes(classes)

results = model.infer(frame, confidence=0.002)
detections = sv.Detections.from_inference(results).with_nms(threshold=0.1)
```

### 8. Filter Detections by Area

Filter out detections occupying over 10% of frame area:
```python
video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
width, height = video_info.resolution_wh
frame_area = width * height

detections = detections[(detections.area / frame_area) < 0.10]
```

### 9. Process and Save Annotated Video

Define output path and process video frames:
```python
TARGET_VIDEO_PATH = f"{HOME}/yellow-filling-output.mp4"
with sv.VideoSink(target_path=TARGET_VIDEO_PATH, video_info=video_info) as sink:
    for frame in tqdm(generator, total=video_info.total_frames):
        results = model.infer(frame, confidence=0.002)
        detections = sv.Detections.from_inference(results).with_nms(threshold=0.1)
        detections = detections[(detections.area / frame_area) < 0.10]

        annotated_frame = frame.copy()
        annotated_frame = BOUNDING_BOX_ANNOTATOR.annotate(annotated_frame, detections)
        annotated_frame = LABEL_ANNOTATOR.annotate(annotated_frame, detections)
        sink.write_frame(annotated_frame)
```

## Summary

YOLO-World's zero-shot object detection provides flexibility and speed without training. By setting target classes and adjusting confidence and NMS, users can customize detection for images and videos with efficient visualization tools.
