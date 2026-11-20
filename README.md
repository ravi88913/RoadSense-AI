# RoadSense AI

## Vision-Based Road Segmentation and Path Guidance Using Deep Learning

RoadSense AI is a Streamlit-based research prototype that performs semantic road-scene segmentation and draws a suggested path over the detected drivable road region. It uses a U-Net segmentation model with a ResNet34 encoder, trained for 12 output classes derived from the CamVid dataset labels.

The application supports two ways to run inference:

- **Image Analysis**: upload a JPG, JPEG, or PNG road image and inspect the segmentation overlay, obstacle-aware road mask, and path guidance result.
- **Live Camera**: process webcam frames in a browser through Streamlit WebRTC and display live path guidance.

> **Safety disclaimer:** RoadSense AI is a research and visualization prototype. It does not control a vehicle, is not a driver-assistance system, and must not be used to make real-world driving or safety decisions.

## Problem statement

Interpreting a road scene requires distinguishing drivable surfaces from surrounding infrastructure, dynamic road users, and other scene elements. RoadSense AI explores this problem by assigning a semantic class to each image pixel, extracting a road mask, excluding selected obstacle classes, and fitting a visual guidance line through the remaining road area.

## Objectives

- Segment road-scene images into 12 semantic classes.
- Identify whether a road is present in a central lower-image region.
- Produce an obstacle-aware mask for the predicted road class.
- Visualize a smooth candidate path for uploaded images and webcam frames.
- Provide an accessible browser interface for inference demonstrations.

## Key features

- U-Net model with a ResNet34 encoder.
- CPU inference using a saved PyTorch checkpoint.
- Image upload workflow for JPG, JPEG, and PNG files.
- Live webcam processing with Streamlit WebRTC.
- Road overlay, obstacle-aware binary mask, path line, status, and inference-time display.
- Morphological cleanup and quadratic curve fitting for path visualization.
- Graceful messages when a road or clear path is not detected.

## Technology stack

| Area | Libraries / tools |
| --- | --- |
| Deep learning | PyTorch, TorchVision, Segmentation Models PyTorch |
| Model | U-Net with ResNet34 encoder |
| Image processing | OpenCV, NumPy, Pillow |
| Web application | Streamlit |
| Live video | Streamlit WebRTC, aiortc, PyAV |
| Dataset basis | CamVid |

## Model architecture and inference

`model.py` creates `segmentation_models_pytorch.Unet` with these settings:

- Encoder: `resnet34`
- Encoder weights at application construction: `None`
- Input channels: 3 (RGB)
- Output channels/classes: 12
- Activation: `None` (raw logits; class prediction uses `argmax`)
- Inference device: CPU

For each image, the application converts it to RGB, resizes it to **480 × 360**, scales pixels to `[0, 1]`, and normalizes them using ImageNet mean `[0.485, 0.456, 0.406]` and standard deviation `[0.229, 0.224, 0.225]`. The predicted class-index mask is resized back to the original image dimensions with nearest-neighbor interpolation.

The app loads `model/best_model.pth`. It accepts checkpoints stored directly as a state dictionary or under `model_state_dict` / `state_dict`, and removes a leading `module.` prefix when present.

## Dataset and classes

The model uses 12 consolidated semantic classes based on CamVid labels. The exact mapping is stored in [`model/class_mapping.json`](model/class_mapping.json).

| ID | Output class | Source CamVid label(s) represented |
| ---: | --- | --- |
| 0 | Road | Road |
| 1 | Sidewalk | Sidewalk |
| 2 | Building | Building |
| 3 | Sky | Sky |
| 4 | Tree | Tree |
| 5 | Pole | Column_Pole |
| 6 | Sign | SignSymbol, TrafficLight |
| 7 | Fence | Fence |
| 8 | Car | Car, SUVPickupTruck, Truck_Bus |
| 9 | Pedestrian | Pedestrian, Child |
| 10 | Bicyclist | Bicyclist |
| 11 | LaneMarking | LaneMkgsDriv, LaneMkgsNonDriv |

The mapping also defines `255` as the ignore index.

## How segmentation and path guidance work

1. The model predicts one of the 12 classes for every pixel.
2. A road-present check examines the central lower area of the image. It requires at least 30% of that region to be predicted as class `0` (`Road`).
3. The road mask is formed from class `0`. Classes `8` (`Car`), `9` (`Pedestrian`), and `10` (`Bicyclist`) are treated as obstacles; their mask is dilated with an elliptical OpenCV kernel and removed from the road mask.
4. The remaining road mask is morphologically closed, sampled across 40 horizontal rows, and grouped into viable road segments.
5. The selected segment is tracked by center position from the bottom upward. If at least eight points are available, a second-order polynomial is fitted and rendered as a magenta line.

The path is a visual heuristic based on semantic predictions, not a motion-planning or collision-avoidance system.

## Recorded training metrics

`training_history.json` contains 25 recorded training epochs. The final recorded validation values are:

| Metric | Value |
| --- | ---: |
| Validation loss | 0.673263 |
| Validation mean IoU | 0.737399 |
| Validation pixel accuracy | 0.936613 |

These values are reported from the training-history file. The repository does not include a separate evaluation script, test-set report, or a statement connecting a particular metric entry to `best_model.pth`.

## Project structure

```text
RoadSense-AI/
├── app.py                  # Streamlit UI, image analysis, and live-camera processing
├── model.py                # U-Net creation, checkpoint loading, and image inference
├── training_history.json   # Recorded training and validation history
├── requirements.txt        # Runtime dependencies
└── model/
    ├── best_model.pth      # Trained PyTorch checkpoint
    └── class_mapping.json  # 12-class label mapping
```

## Installation (Windows PowerShell)

### Prerequisites

- Windows 10 or later
- Python **3.12** (the included local environment was created with Python 3.12.3)
- A webcam and browser permission if using Live Camera

Open PowerShell in the project root:

```powershell
Set-Location "C:\path\to\RoadSense-AI"
```

### Create and activate a virtual environment

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks local activation scripts for the current session, run the following once in that session and activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### Install dependencies

Upgrade pip and install the project dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` specifies CPU-compatible dependency ranges and intentionally does not pin a CUDA-specific PyTorch build. For a GPU-enabled PyTorch installation, install the appropriate PyTorch build from the official PyTorch selector before or while resolving these dependencies.

### Check model assets

Before starting the app, ensure these files are present at the exact paths below:

```text
model\best_model.pth
model\class_mapping.json
```

`app.py` loads the checkpoint through `model.py` from `model/best_model.pth`. The class-mapping JSON documents the label mapping and should remain alongside the checkpoint.

## Run the application

With the virtual environment activated, run:

```powershell
streamlit run app.py
```

Streamlit will print a local URL in PowerShell. Open it in a browser, typically `http://localhost:8501`.

To stop the app, press `Ctrl+C` in the PowerShell window.

## Using Image Analysis

1. In the sidebar, choose **Image Analysis**.
2. Upload a `.jpg`, `.jpeg`, or `.png` road image.
3. Select **Analyze Image**.
4. Review the original image, green road segmentation overlay, obstacle-aware road mask, path guidance visualization, inference time, and path status.

If the central lower region does not meet the road-detection threshold, the app shows a warning instead of generating a path.

## Using Live Camera

1. In the sidebar, choose **Live Camera**.
2. Allow camera access when the browser prompts you.
3. Aim the camera toward a road scene.
4. The live view displays one of `ROAD NOT FOUND`, `PATH DETECTED`, `NO CLEAR PATH`, or `PROCESSING ERROR`, along with the last frame's inference time.

Live camera functionality relies on Streamlit WebRTC, aiortc, and PyAV. Browser webcam access may require a secure context or localhost, depending on browser and deployment settings.

## Screenshots

Add screenshots to an `assets/` directory (or update these paths to match your preferred location).

| View | Placeholder |
| --- | --- |
| Home screen | `assets/home-screen.png` |
| Image Analysis output | `assets/image-analysis.png` |
| Live Camera output | `assets/live-camera.png` |

Example once images are added:

```markdown
![Image Analysis output](assets/image-analysis.png)
```

## Limitations

- Inference is explicitly configured for CPU, so live performance depends on the host machine.
- Road detection and path extraction rely on fixed image-region thresholds, class IDs, morphology settings, and a quadratic curve heuristic.
- Only cars, pedestrians, and bicyclists are explicitly removed as obstacles; other classes are not treated as obstacles by the current code.
- Predictions can degrade with unusual camera viewpoints, poor lighting, weather, occlusion, domain shift, or errors in the segmentation mask.
- The application does not estimate depth, road geometry, traffic rules, vehicle state, or safe trajectories.
- The repository contains inference code and recorded training history, not the dataset-preparation or model-training pipeline.

## Future improvements

- Add reproducible dataset preparation, training, validation, and test evaluation scripts.
- Report class-wise metrics and evaluate on a held-out test set.
- Add temporal smoothing and confidence-aware segmentation for video.
- Incorporate additional obstacle and free-space reasoning, depth, and lane geometry.
- Make preprocessing, thresholds, class IDs, and device selection configurable.
- Add tests, logging, model/version metadata, and deployment documentation.
- Benchmark CPU and optional GPU inference under representative conditions.

## Author / contributors

- **Author:** _[Add author name]_
- **Contributors:** _[Add contributor names or GitHub profiles]_

## License

_[Add the project license, if applicable.]_
