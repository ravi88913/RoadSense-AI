
import torch
import segmentation_models_pytorch as smp
from pathlib import Path

NUM_CLASSES = 12
DEVICE = "cpu"
CHECKPOINT_PATH = Path("model/best_model.pth")

def create_model():
    return smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=NUM_CLASSES,
        activation=None
    )

def load_model():
    network = create_model()

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE
    )

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]

    if any(key.startswith("module.") for key in checkpoint):
        checkpoint = {
            key.replace("module.", "", 1): value
            for key, value in checkpoint.items()
        }

    network.load_state_dict(checkpoint)
    network.to(DEVICE)
    network.eval()

    return network

import cv2
import numpy as np
from PIL import Image
import time

IMAGE_HEIGHT = 360
IMAGE_WIDTH = 480

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def predict_image(network, image):
    image = np.array(image.convert("RGB"))
    resized = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))

    normalized = resized.astype(np.float32) / 255.0
    normalized = (normalized - MEAN) / STD

    tensor = torch.from_numpy(normalized.transpose(2, 0, 1))
    tensor = tensor.unsqueeze(0).float()

    start = time.time()

    with torch.no_grad():
        output = network(tensor)
        prediction = torch.argmax(output, dim=1).squeeze(0).numpy()

    inference_time = time.time() - start

    prediction = cv2.resize(
        prediction.astype(np.uint8),
        (image.shape[1], image.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )

    return prediction, inference_time
