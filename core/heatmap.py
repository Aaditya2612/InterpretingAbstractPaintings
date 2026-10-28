"""Gradient-weighted class-activation-map (Grad-CAM) attention heatmap.

Implemented directly against torchvision's ResNet50 so we don't need the
external `pytorch-grad-cam` package. The model is loaded once and reused.
"""

from __future__ import annotations

import threading

import cv2
import numpy as np
from PIL import Image

from . import config

_LOCK = threading.Lock()
_model = None
_device = None
_transform = None

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Output heatmaps are capped to this longest side (display only).
_DISPLAY_DIM = 640


def _ensure_loaded():
    global _model, _device, _transform
    if _model is not None:
        return
    with _LOCK:
        if _model is not None:
            return
        import torch
        from torchvision import models, transforms

        torch.set_num_threads(config.TORCH_THREADS)
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        _model.eval().to(_device)
        _transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=_MEAN, std=_STD),
        ])


def _gradcam(torch, F, model, target_layer, input_tensor):
    """Return a normalised 0..1 saliency map the same size as the input."""
    activations: dict = {}
    gradients: dict = {}

    def forward_hook(m, inp, out):
        activations["out"] = out

    def backward_hook(m, grad_in, grad_out):
        gradients["out"] = grad_out[0]

    h1 = target_layer.register_forward_hook(forward_hook)
    h2 = target_layer.register_full_backward_hook(backward_hook)

    model.zero_grad()
    logits = model(input_tensor)
    target = logits.argmax(dim=1)
    loss = logits[0, target[0]]  # scalar
    loss.backward()

    h1.remove()
    h2.remove()

    weights = gradients["out"].mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
    cam = (weights * activations["out"]).sum(dim=1, keepdim=True)  # [1, 1, H, W]
    cam = F.relu(cam)
    cam = F.interpolate(cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False)
    cam = cam.squeeze().detach().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam


def generate(image_path: str) -> np.ndarray:
    """Produce an RGB attention overlay (uint8 HxWx3), capped at a display size."""
    _ensure_loaded()
    import torch
    import torch.nn.functional as F

    image = Image.open(image_path).convert("RGB")
    # Cap the working size so the output stays small even for huge uploads.
    if max(image.size) > _DISPLAY_DIM:
        image.thumbnail((_DISPLAY_DIM, _DISPLAY_DIM), Image.LANCZOS)

    image_np = np.array(image).astype(np.float32) / 255.0
    h, w = image_np.shape[:2]

    input_tensor = _transform(image).unsqueeze(0).to(_device)
    input_tensor.requires_grad_(True)

    target_layer = _model.layer4[-1]
    cam = _gradcam(torch, F, _model, target_layer, input_tensor)

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    heatmap = cv2.resize(heatmap, (w, h))

    # Blend so the original artwork stays visible underneath.
    overlay = 0.5 * heatmap + 0.5 * image_np
    return np.uint8(255 * np.clip(overlay, 0, 1))


def load_status() -> bool:
    return _model is not None
