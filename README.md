# Fashion MNIST CNN · Flask Demo

A presentation-ready Flask app that runs a TinyVGG-style CNN on **FashionMNIST** and displays live metrics (train/test size, #classes, params, weights size, device, test accuracy/loss) with a one-click **Random Test Image** prediction.

- **App**: `app.py` — UI, inference, and metrics
- **Model/Trainer**: `cnn_model.py` — TinyVGG-style model + simple training loop

---

## ✨ Features
- Single-page UI (no uploads) with a polished, compact layout.
- Metrics strip: Train/Test size, Classes, Params, Weights, Device, Test Top-1, (optional) Epochs & Train/Val loss via sidecar JSON.
- Fast CPU-friendly inference; uses GPU automatically if available.

---

## 🧰 Tech stack
Flask, PyTorch, TorchVision, Pillow.

---

## 📦 Installation

```bash
# 1) (Optional) create and activate a virtualenv
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# 2) Install dependencies
pip install flask torch torchvision pillow

