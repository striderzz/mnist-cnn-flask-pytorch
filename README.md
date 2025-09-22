# Fashion CNN · Flask Demo

A minimal, presentation-ready web app that runs a TinyVGG-style CNN on **FashionMNIST** and shows live metrics (train/test size, params, weights size, device, test accuracy/loss) alongside a one-click **Random Test Image** prediction.  
App: `app.py` · Model/Trainer: `cnn_model.py`.  <!-- cites source files in text -->
``Train/test UI & metrics are rendered by the Flask app``. :contentReference[oaicite:2]{index=2}  
``TinyVGG-style model + training utilities are here``. :contentReference[oaicite:3]{index=3}

---

## ✨ Features
- Single-page UI (no uploads) with **Random Test Image** button.
- Live **metrics strip**: train/test counts, #classes, param count, weights size, device, test top-1 (and easy extension for losses/epochs via a meta file).
- TinyVGG-style CNN (2 conv blocks + linear classifier) with a simple **training loop** and autosave of best weights. :contentReference[oaicite:4]{index=4}
- Fast CPU-friendly inference; GPU auto-used if available. :contentReference[oaicite:5]{index=5}

---

## 🧰 Tech stack
Flask, PyTorch, TorchVision, Pillow.

---

## 📦 Installation

```bash
# 1) Create & activate a venv (optional)
python -m venv .venv && source .venv/bin/activate  # (Windows) .venv\Scripts\activate

# 2) Install deps
pip install flask torch torchvision pillow

# 3) (Optional) train to produce weights
python cnn_model.py   # saves models/03_pytorch_computer_vision_model_2.pth  :contentReference[oaicite:6]{index=6}
