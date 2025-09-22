# FashionMNIST CNN · Flask Demo

A single-page Flask web app that serves a **TinyVGG-style CNN** trained on the [FashionMNIST dataset](https://github.com/zalandoresearch/fashion-mnist).  
The UI shows **model metrics** (train/test size, params, weights, optimizer, loss, epochs, accuracy, precision/recall/F1, etc.) and lets you test the model on random samples with top-3 probabilities.

![Demo Screenshot](docs/screenshot.png)

---

## ✨ Features
- **PyTorch model**: TinyVGG-style CNN trained on FashionMNIST (28×28 grayscale images).
- **Metrics shown in UI**: dataset sizes, params, optimizer, loss fn, epochs, final train/val loss, accuracy, precision, recall, F1.
- **Flask server**: one-click random sample inference; predictions + top-3 probabilities displayed in a single page.
- **Crisp image rendering**: 28×28 inputs upscaled with `NEAREST` to avoid blur.

---

## 📊 How It Works (Step by Step)

### 1. Dataset → Model → Logits → Predictions (PyTorch)
1. **Dataset**: FashionMNIST (60,000 train / 10,000 test images), each 28×28 grayscale.
2. **Transforms**:  
   - Train: random horizontal flip, small affine shifts, `ToTensor()`.  
   - Test: `ToTensor()`.
3. **Model Architecture** (`FashionMNISTModelV2`):
   - **Block 1**: Conv(3×3, 32) → ReLU → Conv(3×3, 32) → ReLU → MaxPool(2) → (28×28 → 14×14)
   - **Block 2**: Conv(3×3, 32) → ReLU → Conv(3×3, 32) → ReLU → MaxPool(2) → (14×14 → 7×7)
   - **Classifier**: Flatten → Linear(32×7×7 → 10 logits)
4. **Logits → Probabilities**:  
   - Forward pass returns raw **logits** `[batch, 10]`.  
   - Apply `torch.softmax` to get class probabilities.  
   - `argmax` picks the top prediction.

### 2. Training Setup
- **Loss function**: CrossEntropyLoss  
- **Optimizer**: Adam  
- **Epochs**: 10–20 typical (configurable)  
- **Batch size**: 128  
- **Device**: CUDA if available, else CPU  
- **Eval metrics**: CE loss, accuracy, precision, recall, F1 (macro average)  

Example training log:
Epoch 01/10 | train_loss=0.6810 | test_loss=0.4474 | test_acc=83.87%
✅ Saved better weights (acc=83.87%)
...
Epoch 10/10 | train_loss=0.3988 | test_loss=0.3662 | test_acc=87.01%


Final metrics (sample run):
- Test CE loss: **0.3096**
- Accuracy: **88.90%**
- Precision / Recall / F1: ~88–89%

### 3. Flask Web App
- **Startup**: Loads trained weights (`.pth`) and optional `.meta.json` with training metadata.
- **Route `/`**: Picks a random test sample, runs inference, and renders predictions.
- **UI**: 
  - Header strip: dataset/model/training metrics
  - Main panel: random image preview, predicted label, confidence, ground truth, top-3 classes
- **Rendering**: Image is upscaled server-side (28→320 px) with nearest neighbor for sharp pixels.

---

## 🧰 Tech Stack
- **PyTorch** for model + training
- **TorchVision** for datasets + transforms
- **Flask** for web UI
- **scikit-learn** for precision/recall/F1
- **Pillow** for image processing

---

## 📦 Installation

```bash
# 1) Clone repo
git clone https://github.com/yourusername/fashion-cnn-flask-demo.git
cd fashion-cnn-flask-demo

# 2) Create venv (optional)
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# 3) Install dependencies
pip install flask torch torchvision pillow scikit-learn
```

## Training

# Train model and save weights
python cnn_model.py

# Optional overrides
EPOCHS=10 LR=0.01 BATCH_SIZE=128 HIDDEN_UNITS=32 python cnn_model.py

