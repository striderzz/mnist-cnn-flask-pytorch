# FashionMNIST CNN · Flask Demo

A single-page Flask web app that serves a TinyVGG-style CNN trained on the [FashionMNIST dataset](https://github.com/zalandoresearch/fashion-mnist).  
The UI shows model metrics (train/test size, params, weights, optimizer, loss, epochs, accuracy, precision/recall/F1) and lets you test the model on random samples with top-3 probabilities.

![Demo Screenshot](docs/screenshot.png)

---

## Features
- PyTorch model: TinyVGG-style CNN trained on FashionMNIST (28×28 grayscale images).
- Metrics in UI: dataset sizes, params, optimizer, loss fn, epochs, final train/val loss, accuracy, precision, recall, F1.
- Flask server: one-click random sample inference; predictions + top-3 probabilities.
- Crisp image rendering: 28×28 inputs upscaled with nearest-neighbor.
- End-to-end pipeline: dataset → model → logits → predictions → web deployment.

---

## How It Works

### Dataset → Model → Logits → Predictions
1. Dataset: FashionMNIST (60,000 train / 10,000 test images), each 28×28 grayscale.  
   Labels: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot.  

2. Transforms:  
   - Train: random horizontal flip, affine shifts, ToTensor.  
   - Test: ToTensor only.  

3. Model Architecture (FashionMNISTModelV2):  
   - Block 1: Conv(3×3, 32) → ReLU → Conv(3×3, 32) → ReLU → MaxPool(2) (28×28 → 14×14)  
   - Block 2: Conv(3×3, 32) → ReLU → Conv(3×3, 32) → ReLU → MaxPool(2) (14×14 → 7×7)  
   - Classifier: Flatten → Linear(32×7×7 → 10 logits)  

4. Forward flow:  
   Input tensor (batch, 1, 28, 28) → CNN → logits (batch, 10) → Softmax → probabilities → Argmax → prediction.  

---

## Training Setup
- Loss function: CrossEntropyLoss  
- Optimizer: Adam  
- Epochs: 10–20 typical (configurable)  
- Batch size: 128  
- Device: CUDA if available, else CPU  
- Evaluation metrics: CE loss, Accuracy, Precision, Recall, F1 (macro)  

### Example Training Log
