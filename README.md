
# FashionMNIST CNN · Flask Demo

A single-page Flask web app that serves a TinyVGG-style CNN trained on the [FashionMNIST dataset](https://github.com/zalandoresearch/fashion-mnist).
The UI shows model metrics (train/test size, params, weights, optimizer, loss, epochs, accuracy, precision/recall/F1) and lets you test the model on random samples with top-3 probabilities.


## Screenshot
<img width="1377" height="774" alt="Screenshot (62)" src="https://github.com/user-attachments/assets/f77c3555-8a76-4983-b1ca-34399df6cc95" />

---

## Features

* PyTorch model: TinyVGG-style CNN trained on FashionMNIST (28×28 grayscale images).
* Metrics in UI: dataset sizes, params, optimizer, loss fn, epochs, final train/val loss, accuracy, precision, recall, F1.
* Flask server: one-click random sample inference; predictions + top-3 probabilities.
* Crisp image rendering: 28×28 inputs upscaled with nearest-neighbor.
* End-to-end pipeline: dataset → model → logits → predictions → web deployment.

---

## How It Works

### Dataset → Model → Logits → Predictions

1. Dataset: FashionMNIST (60,000 train / 10,000 test images), each 28×28 grayscale.
   Labels: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot.

2. Transforms:

   * Train: random horizontal flip, affine shifts, ToTensor.
   * Test: ToTensor only.

3. Model Architecture (FashionMNISTModelV2):

   * Block 1: Conv(3×3, 32) → ReLU → Conv(3×3, 32) → ReLU → MaxPool(2) (28×28 → 14×14)
   * Block 2: Conv(3×3, 32) → ReLU → Conv(3×3, 32) → ReLU → MaxPool(2) (14×14 → 7×7)
   * Classifier: Flatten → Linear(32×7×7 → 10 logits)

4. Forward flow:
   Input tensor `(batch, 1, 28, 28)` → CNN → logits `(batch, 10)` → Softmax → probabilities → Argmax → prediction.

---

## Training Setup

* Loss function: CrossEntropyLoss
* Optimizer: Adam
* Epochs: 10–20 typical (configurable)
* Batch size: 128
* Device: CUDA if available, else CPU
* Evaluation metrics: CE loss, Accuracy, Precision, Recall, F1 (macro)

### Example Training Log

```
Epoch 01/10 | train_loss=0.6810 | test_loss=0.4474 | test_acc=83.87%
  Saved better weights (acc=83.87%)
...
Epoch 10/10 | train_loss=0.3988 | test_loss=0.3662 | test_acc=87.01%
```

### Final Metrics (sample run)

* Test CE loss: 0.3096
* Accuracy: 88.90%
* Precision / Recall / F1: \~88–89%

Training metadata (saved as `.meta.json`):

```json
{
  "epochs": 10,
  "final_train_loss": 0.3621,
  "final_val_loss": 0.3096,
  "optimizer": "Adam",
  "loss_fn": "CrossEntropyLoss",
  "lr": 0.01,
  "best_test_acc": 88.90,
  "trained_at": "2025-09-21T23:14:13"
}
```

---

## Flask Web App

* Startup: loads trained weights (`.pth`) and metadata (`.meta.json`).
* Route `/`: picks a random test sample, runs inference, and renders predictions.
* UI: header metrics + prediction panel with image, label, confidence, ground truth, top-3 probabilities.
* Image rendering: 28×28 input upscaled to 320×320 with nearest-neighbor.

---

## Tech Stack

* PyTorch
* TorchVision
* Flask
* scikit-learn
* Pillow

---

## Installation

```bash
git clone https://github.com/yourusername/fashion-cnn-flask-demo.git
cd fashion-cnn-flask-demo

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install flask torch torchvision pillow scikit-learn
```

---

## Training

```bash
python cnn_model.py

# Optional overrides
EPOCHS=10 LR=0.01 BATCH_SIZE=128 HIDDEN_UNITS=32 python cnn_model.py
```

Artifacts created:

* `models/03_pytorch_computer_vision_model_2.pth` (weights)
* `models/03_pytorch_computer_vision_model_2.pth.meta.json` (training metadata)

---

## Run the App

```bash
export MODEL_PATH=models/03_pytorch_computer_vision_model_2.pth
python app.py
```

Open: [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## Overview (Study Notes)

* Problem: Multiclass classification (10 classes) on FashionMNIST.
* Dataset: 28×28 grayscale clothing images.
* Model: TinyVGG-style CNN with 2 conv blocks + linear classifier.
* Forward flow:

  ```
  Image → CNN (Conv+ReLU+Pool) → Flatten → Linear → Logits → Softmax → Probabilities → Argmax → Prediction
  ```
* Training: Adam optimizer, CrossEntropyLoss, \~10 epochs, \~89% accuracy.
* Evaluation: CE loss, Accuracy, Precision, Recall, F1.
* Serving: Flask app loads trained model, samples random test data, shows prediction + top-3.
* Key takeaway: Demonstrates an end-to-end ML project:

  1. Data preprocessing
  2. Model training (PyTorch)
  3. Model evaluation (metrics)
  4. Model deployment (Flask web app)

---

## Project Structure

```
.
├── app.py               # Flask app (UI + inference)
├── cnn_model.py         # Model definition + training loop
├── models/
│   └── 03_pytorch_computer_vision_model_2.pth
│   └── 03_pytorch_computer_vision_model_2.pth.meta.json
├── templates/
│   └── index.html       # UI template
└── docs/
    └── screenshot.png   # Demo screenshot
```

---

## License

MIT
