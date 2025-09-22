from cnn_model import FashionMNISTModelV2
import os, io, random, base64, json
from pathlib import Path
from flask import Flask, render_template
from PIL import Image
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch import nn
from sklearn.metrics import precision_score, recall_score, f1_score

# ----------------------------
# Config
# ----------------------------
MODEL_PATH = os.environ.get("MODEL_PATH", "models/03_pytorch_computer_vision_model_2.pth")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 512))
PIN_MEMORY = torch.cuda.is_available()  # avoid warning on CPU

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# ----------------------------
# App
# ----------------------------
app = Flask(__name__)
app.secret_key = "presentable-demo"

# ----------------------------
# Model
# ----------------------------
model = FashionMNISTModelV2(input_shape=1, hidden_units=32, output_shape=len(CLASS_NAMES))
if os.path.exists(MODEL_PATH):
    state = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state)
else:
    print(f"[WARN] MODEL_PATH not found at {MODEL_PATH}. The app will run but predictions will be random.")
model.to(DEVICE).eval()

# ----------------------------
# Datasets
# ----------------------------
_train_len = None
_test_ds = None
_test_len = None

def get_train_len():
    global _train_len
    if _train_len is None:
        train = datasets.FashionMNIST(root="data", train=True, download=True)
        _train_len = len(train)
    return _train_len

def get_test_ds_plain():
    """Plain (no transform) test set for sampling a raw image."""
    global _test_ds, _test_len
    if _test_ds is None:
        _test_ds = datasets.FashionMNIST(root="data", train=False, download=True)
        _test_len = len(_test_ds)
    return _test_ds

# ----------------------------
# Preprocess & helpers
# ----------------------------
inference_tf = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
])

@torch.inference_mode()
def predict_pil(img: Image.Image):
    """Returns (pred_label, pred_conf_float, top3_list[(label, prob_float)])"""
    x = inference_tf(img).unsqueeze(0).to(DEVICE)
    probs = torch.softmax(model(x), dim=1).squeeze(0)  # [10]
    idx = int(probs.argmax().item())
    conf = float(probs[idx].item())
    top3p, top3i = torch.topk(probs, k=3)
    top3 = [(CLASS_NAMES[int(i)], float(p)) for p, i in zip(top3p.tolist(), top3i.tolist())]
    return CLASS_NAMES[idx], conf, top3

def pil_to_data_url(img: Image.Image, fmt="PNG", display_size=256):
    """Upscale only for display so 28x28 looks crisp; keep model input at 28x28."""
    disp = img.convert("RGB").resize((display_size, display_size), Image.NEAREST)
    buf = io.BytesIO()
    disp.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/{fmt.lower()};base64,{b64}"

# ----------------------------
# Training meta (optional sidecar JSON next to weights)
# ----------------------------
META_PATH = Path(MODEL_PATH + ".meta.json")

def load_training_meta():
    """
    Optional JSON with keys:
      epochs, final_train_loss, final_val_loss, optimizer, loss_fn, lr, trained_at
    """
    if META_PATH.exists():
        try:
            with open(META_PATH, "r", encoding="utf-8") as f:
                m = json.load(f)
            return {
                "epochs": m.get("epochs"),
                "final_train_loss": m.get("final_train_loss"),
                "final_val_loss": m.get("final_val_loss"),
                "optimizer": m.get("optimizer"),
                "loss_fn": m.get("loss_fn"),
                "lr": m.get("lr"),
                "trained_at": m.get("trained_at"),
            }
        except Exception as e:
            print(f"[WARN] Failed to read meta: {e}")
    # sensible defaults if meta missing
    return {
        "epochs": None,
        "final_train_loss": None,
        "final_val_loss": None,
        "optimizer": "Adam",
        "loss_fn": "CrossEntropyLoss",
        "lr": None,
        "trained_at": None,
    }

# ----------------------------
# Metrics (computed once & cached)
# ----------------------------
_metrics_cache = None

def compute_metrics():
    """Evaluate once on test set and cache summary metrics for the UI."""
    global _metrics_cache
    if _metrics_cache is not None:
        return _metrics_cache

    train_size = get_train_len()
    test_plain = get_test_ds_plain()  # no transform
    test_size = len(test_plain)
    num_classes = len(CLASS_NAMES)
    num_params = sum(p.numel() for p in model.parameters())
    weights_mb = f"{os.path.getsize(MODEL_PATH)/1e6:.2f}" if os.path.exists(MODEL_PATH) else "—"
    device_name = "CUDA" if DEVICE == "cuda" else "CPU"

    # Build transformed test set & loader for evaluation
    test_eval = datasets.FashionMNIST(
        root="data", train=False, download=True,
        transform=transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ])
    )
    loader = DataLoader(
        test_eval, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=PIN_MEMORY
    )

    criterion = nn.CrossEntropyLoss()
    total = 0
    correct = 0
    loss_sum = 0.0
    all_preds, all_labels = [], []

    with torch.inference_mode():
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            logits = model(x)
            loss = criterion(logits, y)
            loss_sum += loss.item() * y.size(0)

            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)

            all_preds.extend(pred.detach().cpu().tolist())
            all_labels.extend(y.detach().cpu().tolist())

    test_acc = (correct / total) * 100.0
    test_loss = loss_sum / total
    precision = precision_score(all_labels, all_preds, average="macro") * 100.0
    recall = recall_score(all_labels, all_preds, average="macro") * 100.0
    f1 = f1_score(all_labels, all_preds, average="macro") * 100.0

    meta = load_training_meta()

    _metrics_cache = {
        # dataset / model facts
        "train_size": train_size,
        "test_size": test_size,
        "num_classes": num_classes,
        "num_params": f"{num_params:,}",
        "weights_mb": weights_mb,
        "device": device_name,
        # eval
        "test_acc": f"{test_acc:.2f}",
        "test_loss": f"{test_loss:.4f}",
        "precision": f"{precision:.2f}",
        "recall": f"{recall:.2f}",
        "f1": f"{f1:.2f}",
        # training meta (optional; defaults if missing)
        "epochs": meta["epochs"] if meta["epochs"] is not None else "—",
        "final_train_loss": meta["final_train_loss"] if meta["final_train_loss"] is not None else "—",
        "final_val_loss": meta["final_val_loss"] if meta["final_val_loss"] is not None else "—",
        "optimizer": meta["optimizer"] or "Adam",
        "loss_fn": meta["loss_fn"] or "CrossEntropyLoss",
        "lr": meta["lr"] if meta["lr"] is not None else "—",
        "trained_at": meta["trained_at"] or "—",
    }
    return _metrics_cache

def render_page(img_pil, truth_idx):
    pred, conf, top3 = predict_pil(img_pil)
    return render_template(
        "index.html",
        metrics=compute_metrics(),
        image_data=pil_to_data_url(img_pil, display_size=256),
        pred_label=pred,
        pred_conf=f"{conf*100:.2f}%",
        top3=top3,
        truth_label=CLASS_NAMES[truth_idx],
    )

# ----------------------------
# Single-page route (no upload)
# ----------------------------
@app.route("/", methods=["GET"])
def index():
    ds = get_test_ds_plain()
    i = random.randrange(len(ds))
    img, y = ds[i]
    # Ensure PIL image (FashionMNIST returns PIL when no transform, but guard anyway)
    if not isinstance(img, Image.Image):
        import numpy as np, torch as T
        if isinstance(img, T.Tensor):
            arr = (img.squeeze().numpy() * 255).astype("uint8")
            img = Image.fromarray(arr, mode="L")
    return render_page(img, y)

if __name__ == "__main__":
    # If scikit-learn is missing, give a helpful error before starting
    try:
        _ = precision_score
    except Exception:
        print("[ERROR] scikit-learn is required: pip install scikit-learn")
        raise
    app.run(host="127.0.0.1", port=5000, debug=True)
