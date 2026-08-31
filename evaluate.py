"""Test evaluation, reports, confusion matrix, and single-image inference."""

from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torchvision import transforms
from PIL import Image

from dataset import CLASS_NAMES


def load_model(model, checkpoint_path="best_emotion_model.pth", device=None):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state)
    return model.to(device).eval(), device


def evaluate_model(model, test_loader, checkpoint_path="best_emotion_model.pth", output_dir="evaluation"):
    model, device = load_model(model, checkpoint_path)
    labels, predictions = [], []
    with torch.inference_mode():
        for images, targets in test_loader:
            output = model(images.to(device)).argmax(1).cpu().numpy()
            predictions.extend(output.tolist())
            labels.extend(targets.numpy().tolist())
    report = classification_report(labels, predictions, target_names=CLASS_NAMES, zero_division=0)
    print(report)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    matrix = confusion_matrix(labels, predictions, labels=range(len(CLASS_NAMES)))
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(output_path / "confusion_matrix.png", dpi=150)
    plt.close()
    return report, matrix


def predict_image(image_path, model, checkpoint_path="best_emotion_model.pth"):
    model, device = load_model(model, checkpoint_path)
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    preprocess = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    tensor = preprocess(Image.fromarray(np.uint8(image))).unsqueeze(0).to(device)
    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor), dim=1)[0]
    index = int(probabilities.argmax())
    result = CLASS_NAMES[index], float(probabilities[index])
    print(f"Predicted emotion: {result[0]} ({result[1]:.2%})")
    return result