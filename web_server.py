"""Local web UI: training dashboard and multi-face emotion camera."""

from pathlib import Path
import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image
import torch
from torchvision import transforms

from dataset import CLASS_NAMES
from model import EmotionResNet18


ROOT = Path(__file__).resolve().parent
DEFAULT_FACE_DETECTION = {
    "scaleFactor": 1.15,
    "minNeighbors": 6,
    "minSize": (70, 70),
    "maxSize": (500, 500),
}
MIN_EMOTION_CONFIDENCE = 0.55

app = Flask(__name__)
face_detector = cv2.CascadeClassifier(str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"))
checkpoint_path = ROOT / "artifacts" / "best_emotion_model.pth"
emotion_model = None
preprocess = transforms.Compose([
    transforms.ToPILImage(), transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def should_accept_prediction(emotion, confidence):
    return isinstance(emotion, str) and confidence >= MIN_EMOTION_CONFIDENCE


@app.get("/")
def dashboard():
    return send_from_directory(ROOT / "web", "dashboard.html")


@app.get("/camera")
def camera():
    return send_from_directory(ROOT / "web", "camera.html")


@app.get("/<path:filename>")
def static_files(filename):
    return send_from_directory(ROOT / "web", filename)


@app.get("/artifacts/<path:filename>")
def artifacts(filename):
    return send_from_directory(ROOT / "artifacts", filename)


def predict_emotion(face):
    global emotion_model
    if emotion_model is None:
        if not checkpoint_path.is_file():
            return "Modelo no cargado", 0.0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        emotion_model = EmotionResNet18().to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        emotion_model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
        emotion_model.eval()
    device = next(emotion_model.parameters()).device
    with torch.inference_mode():
        probabilities = torch.softmax(emotion_model(preprocess(face).unsqueeze(0).to(device)), dim=1)[0]
    index = int(probabilities.argmax())
    return CLASS_NAMES[index], float(probabilities[index])


@app.post("/api/detect")
def detect():
    payload = request.files.get("image")
    if payload is None:
        return jsonify({"error": "image is required"}), 400
    image = cv2.imdecode(np.frombuffer(payload.read(), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return jsonify({"error": "invalid image"}), 400
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=DEFAULT_FACE_DETECTION["scaleFactor"],
        minNeighbors=DEFAULT_FACE_DETECTION["minNeighbors"],
        minSize=DEFAULT_FACE_DETECTION["minSize"],
        maxSize=DEFAULT_FACE_DETECTION["maxSize"],
    )
    results = []
    for x, y, width, height in faces:
        crop = image[max(0, y):min(image.shape[0], y + height), max(0, x):min(image.shape[1], x + width)]
        if crop.size == 0:
            continue
        emotion, confidence = predict_emotion(crop)
        if not should_accept_prediction(emotion, confidence):
            continue
        results.append({"x": int(x), "y": int(y), "width": int(width), "height": int(height),
                        "emotion": emotion, "confidence": confidence})
    return jsonify({"faces": results})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)