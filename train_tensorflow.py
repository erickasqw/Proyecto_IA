"""TensorFlow/Keras training entry point for the same FER image folders.

Run this after selecting a working Python interpreter:
    python train_tensorflow.py
"""

from pathlib import Path
import json

import tensorflow as tf

from dataset import CLASS_NAMES
from utils_zip import extract_datasets


IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42


def _image_paths_and_labels(roots):
    aliases = {"anger": "Angry", "anger": "Angry", "angry": "Angry",
               "disgust": "Disgust", "fear": "Fear", "happy": "Happy",
               "happiness": "Happy", "sad": "Sad", "sadness": "Sad",
               "surprise": "Surprise", "neutral": "Neutral"}
    paths, labels = [], []
    for root_value in roots:
        for path in sorted(Path(root_value).rglob("*")):
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                continue
            label_name = next((aliases.get(part.lower()) for part in reversed(path.parts[:-1])
                               if part.lower() in aliases), None)
            if label_name is not None:
                paths.append(str(path))
                labels.append(CLASS_NAMES.index(label_name))
    if not paths:
        raise RuntimeError("No labeled images were found.")
    return paths, labels


def train_tensorflow(roots, output_dir="artifacts/tensorflow"):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths, labels = _image_paths_and_labels(roots)
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels)).shuffle(len(paths), seed=SEED)
    train_size = int(len(paths) * 0.8)
    validation_size = int(len(paths) * 0.1)
    train_data = dataset.take(train_size)
    validation_data = dataset.skip(train_size).take(validation_size)

    def load_image(path, label, training=False):
        image = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        image = tf.image.resize(image, IMAGE_SIZE) / 255.0
        if training:
            image = tf.image.random_flip_left_right(image)
        return image, label

    train_data = train_data.map(lambda path, label: load_image(path, label, True)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    validation_data = validation_data.map(load_image).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    model = tf.keras.Sequential([
        tf.keras.applications.MobileNetV2(input_shape=(*IMAGE_SIZE, 3), include_top=False,
                                           weights="imagenet"),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(len(CLASS_NAMES), activation="softmax"),
    ])
    model.layers[0].trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    history = model.fit(train_data, validation_data=validation_data, epochs=30,
                        callbacks=[tf.keras.callbacks.EarlyStopping(patience=7, restore_best_weights=True)])
    model.save(output / "emotion_model.keras")
    (output / "training_history.json").write_text(json.dumps({
        "framework": "TensorFlow/Keras", "classes": CLASS_NAMES,
        "epochs_completed": len(history.history["loss"]),
        "history": history.history,
    }, indent=2), encoding="utf-8")
    return history


if __name__ == "__main__":
    train_tensorflow(extract_datasets().values())