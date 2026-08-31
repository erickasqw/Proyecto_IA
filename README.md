# Reconocimiento de emociones

El proyecto clasifica siete emociones: Angry, Disgust, Fear, Happy, Sad, Surprise y Neutral.

## Verificacion del estado actual

- Hay 52.207 imagenes en `data_raw` (FER, CK+ y RAF).
- El repositorio no contenia checkpoint ni historial de entrenamiento, por lo que no era posible afirmar que el modelo ya se hubiera entrenado.
- `train.py` usa PyTorch: ResNet-18 con pesos ImageNet, cabeza de 7 clases, AdamW, ReduceLROnPlateau y early stopping.
- Ahora el entrenamiento guarda `artifacts/best_emotion_model.pth` y `artifacts/training_history.json`, y la evaluacion guarda la matriz en `artifacts/evaluation/confusion_matrix.png`.

## Entrenamiento TensorFlow/Keras

`train_tensorflow.py` contiene el flujo equivalente en TensorFlow:

1. Descubre las imagenes etiquetadas de FER, CK+ y RAF.
2. Mezcla y separa 80% para entrenamiento y 10% para validacion.
3. Redimensiona a 224x224, normaliza a `[0, 1]` y aplica volteo horizontal al entrenamiento.
4. Usa MobileNetV2 preentrenada en ImageNet como extractor congelado.
5. Entrena una capa softmax de 7 clases con Adam y `sparse_categorical_crossentropy`.
6. Detiene temprano y restaura los mejores pesos.
7. Guarda `artifacts/tensorflow/emotion_model.keras` y `artifacts/tensorflow/training_history.json`.

Ejecutar con un Python funcional:

```powershell
python train_tensorflow.py
```

Para el flujo PyTorch:

```powershell
python main.py
```

## Interfaces web

Instalar dependencias y arrancar:

```powershell
python -m pip install -r requirements.txt
python web_server.py
```

Abrir:

- `http://127.0.0.1:5000/` muestra el grafico del entrenamiento y el mejor accuracy de validacion.
- `http://127.0.0.1:5000/camera` activa la camara, detecta todos los rostros visibles y muestra una etiqueta de emocion por cada rostro.

La camara usa el checkpoint PyTorch cuando `artifacts/best_emotion_model.pth` existe. Si aun no se ha entrenado, sigue mostrando los recuadros de todos los rostros y avisa `Modelo no cargado`.

## Bloqueo detectado en este equipo

El entorno `env` fue creado con Python 3.12.8, pero su ejecutable original `C:\Program Files\Python312\python.exe` ya no existe. Hay Python 3.10 disponible, pero no tiene las dependencias instaladas. Se debe seleccionar un intérprete instalado y ejecutar la instalacion de `requirements.txt` antes de entrenar o iniciar la web.

En Windows con la RTX 3050, `requirements.txt` fija PyTorch con CUDA 12.6. Comprueba que la GPU este activa con:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
