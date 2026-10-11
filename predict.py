import numpy as np
import tensorflow as tf
from PIL import Image
from pathlib import Path

# Load the trained model
model_path = Path("model/crop_disease_model.keras")
labels_path = Path("model/class_names.npy")

if not model_path.exists() or not labels_path.exists():
    raise FileNotFoundError(
        "Model files not found. Check the model folder."
    )

model = tf.keras.models.load_model(model_path)
class_names = np.load(labels_path, allow_pickle=True)

# Put your test plant photo here
image_path = Path("test_leaf.jpg")

if not image_path.exists():
    print("Please put a plant photo in this project folder")
    print("and name it test_leaf.jpg")
else:
    image = Image.open(image_path).convert("RGB")
    image = image.resize((128, 128))

    image_array = np.array(image, dtype=np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    predictions = model.predict(image_array, verbose=0)[0]
    best_index = int(np.argmax(predictions))

    print("\nPredicted category:", class_names[best_index])
    print("Model score:", round(float(predictions[best_index]) * 100, 2), "%")
    print("\nNote: This is an experimental prediction, not a confirmed diagnosis.")