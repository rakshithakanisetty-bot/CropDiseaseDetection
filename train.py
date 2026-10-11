import pandas as pd
import numpy as np
import tensorflow as tf

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers, models
from PIL import Image

# Dataset location
dataset_path = Path("dataset/Crop_Disease_Images")
csv_path = dataset_path / "annotations.csv"

print("Loading dataset...")

if not csv_path.exists():
    raise FileNotFoundError(
        f"Cannot find {csv_path}. Check the dataset folder."
    )

data = pd.read_csv(csv_path)

# Keep rows with usable image paths and labels
data["image_path"] = data["image_file"].apply(
    lambda p: dataset_path / str(p)
)
data = data[
    data["image_path"].apply(lambda p: p.is_file())
].copy()

data["crop"] = data["crop"].fillna("Unknown").astype(str)
data["diagnosis"] = data["diagnosis"].fillna("Unknown").astype(str)

# Predict a crop-and-diagnosis combination
data["label"] = data["crop"] + " - " + data["diagnosis"]

# Keep only categories with at least 5 photos
counts = data["label"].value_counts()
data = data[data["label"].isin(counts[counts >= 5].index)].copy()

if len(data) < 10 or data["label"].nunique() < 2:
    raise ValueError(
        "Not enough images with repeated labels to train. "
        "Check the dataset and label counts."
    )

print("Usable images:", len(data))
print("Categories:", data["label"].nunique())

# Convert category names to numeric labels
encoder = LabelEncoder()
data["target"] = encoder.fit_transform(data["label"])

# Load and resize images
images = []
targets = []

for _, row in data.iterrows():
    try:
        with Image.open(row["image_path"]) as img:
            img = img.convert("RGB").resize((128, 128))
            images.append(np.asarray(img, dtype=np.float32))
            targets.append(row["target"])
    except Exception as error:
        print("Skipping unreadable image:", row["image_path"])

X = np.array(images) / 255.0
y = np.array(targets)

if len(X) < 10 or len(np.unique(y)) < 2:
    raise ValueError("Too few readable images to train the model.")

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training images:", len(X_train))
print("Validation images:", len(X_val))

# Build a basic convolutional neural network
model = models.Sequential([
    layers.Input(shape=(128, 128, 3)),
    layers.Conv2D(16, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(len(encoder.classes_), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("Starting training...")

model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=8,
    batch_size=16
)

# Save the trained model and its class names
Path("model").mkdir(exist_ok=True)
model.save("model/crop_disease_model.keras")
np.save("model/class_names.npy", encoder.classes_)

print("\nTraining finished!")
print("Model saved to model/crop_disease_model.keras")