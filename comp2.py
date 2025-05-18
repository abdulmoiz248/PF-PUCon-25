import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models, regularizers
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import random
from scipy.ndimage import map_coordinates, gaussian_filter
from tensorflow.keras.callbacks import ReduceLROnPlateau

# Paths & Config
train_dir = r"datasets\comp2\Dataset\train"
test_dir = r"datasets\comp2\Dataset\Test"
IMG_SIZE = (128, 128)

import cv2
import numpy as np

def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, IMG_SIZE)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 1: Unsharp masking
    gaussian = cv2.GaussianBlur(gray, (9, 9), 10.0)
    sharpened = cv2.addWeighted(gray, 1.5, gaussian, -0.5, 0)

    # Step 2: Slight blur
    blurred = cv2.GaussianBlur(sharpened, (3, 3), 0)

    # Step 3: Adaptive thresholding
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )

    # Step 4: Morphological opening to remove noise
    kernel_open = np.ones((2, 2), np.uint8)
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)

    # Step 5: Remove small components (noise filtering)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(opened, connectivity=8)
    min_area = 50  # Adjust this to remove smaller blobs (increase if needed)

    cleaned = np.zeros_like(opened)
    for i in range(1, num_labels):  # Skip background
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == i] = 255

    return cleaned

# ========= Elastic Deformation =========
def elastic_transform(image, alpha=8, sigma=4):
    random_state = np.random.RandomState(None)
    shape = image.shape

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))

    distorted = map_coordinates(image, indices, order=1, mode='reflect')
    return distorted.reshape(shape)

# ========= Augmentation Function =========
def augment_image(image):
    rows, cols = image.shape

    # Rotation
    angle = random.uniform(-15, 15)
    M_rot = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1.0)
    image = cv2.warpAffine(image, M_rot, (cols, rows), borderMode=cv2.BORDER_REFLECT)

    # Shear
    shear_angle = np.deg2rad(random.uniform(-10, 10))
    M_shear = np.array([[1, np.tan(shear_angle), 0], [0, 1, 0]], dtype=np.float32)
    image = cv2.warpAffine(image, M_shear, (cols, rows), borderMode=cv2.BORDER_REFLECT)

    # Elastic deformation
    image = elastic_transform(image, alpha=8, sigma=4)

    return image

# ========= Load & Augment Training Data =========
X = []
y = []
labels = sorted(os.listdir(train_dir))
label_map = {label: idx for idx, label in enumerate(labels)}

for label in labels:
    folder_path = os.path.join(train_dir, label)
    for file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, file)
        processed = preprocess_image(img_path)

        # Original
        X.append(processed)
        y.append(label_map[label])

        # Augmented
        augmented = augment_image(processed)
        X.append(augmented)
        y.append(label_map[label])

# ========= Show Sample Preprocessed Images =========
plt.figure(figsize=(10, 4))
for i in range(4):
    plt.subplot(1, 4, i+1)
    plt.imshow(X[i], cmap='gray')
    plt.title(f"Label: {y[i]}")
    plt.axis('off')
plt.suptitle("Sample Preprocessed Images")
plt.tight_layout()
plt.show()

# ========= Prepare Data =========
X = np.array(X).reshape(-1, IMG_SIZE[0], IMG_SIZE[1], 1) / 255.0
y = to_categorical(y, num_classes=len(label_map))

# ========= Split Data =========
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# ========= CNN Model =========
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(1e-4), input_shape=(IMG_SIZE[0], IMG_SIZE[1], 1)),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),
    layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
    layers.Dropout(0.6),
    layers.Dense(len(label_map), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ========= Callbacks =========
reduce_lr = ReduceLROnPlateau(factor=0.5, patience=3)

# ========= Train Model =========
model.fit(X_train, y_train,
          validation_data=(X_val, y_val),
          epochs=25, batch_size=32,
          callbacks=[reduce_lr])

# ========= Save Model =========
model.save("palm_cnn_model_final_augmented.h5")
print("✅ Model saved as palm_cnn_model_final_augmented.h5")

# ========= Evaluate Model =========
val_preds = model.predict(X_val)
y_true = np.argmax(y_val, axis=1)
y_pred = np.argmax(val_preds, axis=1)

print("\n📊 Classification Report:\n")
print(classification_report(y_true, y_pred, target_names=label_map.keys()))
print("🧩 Confusion Matrix:\n")
print(confusion_matrix(y_true, y_pred))

# ========= Predict on Test Data =========
reverse_label_map = {v: k for k, v in label_map.items()}
predictions = []

for file in tqdm(sorted(os.listdir(test_dir))):
    img_path = os.path.join(test_dir, file)
    processed = preprocess_image(img_path)
    processed = np.expand_dims(processed, axis=(0, -1)) / 255.0
    pred = model.predict(processed)
    label = reverse_label_map[np.argmax(pred)]
    predictions.append((file.replace('.jpg', ''), label))

# ========= Save Submission =========
submission_df = pd.DataFrame(predictions, columns=["Id", "Owner"])
submission_df.to_csv("submission_augmented_no_zoom_brightness.csv", index=False)
print("✅ Saved submission_augmented_no_zoom_brightness.csv")