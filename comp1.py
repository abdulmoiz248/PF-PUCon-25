import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import median_absolute_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Load data
df = pd.read_csv("train.csv")

# Drop unnecessary columns
df = df.drop(columns=['id'])

# Separate features and target
X = df.drop(columns=['Hardness']).values
y = df['Hardness'].values

# Split into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Build deep neural network model
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)
])

# Compile model
model.compile(optimizer='adam', loss='mae')

# Train model with early stopping
model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=64,
    callbacks=[keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)],
    verbose=1
)

# Evaluate with Median Absolute Error
y_pred = model.predict(X_val).flatten()
medae = median_absolute_error(y_val, y_pred)
print("Median Absolute Error:", medae)