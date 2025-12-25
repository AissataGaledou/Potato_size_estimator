# train.py — binary classification (Potato vs Non-Potato)

import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Load preprocessed data
X_train = np.load("processed_data/X_train.npy")
y_train = np.load("processed_data/y_train.npy")
X_val = np.load("processed_data/X_val.npy")
y_val = np.load("processed_data/y_val.npy")

# Build model (MobileNetV2)
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
preds = Dense(1, activation="sigmoid")(x)  # ✅ Binary output

model = Model(inputs=base_model.input, outputs=preds)

# Freeze base layers for initial training
for layer in base_model.layers:
    layer.trainable = False

# Compile model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="binary_crossentropy",  # ✅ For 2-class
    metrics=["accuracy"]
)

# Train model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=5,
    batch_size=32
)

# Save model
model.save("mobilenetv2_potato_binary.keras")
print("✅ Model trained and saved as 'mobilenetv2_potato_binary.keras'")

