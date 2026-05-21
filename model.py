import json
import os
from itertools import product

import numpy as np
from keras._tf_keras.keras import Model
from keras._tf_keras.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras._tf_keras.keras.layers import (
    Attention,
    BatchNormalization,
    Concatenate,
    Dense,
    Dropout,
    GRU,
    Input,
)
from keras._tf_keras.keras.optimizers import Adam
from keras._tf_keras.keras.utils import to_categorical
from sklearn import metrics
from sklearn.model_selection import train_test_split

from label_registry import (
    DEFAULT_FRAMES,
    DEFAULT_SEQUENCES,
    LabelEntry,
    class_to_data_dir,
    labels_with_complete_data,
    load_labels,
)

PATH = os.path.join("data")
SEQUENCES = DEFAULT_SEQUENCES
FRAMES = DEFAULT_FRAMES
SPLITS_PATH = "dataset/splits.json"


def labels_with_data(labels_meta: list[LabelEntry]) -> list[LabelEntry]:
    """Keep only classes that have been fully recorded under data/<data_dir>/."""
    ready = labels_with_complete_data(labels_meta)
    skipped = len(labels_meta) - len(ready)
    if skipped:
        print(f"Skipping {skipped} label(s) with no or incomplete data.")
    return ready


def create_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    x = GRU(64, return_sequences=True, activation="tanh", kernel_regularizer="l2")(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = GRU(128, return_sequences=True, activation="tanh", kernel_regularizer="l2")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    attention = Attention()([x, x])
    x = Concatenate()([x, attention])
    x = GRU(64, return_sequences=False, activation="tanh", kernel_regularizer="l2")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Dense(64, activation="tanh", kernel_regularizer="l2")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    outputs = Dense(num_classes, activation="softmax")(x)
    return Model(inputs=inputs, outputs=outputs)


def load_dataset():
    labels_meta = labels_with_data(load_labels())
    if not labels_meta:
        raise RuntimeError("No classes with complete training data found under data/.")
    actions = np.array([label.class_id for label in labels_meta])
    class_dir_map = class_to_data_dir(labels_meta)
    label_map = {label: num for num, label in enumerate(actions)}

    landmarks, labels = [], []
    for action, sequence in product(actions, range(SEQUENCES)):
        temp = []
        data_dir = class_dir_map[action]
        for frame in range(FRAMES):
            npy = np.load(os.path.join(PATH, data_dir, str(sequence), f"{frame}.npy"))
            temp.append(npy)
        landmarks.append(temp)
        labels.append(label_map[action])

    x = np.array(landmarks)
    y = to_categorical(labels).astype(int)
    return x, y, actions


def load_splits():
    if not os.path.exists(SPLITS_PATH):
        return {}
    with open(SPLITS_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


def main():
    x, y, actions = load_dataset()
    _ = load_splits()  # Placeholder for explicit split support in next iterations.

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.15, random_state=34, stratify=y
    )

    model = create_model(input_shape=(FRAMES, 126), num_classes=len(actions))
    optimizer = Adam(learning_rate=0.0005)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["categorical_accuracy"],
    )

    early_stopping = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7, min_lr=0.00001)

    model.fit(
        x_train,
        y_train,
        epochs=150,
        batch_size=16,
        validation_split=0.2,
        callbacks=[early_stopping, reduce_lr],
        verbose=1,
    )

    model.save("my_model")

    predictions = np.argmax(model.predict(x_test), axis=1)
    test_labels = np.argmax(y_test, axis=1)
    accuracy = metrics.accuracy_score(test_labels, predictions)
    print(f"\nTest Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(metrics.classification_report(test_labels, predictions, target_names=actions))


if __name__ == "__main__":
    main()