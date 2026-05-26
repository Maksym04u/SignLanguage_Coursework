import json
import os
from dataclasses import asdict
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
    LANGUAGE_DIM,
    LANGUAGE_INDEX,
    LabelEntry,
    class_to_data_dir,
    language_from_data_dir,
    language_to_vector,
    labels_with_complete_data,
    load_labels,
)

PATH = os.path.join("data")
SEQUENCES = DEFAULT_SEQUENCES
FRAMES = DEFAULT_FRAMES
SPLITS_PATH = "dataset/splits.json"
MODEL_PATH = "my_model.h5"
MODEL_LABELS_PATH = "dataset/model_labels.json"


def labels_with_data(labels_meta: list[LabelEntry]) -> list[LabelEntry]:
    """Keep only classes that have been fully recorded under data/<data_dir>/."""
    ready = labels_with_complete_data(labels_meta)
    skipped = len(labels_meta) - len(ready)
    if skipped:
        print(f"Skipping {skipped} label(s) with no or incomplete data.")
    return ready


def create_model(input_shape, num_classes, language_dim: int = LANGUAGE_DIM):
    """
    Sequence branch processes hand motion; language one-hot is fused after the
    temporal GRU so identical poses can map to different classes per language.
    """
    seq_input = Input(shape=input_shape, name="sequence")
    x = GRU(64, return_sequences=True, activation="tanh", kernel_regularizer="l2")(seq_input)
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

    lang_input = Input(shape=(language_dim,), name="language")
    x = Concatenate()([x, lang_input])
    x = Dense(64, activation="tanh", kernel_regularizer="l2")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    outputs = Dense(num_classes, activation="softmax")(x)
    return Model(inputs=[seq_input, lang_input], outputs=outputs)


def load_dataset():
    labels_meta = labels_with_data(load_labels())
    if not labels_meta:
        raise RuntimeError("No classes with complete training data found under data/.")

    actions = np.array([label.class_id for label in labels_meta])
    class_dir_map = class_to_data_dir(labels_meta)
    label_map = {label: num for num, label in enumerate(actions)}

    # Language vector per class_id (from data_dir prefix: en_*, uk_*).
    lang_vector_by_class = {}
    for label in labels_meta:
        lang_code = language_from_data_dir(label.data_dir)
        if lang_code != label.language:
            print(
                f"Note: data_dir language {lang_code!r} != labels.json "
                f"{label.language!r} for {label.class_id}"
            )
        lang_vector_by_class[label.class_id] = language_to_vector(lang_code)

    landmarks, lang_vectors, labels = [], [], []
    for action, sequence in product(actions, range(SEQUENCES)):
        temp = []
        data_dir = class_dir_map[action]
        for frame in range(FRAMES):
            npy = np.load(os.path.join(PATH, data_dir, str(sequence), f"{frame}.npy"))
            temp.append(npy)
        landmarks.append(temp)
        lang_vectors.append(lang_vector_by_class[action])
        labels.append(label_map[action])

    x = np.array(landmarks, dtype=np.float32)
    lang = np.array(lang_vectors, dtype=np.float32)
    y = to_categorical(labels).astype(int)
    return x, lang, y, actions, labels_meta


def save_model_labels(labels_meta: list[LabelEntry], path: str = MODEL_LABELS_PATH) -> None:
    """Persist the exact class list/order used for this training run (for API inference)."""
    payload = {
        "version": 2,
        "model_path": MODEL_PATH,
        "language_dim": LANGUAGE_DIM,
        "language_index": LANGUAGE_INDEX,
        "labels": [asdict(entry) for entry in labels_meta],
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    print(f"Saved model label manifest ({len(labels_meta)} classes) to {path}")


def load_splits():
    if not os.path.exists(SPLITS_PATH):
        return {}
    with open(SPLITS_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


def main():
    x, lang, y, actions, labels_meta = load_dataset()
    _ = load_splits()

    x_train, x_test, lang_train, lang_test, y_train, y_test = train_test_split(
        x, lang, y, test_size=0.15, random_state=34, stratify=y
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
        [x_train, lang_train],
        y_train,
        epochs=150,
        batch_size=16,
        validation_split=0.2,
        callbacks=[early_stopping, reduce_lr],
        verbose=1,
    )

    model.save(MODEL_PATH)
    save_model_labels(labels_meta)

    predictions = np.argmax(model.predict([x_test, lang_test], verbose=0), axis=1)
    test_labels = np.argmax(y_test, axis=1)
    accuracy = metrics.accuracy_score(test_labels, predictions)
    print(f"\nTest Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(metrics.classification_report(test_labels, predictions, target_names=actions))


if __name__ == "__main__":
    main()
