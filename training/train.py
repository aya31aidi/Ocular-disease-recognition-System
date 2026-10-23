"""Reproducible training script for the ocular disease classifier.

Same architecture and input contract as the notebook (VGG19 backbone, 224x224
BGR images with raw 0-255 pixels, eight classes in the same order), so the
resulting .h5 file is a drop-in replacement for the web app. Differences:

* every fundus image is used once, labelled from its own eye's keywords;
  the notebook built class lists from patient-level rows, so the same photo
  could end up in both the training and the test split
* train/validation/test are split by patient, never by image
* class weights counter the imbalance (diabetic retinopathy ~5x hypertension)
* light augmentation, early stopping and learning-rate decay against overfitting
* a metrics file with per-class precision/recall and a confusion matrix

Usage:
    python training/train.py --data odir5k --out models
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
# Keras progress bars use Unicode blocks that the default Windows console codepage cannot print.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight

IMAGE_SIZE = 224
CLASS_NAMES = ["Normal", "Cataract", "Diabetic retinopathy", "Glaucoma",
               "Hypertension", "Myopia", "AMD", "Other"]

# Checked in this order, mirroring process_dataset() in the notebook.
KEYWORD_RULES = [
    (1, "cataract"),
    (2, "non proliferative retinopathy"),
    (3, "glaucoma"),
    (4, "hypertensive"),
    (5, "myopia"),
    (6, "macular degeneration"),
    (7, "drusen"),
]


def label_for(keywords: str) -> int | None:
    keywords = keywords.lower()
    if keywords.strip() == "normal fundus":
        return 0
    matches = {label for label, term in KEYWORD_RULES if term in keywords}
    # Images that match several classes are ambiguous for a single-label model.
    return matches.pop() if len(matches) == 1 else None


def build_index(data_dir: Path, normal_cap: int, seed: int) -> pd.DataFrame:
    df = pd.read_csv(data_dir / "full_df.csv")
    rows = []
    for _, r in df.iterrows():
        for side in ("Left", "Right"):
            rows.append({
                "patient": r["ID"],
                "filename": r[f"{side}-Fundus"],
                "label": label_for(str(r[f"{side}-Diagnostic Keywords"])),
            })
    index = (pd.DataFrame(rows)
             .dropna(subset=["label"])
             .drop_duplicates(subset="filename")
             .astype({"label": int}))
    index = index[index["filename"].map(lambda f: (data_dir / "preprocessed_images" / f).exists())]

    normal = index[index.label == 0]
    if len(normal) > normal_cap:
        index = pd.concat([index[index.label != 0], normal.sample(normal_cap, random_state=seed)])
    return index.reset_index(drop=True)


def patient_split(index: pd.DataFrame, seed: int):
    outer = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
    train_idx, rest_idx = next(outer.split(index, groups=index.patient))
    rest = index.iloc[rest_idx]
    inner = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    val_idx, test_idx = next(inner.split(rest, groups=rest.patient))
    return index.iloc[train_idx], rest.iloc[val_idx], rest.iloc[test_idx]


def make_dataset(frame: pd.DataFrame, image_dir: Path, batch: int, training: bool):
    paths = [str(image_dir / f) for f in frame.filename]
    labels = tf.one_hot(frame.label.values, len(CLASS_NAMES))

    def load(path, label):
        img = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
        img = tf.image.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        img = tf.reverse(img, axis=[-1])  # RGB -> BGR, matching cv2.imread in the notebook and the app
        return img, label

    def augment(img, label):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, 20.0)
        img = tf.image.random_contrast(img, 0.85, 1.15)
        return tf.clip_by_value(img, 0.0, 255.0), label

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(paths), reshuffle_each_iteration=True)
    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)


def build_model(backbone_weights: str | None):
    vgg = tf.keras.applications.VGG19(weights=backbone_weights, include_top=False,
                                      input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    vgg.trainable = False
    model = tf.keras.Sequential([
        vgg,
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(len(CLASS_NAMES), activation="softmax"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True, help="folder with full_df.csv and preprocessed_images/")
    parser.add_argument("--out", type=Path, default=Path("models"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--normal-cap", type=int, default=700, help="downsample normal images (notebook used 700)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backbone-weights", default="imagenet", help="'imagenet' or 'none' (smoke tests)")
    args = parser.parse_args()

    tf.keras.utils.set_random_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    image_dir = args.data / "preprocessed_images"

    index = build_index(args.data, args.normal_cap, args.seed)
    train, val, test = patient_split(index, args.seed)
    assert not set(train.patient) & set(test.patient), "patient leakage between train and test"
    print(f"images: {len(index)}  train {len(train)} / val {len(val)} / test {len(test)}")
    print(index.label.value_counts().sort_index().rename(dict(enumerate(CLASS_NAMES))).to_string())

    weights = compute_class_weight("balanced", classes=np.arange(len(CLASS_NAMES)), y=train.label)
    class_weight = {i: float(w) for i, w in enumerate(weights)}

    model = build_model(None if args.backbone_weights == "none" else args.backbone_weights)
    model_path = args.out / "ocular_disease_model.h5"
    history = model.fit(
        make_dataset(train, image_dir, args.batch, training=True),
        validation_data=make_dataset(val, image_dir, args.batch, training=False),
        epochs=args.epochs,
        class_weight=class_weight,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, min_lr=1e-5),
        ],
    )
    model.save(model_path)

    test_ds = make_dataset(test, image_dir, args.batch, training=False)
    probs = model.predict(test_ds, verbose=0)
    y_pred = probs.argmax(axis=1)
    y_true = test.label.values
    labels = list(range(len(CLASS_NAMES)))
    report = classification_report(y_true, y_pred, labels=labels, target_names=CLASS_NAMES,
                                   output_dict=True, zero_division=0)
    metrics = {
        "test_accuracy": float((y_pred == y_true).mean()),
        "split": {"train": len(train), "validation": len(val), "test": len(test)},
        "class_weight": class_weight,
        "history": {k: [float(v) for v in vals] for k, vals in history.history.items()},
        "report": report,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "class_names": CLASS_NAMES,
    }
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(classification_report(y_true, y_pred, labels=labels, target_names=CLASS_NAMES, zero_division=0))
    print(f"saved {model_path} and metrics.json")


if __name__ == "__main__":
    main()
