"""Facts about the trained model, taken from the outputs of
Ocular_Disease_Recognition_final_version.ipynb so the site never overstates them."""

MODEL_CARD = {
    "name": "VGG19 transfer-learning classifier",
    "dataset": "ODIR-5K (Peking University, 2019)",
    "dataset_url": "https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k",
    "images": 5396,
    "split": {"train": 3777, "validation": 809, "test": 810},
    "input": "224 x 224 x 3, BGR, raw 0-255 pixels",
    "test_accuracy": 0.7099,
    "test_loss": 1.2767,
    "best_val_accuracy": 0.7478,
    "epochs": 10,
    "optimizer": "Adam (lr 1e-3), categorical cross-entropy, batch size 32",
    "params_total": 26_517_064,
    "params_trainable": 6_491_656,
    "layers": [
        {"name": "Input", "detail": "Fundus image", "shape": "224 x 224 x 3", "params": 0, "frozen": None},
        {"name": "VGG19", "detail": "16 conv layers, ImageNet weights", "shape": "7 x 7 x 512", "params": 20_024_384, "frozen": True},
        {"name": "Flatten", "detail": "", "shape": "25,088", "params": 0, "frozen": None},
        {"name": "Dense + ReLU", "detail": "256 units", "shape": "256", "params": 6_422_784, "frozen": False},
        {"name": "BatchNorm", "detail": "", "shape": "256", "params": 1_024, "frozen": False},
        {"name": "Dense + ReLU", "detail": "256 units", "shape": "256", "params": 65_792, "frozen": False},
        {"name": "BatchNorm", "detail": "", "shape": "256", "params": 1_024, "frozen": False},
        {"name": "Dense + Softmax", "detail": "8 classes", "shape": "8", "params": 2_056, "frozen": False},
    ],
    "history": {
        "accuracy":     [0.5314, 0.7630, 0.8517, 0.9023, 0.9208, 0.9367, 0.9449, 0.9481, 0.9481, 0.9473],
        "val_accuracy": [0.4574, 0.5760, 0.6675, 0.7095, 0.7157, 0.7478, 0.7429, 0.7182, 0.7281, 0.7318],
        "loss":         [1.4138, 0.7278, 0.4746, 0.3496, 0.2820, 0.2381, 0.1876, 0.1794, 0.1748, 0.1537],
        "val_loss":     [1.9731, 1.3915, 1.1867, 1.0898, 1.1349, 1.1288, 1.1072, 1.2394, 1.1496, 1.1153],
    },
    "limitations": [
        ("Overfitting after epoch 6",
         "Training accuracy keeps climbing to 95% while validation accuracy peaks at 74.8% and drifts down. "
         "Early stopping and data augmentation are the obvious next steps."),
        ("Possible duplicate images across splits",
         "full_df.csv lists every patient twice (one row per eye), and the class lists are built from "
         "patient-level keywords, so the same photo can land in both the training and the test set. "
         "The 71% test accuracy is therefore likely optimistic; training/train.py fixes this with "
         "de-duplication and a patient-level split."),
        ("Class imbalance",
         "Diabetic retinopathy has 1,818 images while hypertension has 374. Without class weights the "
         "model leans towards the majority class."),
        ("Single-label output",
         "Real patients often have more than one condition, and ODIR itself is multi-label. "
         "This model always picks exactly one of eight classes."),
        ("One dataset, one population",
         "ODIR-5K comes from a limited set of Chinese hospitals and cameras. Performance on other "
         "devices and populations has not been measured."),
    ],
}


def line_chart(series, y_min, y_max, width=560, height=240, pad_left=44, pad_bottom=28, pad_top=12, pad_right=12):
    """Pre-compute SVG coordinates so the template stays free of maths."""
    n = len(series[0][1])
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom

    def x(i):
        return pad_left + plot_w * i / (n - 1)

    def y(v):
        return pad_top + plot_h * (1 - (v - y_min) / (y_max - y_min))

    ticks = 5
    y_ticks = [
        {"y": round(y(y_min + (y_max - y_min) * t / (ticks - 1)), 1),
         "label": f"{y_min + (y_max - y_min) * t / (ticks - 1):.1f}"}
        for t in range(ticks)
    ]
    x_ticks = [{"x": round(x(i), 1), "label": i + 1} for i in range(n)]
    lines = [
        {
            "name": name,
            "points": " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(values)),
            "dots": [{"x": round(x(i), 1), "y": round(y(v), 1), "value": v, "epoch": i + 1}
                     for i, v in enumerate(values)],
            "last": values[-1],
        }
        for name, values in series
    ]
    return {
        "width": width, "height": height,
        "left": pad_left, "right": width - pad_right,
        "top": pad_top, "bottom": height - pad_bottom,
        "y_ticks": y_ticks, "x_ticks": x_ticks, "lines": lines,
    }
