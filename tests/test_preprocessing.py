import numpy as np
import pytest

from ocular.conditions import CLASS_KEYS
from ocular.ml.preprocessing import (ImageError, crop_to_fundus, load_rgb, quality_warnings,
                                     to_model_input)

from .conftest import make_fundus

FORMATS = {"JPEG", "PNG", "BMP", "TIFF", "WEBP"}


def test_model_input_is_bgr_raw_pixels():
    # The notebook trained on cv2.imread output, which is BGR in 0-255.
    rgb = np.zeros((300, 300, 3), np.uint8)
    rgb[..., 0] = 200  # red
    rgb[..., 2] = 40   # blue
    x = to_model_input(rgb)
    assert x.shape == (1, 224, 224, 3)
    assert x.dtype == np.float32
    assert x[0, 100, 100, 0] == 40   # blue first
    assert x[0, 100, 100, 2] == 200  # red last


@pytest.mark.parametrize("fmt,mode", [("PNG", "RGBA"), ("PNG", "L"), ("PNG", "P"), ("BMP", "RGB"), ("WEBP", "RGB")])
def test_load_handles_any_colour_mode(fmt, mode):
    rgb = load_rgb(make_fundus(fmt=fmt, mode=mode), FORMATS)
    assert rgb.ndim == 3 and rgb.shape[2] == 3


def test_load_rejects_non_images():
    with pytest.raises(ImageError):
        load_rgb(b"definitely not an image", FORMATS)


def test_load_rejects_empty_and_tiny():
    with pytest.raises(ImageError):
        load_rgb(b"", FORMATS)
    with pytest.raises(ImageError, match="too small"):
        load_rgb(make_fundus(32, 32), FORMATS)


def test_load_rejects_disallowed_format():
    with pytest.raises(ImageError, match="Unsupported"):
        load_rgb(make_fundus(fmt="PNG"), {"JPEG"})


def test_crop_removes_black_frame_and_squares():
    rgb = load_rgb(make_fundus(900, 500), FORMATS)
    cropped = crop_to_fundus(rgb)
    h, w = cropped.shape[:2]
    assert h == w
    assert w < 900


def test_crop_leaves_blank_images_alone():
    blank = np.zeros((200, 300, 3), np.uint8)
    assert crop_to_fundus(blank).shape == blank.shape


def test_quality_warnings():
    fundus = load_rgb(make_fundus(), FORMATS)
    assert quality_warnings(fundus, crop_to_fundus(fundus)) == []

    blue = np.zeros((400, 400, 3), np.uint8)
    blue[..., 2] = 180
    assert any("fundus" in w for w in quality_warnings(blue, blue))

    small = load_rgb(make_fundus(120, 120), FORMATS)
    assert any("Resolution" in w for w in quality_warnings(small, small))


def test_class_order_matches_training_notebook():
    assert CLASS_KEYS == [
        "normal", "cataract", "diabetic_retinopathy", "glaucoma",
        "hypertension", "myopia", "amd", "other",
    ]


def test_lfs_pointer_falls_back_to_demo_with_clear_reason(tmp_path):
    from ocular.ml.predictor import create_predictor

    pointer = tmp_path / "ocular_disease_model.h5"
    pointer.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 158110480\n")
    predictor = create_predictor({"ALLOWED_IMAGE_FORMATS": FORMATS, "DEMO_MODE": "auto", "MODEL_PATH": pointer})
    assert predictor.status["mode"] == "demo"
    assert "git lfs pull" in predictor.status["reason"]
