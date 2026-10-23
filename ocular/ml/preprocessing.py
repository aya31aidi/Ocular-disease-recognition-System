"""Image loading and preprocessing shared by the web app and the API.

The model was trained in the notebook on images read with cv2.imread, which
returns BGR channel order, resized with cv2.resize to 224x224 and fed as raw
0-255 values (no normalisation). Inference has to reproduce exactly that,
otherwise red-dominant fundus photos reach the network with red and blue swapped.
"""

import io

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

MODEL_INPUT_SIZE = 224
DISPLAY_SIZE = 512
MIN_SIDE = 64


class ImageError(ValueError):
    """Raised when an upload cannot be used for analysis."""


def load_rgb(data: bytes, allowed_formats: set[str]) -> np.ndarray:
    if not data:
        raise ImageError("The uploaded file is empty.")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise ImageError("The file could not be read as an image.") from exc

    if img.format not in allowed_formats:
        readable = ", ".join(sorted(allowed_formats))
        raise ImageError(f"Unsupported format {img.format or 'unknown'}. Accepted: {readable}.")

    # Phone photos often carry their rotation in EXIF rather than in the pixels.
    img = ImageOps.exif_transpose(img)
    # Handles palette, grayscale, 16-bit and RGBA inputs, which used to crash predict().
    img = img.convert("RGB")

    if min(img.size) < MIN_SIDE:
        raise ImageError(f"The image is too small ({img.width}x{img.height}px).")
    return np.asarray(img)


def crop_to_fundus(rgb: np.ndarray, threshold: int = 12) -> np.ndarray:
    """Trim the black frame around the retinal disc and pad the result to a square.

    The ODIR preprocessed images the model learned from are tight square crops,
    while raw camera exports are usually wide with large black margins.
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    mask = gray > threshold
    rows = np.flatnonzero(mask.mean(axis=1) > 0.02)
    cols = np.flatnonzero(mask.mean(axis=0) > 0.02)
    if rows.size == 0 or cols.size == 0:
        return rgb

    cropped = rgb[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    h, w = cropped.shape[:2]
    # Ignore crops that would throw away most of the picture; likely not a fundus frame.
    if h * w < 0.2 * rgb.shape[0] * rgb.shape[1]:
        return rgb

    side = max(h, w)
    square = np.zeros((side, side, 3), dtype=rgb.dtype)
    top, left = (side - h) // 2, (side - w) // 2
    square[top:top + h, left:left + w] = cropped
    return square


def to_model_input(rgb: np.ndarray) -> np.ndarray:
    resized = cv2.resize(rgb, (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    bgr = resized[..., ::-1]
    return np.ascontiguousarray(bgr, dtype=np.float32)[np.newaxis, ...]


def to_display(rgb: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    scale = DISPLAY_SIZE / max(h, w)
    if scale >= 1:
        return rgb
    return cv2.resize(rgb, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


def quality_warnings(original: np.ndarray, fundus: np.ndarray) -> list[str]:
    warnings = []
    if min(original.shape[:2]) < MODEL_INPUT_SIZE:
        warnings.append(
            "Resolution is below the model's 224x224 input, so fine detail such as "
            "microaneurysms may be lost."
        )

    gray = cv2.cvtColor(fundus, cv2.COLOR_RGB2GRAY)
    lit = gray > 12
    if lit.mean() < 0.05:
        warnings.append("The image is almost entirely black.")
        return warnings

    brightness = gray[lit].mean()
    if brightness < 35:
        warnings.append("The image is very dark; underexposed photos reduce reliability.")
    elif brightness > 215:
        warnings.append("The image is overexposed; washed-out areas hide retinal detail.")

    r, g, b = (fundus[..., i][lit].mean() for i in range(3))
    if not (r > g and r > 1.15 * b):
        warnings.append(
            "The colours do not look like a typical retinal fundus photograph. "
            "The model was only trained on fundus images, so this result may be meaningless."
        )
    return warnings


def encode_jpeg(rgb: np.ndarray, quality: int = 90) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


def overlay_heatmap(rgb: np.ndarray, cam: np.ndarray, strength: float = 0.65) -> np.ndarray:
    """Blend a [0, 1] activation map over the image, keeping cold areas untouched."""
    h, w = rgb.shape[:2]
    cam = cv2.resize(cam.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    cam = np.clip(cam, 0, 1)
    colored = cv2.applyColorMap(np.uint8(cam * 255), cv2.COLORMAP_TURBO)[..., ::-1]
    alpha = (strength * cam ** 0.8)[..., np.newaxis]
    blended = rgb.astype(np.float32) * (1 - alpha) + colored.astype(np.float32) * alpha
    return np.clip(blended, 0, 255).astype(np.uint8)
