"""Image loading and preprocessing shared by the web app and the API.

The model was trained in the notebook on images read with cv2.imread, which
returns BGR channel order, resized with cv2.resize to 224x224 and fed as raw
0-255 values (no normalisation). Inference has to reproduce exactly that,
otherwise red-dominant fundus photos reach the network with red and blue swapped.

Only NumPy and Pillow are used here (OpenCV alone is ~120 MB, too heavy for
serverless hosting); resize_linear() reproduces cv2.INTER_LINEAR.
"""

import io

import numpy as np
from PIL import Image, ImageFilter, ImageOps, UnidentifiedImageError

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


def to_gray(rgb: np.ndarray) -> np.ndarray:
    # Same luma weights as cv2.COLOR_RGB2GRAY.
    return rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114


def resize_linear(img: np.ndarray, width: int, height: int) -> np.ndarray:
    """Bilinear resize matching cv2.resize(..., interpolation=cv2.INTER_LINEAR):
    half-pixel centres, edge clamping and no anti-aliasing when shrinking."""
    src_h, src_w = img.shape[:2]

    def axis(dst, src):
        pos = (np.arange(dst) + 0.5) * (src / dst) - 0.5
        pos = np.clip(pos, 0, None)
        lo = np.minimum(np.floor(pos).astype(int), src - 1)
        hi = np.minimum(lo + 1, src - 1)
        return lo, hi, (pos - lo).astype(np.float32)

    y0, y1, fy = axis(height, src_h)
    x0, x1, fx = axis(width, src_w)
    src = img.astype(np.float32)
    fx = fx[np.newaxis, :, np.newaxis] if img.ndim == 3 else fx[np.newaxis, :]
    fy = fy[:, np.newaxis, np.newaxis] if img.ndim == 3 else fy[:, np.newaxis]

    top = src[y0][:, x0] * (1 - fx) + src[y0][:, x1] * fx
    bottom = src[y1][:, x0] * (1 - fx) + src[y1][:, x1] * fx
    out = top * (1 - fy) + bottom * fy
    return np.clip(np.rint(out), 0, 255).astype(np.uint8) if img.dtype == np.uint8 else out


def crop_to_fundus(rgb: np.ndarray, threshold: int = 12) -> np.ndarray:
    """Trim the black frame around the retinal disc and pad the result to a square.

    The ODIR preprocessed images the model learned from are tight square crops,
    while raw camera exports are usually wide with large black margins.
    """
    mask = to_gray(rgb) > threshold
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
    resized = resize_linear(rgb, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)
    bgr = resized[..., ::-1]
    return np.ascontiguousarray(bgr, dtype=np.float32)[np.newaxis, ...]


def to_display(rgb: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    scale = DISPLAY_SIZE / max(h, w)
    if scale >= 1:
        return rgb
    size = (round(w * scale), round(h * scale))
    return np.asarray(Image.fromarray(rgb).resize(size, Image.Resampling.LANCZOS))


def quality_warnings(original: np.ndarray, fundus: np.ndarray) -> list[str]:
    warnings = []
    if min(original.shape[:2]) < MODEL_INPUT_SIZE:
        warnings.append(
            "Resolution is below the model's 224x224 input, so fine detail such as "
            "microaneurysms may be lost."
        )

    gray = to_gray(fundus)
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


def blur(values: np.ndarray, radius: float) -> np.ndarray:
    """Gaussian blur of a [0, 1] float map, via Pillow."""
    img = Image.fromarray(np.uint8(np.clip(values, 0, 1) * 255))
    return np.asarray(img.filter(ImageFilter.GaussianBlur(radius)), dtype=np.float32) / 255


def turbo(values: np.ndarray) -> np.ndarray:
    """Google's Turbo colormap (polynomial approximation), [0, 1] -> RGB uint8."""
    x = np.clip(values, 0, 1)[..., np.newaxis]
    powers = np.concatenate([np.ones_like(x), x, x ** 2, x ** 3, x ** 4, x ** 5], axis=-1)
    coeffs = np.array([
        [0.13572138, 0.09140261, 0.10667330],
        [4.61539260, 2.19418839, 12.64194608],
        [-42.66032258, 4.84296658, -60.58204836],
        [132.13108234, -14.18503333, 110.36276771],
        [-152.94239396, 4.27729857, -89.90310912],
        [59.28637943, 2.82956604, 27.34824973],
    ])
    return np.uint8(np.clip(powers @ coeffs, 0, 1) * 255)


def overlay_heatmap(rgb: np.ndarray, cam: np.ndarray, strength: float = 0.65) -> np.ndarray:
    """Blend a [0, 1] activation map over the image, keeping cold areas untouched."""
    h, w = rgb.shape[:2]
    cam_img = Image.fromarray(cam.astype(np.float32), mode="F").resize((w, h), Image.Resampling.BICUBIC)
    cam = np.clip(np.asarray(cam_img), 0, 1)
    alpha = (strength * cam ** 0.8)[..., np.newaxis]
    blended = rgb.astype(np.float32) * (1 - alpha) + turbo(cam).astype(np.float32) * alpha
    return np.clip(blended, 0, 255).astype(np.uint8)
