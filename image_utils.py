import cv2
from pathlib import Path

from config import IMAGE_SIZE, SUPPORTED_FORMATS


def load_image(image_path):
    """
    Load an image from the given path.
    """

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    if image_path.suffix.lower() not in SUPPORTED_FORMATS:
        raise ValueError("Unsupported image format.")

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError("Unable to read image.")

    return image


def convert_to_grayscale(image):
    """
    Convert BGR image into grayscale.
    """

    if image is None:
        raise ValueError("Invalid image.")

    if len(image.shape) == 2:
        return image

    return cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )


def resize_image(image, size=IMAGE_SIZE):
    """
    Resize image to the specified dimensions.
    """

    if image is None:
        raise ValueError("Invalid image.")

    return cv2.resize(
        image,
        size,
        interpolation=cv2.INTER_AREA
    )


def preprocess_image(image):
    """
    Apply basic image preprocessing.
    """

    resized = resize_image(image)

    grayscale = convert_to_grayscale(resized)

    return resized, grayscale


def get_image_info(image):
    """
    Return image dimensions and channel information.
    """

    height, width = image.shape[:2]

    channels = 1 if len(image.shape) == 2 else image.shape[2]

    return {
        "height": height,
        "width": width,
        "channels": channels
    }