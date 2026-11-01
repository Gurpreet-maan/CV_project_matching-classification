import cv2

from feature_detector import create_detector


def extract_descriptors(image, method="SIFT"):
    """
    Detect keypoints and extract their descriptors.
    """

    if image is None:
        raise ValueError("Invalid image.")

    detector = create_detector(method)

    keypoints, descriptors = detector.detectAndCompute(
        image,
        None
    )

    if descriptors is None:
        return keypoints, None

    return keypoints, descriptors


def get_descriptor_info(descriptors):
    """
    Return descriptor information.
    """

    if descriptors is None:
        return {
            "count": 0,
            "dimensions": 0,
            "dtype": None
        }

    return {
        "count": descriptors.shape[0],
        "dimensions": descriptors.shape[1],
        "dtype": str(descriptors.dtype)
    }


def compare_descriptors(
    descriptors1,
    descriptors2
):
    """
    Check whether two descriptor arrays
    have compatible dimensions.
    """

    if descriptors1 is None or descriptors2 is None:
        return False

    return descriptors1.shape[1] == descriptors2.shape[1]