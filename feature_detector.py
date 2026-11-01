import cv2

from config import SIFT_FEATURES, ORB_FEATURES


def create_detector(method="SIFT"):
    """
    Create a feature detector based on the selected method.
    """

    method = method.upper()

    if method == "SIFT":

        detector = cv2.SIFT_create(
            nfeatures=SIFT_FEATURES
        )

    elif method == "ORB":

        detector = cv2.ORB_create(
            nfeatures=ORB_FEATURES
        )

    else:
        raise ValueError(
            "Unsupported detector. Choose SIFT or ORB."
        )

    return detector


def detect_keypoints(image, method="SIFT"):
    """
    Detect keypoints in an image.
    """

    if image is None:
        raise ValueError("Invalid image.")

    detector = create_detector(method)

    keypoints = detector.detect(image, None)

    return keypoints


def get_keypoint_count(keypoints):
    """
    Return total number of detected keypoints.
    """

    return len(keypoints)


def draw_keypoints(image, keypoints, method="SIFT"):
    """
    Draw detected keypoints on the original image.
    """

    output = cv2.drawKeypoints(
        image,
        keypoints,
        None,
        color=(0, 255, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    return output