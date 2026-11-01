import cv2
import numpy as np

from config import RATIO_THRESHOLD


def create_matcher(method="SIFT", matcher_type="BF"):
    """
    Create BFMatcher or FLANN matcher.
    """

    method = method.upper()
    matcher_type = matcher_type.upper()

    if method == "SIFT":
        norm = cv2.NORM_L2

    elif method == "ORB":
        norm = cv2.NORM_HAMMING

    else:
        raise ValueError("Choose SIFT or ORB.")

    if matcher_type == "BF":

        return cv2.BFMatcher(
            norm,
            crossCheck=False
        )

    elif matcher_type == "FLANN":

        if method == "SIFT":

            index_params = {
                "algorithm": 1,
                "trees": 5
            }

            search_params = {
                "checks": 50
            }

        else:

            index_params = {
                "algorithm": 6,
                "table_number": 6,
                "key_size": 12,
                "multi_probe_level": 1
            }

            search_params = {
                "checks": 50
            }

        return cv2.FlannBasedMatcher(
            index_params,
            search_params
        )

    else:
        raise ValueError("Choose BF or FLANN.")


def match_descriptors(
    descriptors1,
    descriptors2,
    method="SIFT",
    matcher_type="BF"
):
    """
    Match two descriptor arrays using KNN.
    """

    if descriptors1 is None or descriptors2 is None:
        return []

    if len(descriptors1) < 2 or len(descriptors2) < 2:
        return []

    if descriptors1.dtype != descriptors2.dtype:
        raise ValueError("Descriptor data types must match.")

    matcher = create_matcher(
        method,
        matcher_type
    )

    matches = matcher.knnMatch(
        descriptors1,
        descriptors2,
        k=2
    )

    return matches


def apply_ratio_test(
    matches,
    threshold=RATIO_THRESHOLD
):
    """
    Apply Lowe's Ratio Test.
    """

    good_matches = []

    for pair in matches:

        if len(pair) < 2:
            continue

        best_match, second_match = pair

        if best_match.distance < threshold * second_match.distance:

            good_matches.append(best_match)

    return good_matches


def calculate_match_statistics(
    total_matches,
    good_matches
):
    """
    Calculate matching statistics.
    """

    total = len(total_matches)

    good = len(good_matches)

    match_percentage = (
        (good / total) * 100
        if total > 0
        else 0
    )

    return {
        "total_matches": total,
        "good_matches": good,
        "match_percentage": round(match_percentage, 2)
    }


def verify_matches_ransac(
    keypoints1,
    keypoints2,
    good_matches
):
    """
    Verify matches using RANSAC homography.
    """

    if len(good_matches) < 4:
        return None, None, 0

    points1 = np.float32([
        keypoints1[m.queryIdx].pt
        for m in good_matches
    ]).reshape(-1, 1, 2)

    points2 = np.float32([
        keypoints2[m.trainIdx].pt
        for m in good_matches
    ]).reshape(-1, 1, 2)

    homography, mask = cv2.findHomography(
        points1,
        points2,
        cv2.RANSAC,
        5.0
    )

    if mask is None:
        return homography, None, 0

    inlier_count = int(np.sum(mask))

    return homography, mask.ravel().astype(bool), inlier_count