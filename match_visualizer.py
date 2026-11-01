import cv2
import numpy as np


def draw_feature_matches(
    image1,
    keypoints1,
    image2,
    keypoints2,
    matches,
    max_matches=100
):
    """
    Draw feature matches between two images.
    """

    selected_matches = sorted(
        matches,
        key=lambda match: match.distance
    )[:max_matches]

    output = cv2.drawMatches(
        image1,
        keypoints1,
        image2,
        keypoints2,
        selected_matches,
        None,
        matchColor=(0, 255, 0),
        singlePointColor=(255, 0, 0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    return output


def draw_verified_matches(
    image1,
    keypoints1,
    image2,
    keypoints2,
    good_matches,
    inlier_mask
):
    """
    Draw only RANSAC-verified inlier matches.
    """

    if inlier_mask is None:
        return draw_feature_matches(
            image1,
            keypoints1,
            image2,
            keypoints2,
            good_matches
        )

    inlier_matches = [
        match
        for match, is_inlier in zip(
            good_matches,
            inlier_mask
        )
        if is_inlier
    ]

    return draw_feature_matches(
        image1,
        keypoints1,
        image2,
        keypoints2,
        inlier_matches
    )