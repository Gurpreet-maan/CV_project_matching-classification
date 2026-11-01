import cv2
import matplotlib.pyplot as plt

from config import MATCH_DIR

from image_utils import (
    load_image,
    preprocess_image
)

from descriptor_extractor import extract_descriptors

from feature_matcher import (
    match_descriptors,
    apply_ratio_test,
    calculate_match_statistics,
    verify_matches_ransac
)

from match_visualizer import (
    draw_feature_matches,
    draw_verified_matches
)


def run_matching_test(method="SIFT", matcher_type="BF"):

    # Load two images
    image1 = load_image("input_images/test.jpg")
    image2 = load_image("input_images/test2.jpg")

    # Preprocessing
    image1, gray1 = preprocess_image(image1)
    image2, gray2 = preprocess_image(image2)

    # Extract keypoints and descriptors
    kp1, desc1 = extract_descriptors(gray1, method)
    kp2, desc2 = extract_descriptors(gray2, method)

    print("\nDetector:", method)
    print("Matcher:", matcher_type)

    print("Keypoints Image 1:", len(kp1))
    print("Keypoints Image 2:", len(kp2))

    # Descriptor matching
    total_matches = match_descriptors(
        desc1,
        desc2,
        method,
        matcher_type
    )

    # Lowe's ratio test
    good_matches = apply_ratio_test(total_matches)

    # Statistics
    stats = calculate_match_statistics(
        total_matches,
        good_matches
    )

    print("\nMatching Statistics")
    print("-------------------")

    for key, value in stats.items():
        print(key, ":", value)

    # RANSAC verification
    homography, mask, inliers = verify_matches_ransac(
        kp1,
        kp2,
        good_matches
    )

    inlier_percentage = (
        (inliers / len(good_matches)) * 100
        if good_matches
        else 0
    )

    print("RANSAC Inliers:", inliers)
    print("Inlier Percentage:", round(inlier_percentage, 2))
    print("Homography Found:", homography is not None)

    # Draw all good matches
    match_image = draw_feature_matches(
        image1,
        kp1,
        image2,
        kp2,
        good_matches
    )

    # Draw verified matches
    verified_image = draw_verified_matches(
        image1,
        kp1,
        image2,
        kp2,
        good_matches,
        mask
    )

    # Save results
    cv2.imwrite(
        str(MATCH_DIR / f"{method}_{matcher_type}_all.jpg"),
        match_image
    )

    cv2.imwrite(
        str(MATCH_DIR / f"{method}_{matcher_type}_verified.jpg"),
        verified_image
    )

    # Display
    plt.figure(figsize=(16, 8))

    plt.subplot(2, 1, 1)
    plt.imshow(
        cv2.cvtColor(match_image, cv2.COLOR_BGR2RGB)
    )
    plt.title("Good Feature Matches")
    plt.axis("off")

    plt.subplot(2, 1, 2)
    plt.imshow(
        cv2.cvtColor(verified_image, cv2.COLOR_BGR2RGB)
    )
    plt.title("RANSAC Verified Matches")
    plt.axis("off")

    plt.tight_layout()

    plt.savefig(
        MATCH_DIR / f"{method}_{matcher_type}_comparison.png"
    )

    plt.show()


if __name__ == "__main__":

    run_matching_test(
        method="SIFT",
        matcher_type="BF"
    )