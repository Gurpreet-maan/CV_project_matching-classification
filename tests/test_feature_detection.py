import cv2
import matplotlib.pyplot as plt

from config import KEYPOINT_DIR
from image_utils import load_image, preprocess_image
from feature_detector import (
    detect_keypoints,
    draw_keypoints,
    get_keypoint_count
)
from descriptor_extractor import (
    extract_descriptors,
    get_descriptor_info
)


def test_feature_detection():

    # Load image
    image = load_image("input_images/test.jpg")

    # Preprocessing
    resized, grayscale = preprocess_image(image)

    # Test both algorithms
    methods = ["SIFT", "ORB"]

    results = []

    for method in methods:

        print("\n======================")
        print("ALGORITHM:", method)
        print("======================")

        # Detect keypoints
        keypoints = detect_keypoints(
            grayscale,
            method
        )

        # Extract descriptors
        keypoints, descriptors = extract_descriptors(
            grayscale,
            method
        )

        # Count keypoints
        count = get_keypoint_count(keypoints)

        # Descriptor information
        info = get_descriptor_info(descriptors)

        # Draw keypoints
        output = draw_keypoints(
            resized,
            keypoints,
            method
        )

        # Save output
        output_path = KEYPOINT_DIR / f"{method}_keypoints.jpg"

        cv2.imwrite(str(output_path), output)

        print("Keypoints:", count)
        print("Descriptor count:", info["count"])
        print("Descriptor dimensions:", info["dimensions"])
        print("Data type:", info["dtype"])
        print("Saved:", output_path)

        results.append((method, output))

    # Display comparison
    plt.figure(figsize=(14, 6))

    for index, (method, output) in enumerate(results):

        output_rgb = cv2.cvtColor(
            output,
            cv2.COLOR_BGR2RGB
        )

        plt.subplot(1, 2, index + 1)
        plt.imshow(output_rgb)
        plt.title(f"{method} Keypoints")
        plt.axis("off")

    plt.tight_layout()

    plt.savefig(
        KEYPOINT_DIR / "detector_comparison.png"
    )

    plt.show()


if __name__ == "__main__":
    test_feature_detection()