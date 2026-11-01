
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

from descriptor_database import load_feature_database
from classifier import train_bovw_classifier

from predict import (
    predict_bovw,
    predict_by_matching
)


def test_classifier():

    # Step 1: Load feature database
    database = load_feature_database()

    print("\nFeature database loaded!")

    # Step 2: Train BoVW + SVM
    model = train_bovw_classifier(database)

    print("\nModel training completed!")

    # Step 3: Test image
    test_image = "input_images/test.jpg"

    # Check whether image exists
    image = cv2.imread(test_image)

    if image is None:
        raise FileNotFoundError(
            f"Test image not found: {test_image}"
        )

    # Step 4: BoVW prediction
    print("\n========== BOVW + SVM ==========")

    result1 = predict_bovw(test_image)

    print("Predicted Class:", result1["predicted_class"])
    print("Confidence:", result1["confidence"], "%")
    print("Keypoints:", result1["keypoints"])
    print("Class Scores:", result1["class_scores"])

    # Step 5: Matching-based prediction
    print("\n========== DESCRIPTOR MATCHING ==========")

    result2 = predict_by_matching(test_image)

    print("Predicted Class:", result2["predicted_class"])
    print("Matching Score:", result2["matching_score"], "%")
    print("Keypoints:", result2["keypoints"])
    print("Class Scores:", result2["class_scores"])

    print("\nCLASSIFICATION TEST COMPLETED!")

    # ======================================
    # STEP 6: VISUALIZE CLASSIFICATION RESULT
    # ======================================

    output_dir = Path("outputs/classification_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert image into display copy
    result_image = image.copy()

    # Add prediction information
    text1 = f"BoVW + SVM: {result1['predicted_class']}"
    text2 = f"Matching: {result2['predicted_class']}"

    cv2.putText(
        result_image,
        text1,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        result_image,
        text2,
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    # Save result image
    output_path = output_dir / "prediction_result.jpg"

    cv2.imwrite(str(output_path), result_image)

    # Display result
    plt.figure(figsize=(10, 7))

    plt.imshow(
        cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
    )

    plt.title("VisionMatch AI - Classification Result")

    plt.axis("off")

    plt.tight_layout()
    plt.show()

    print("\nPrediction image saved:", output_path)


if __name__ == "__main__":

    test_classifier()
