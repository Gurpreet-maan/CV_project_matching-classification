import cv2
import numpy as np

from image_utils import load_image, preprocess_image

from descriptor_extractor import extract_descriptors

from descriptor_database import load_feature_database

from feature_matcher import (
    match_descriptors,
    apply_ratio_test
)

from classifier import (
    generate_histogram,
    load_classifier
)


def predict_bovw(image_path):
    """
    Predict image class using BoVW + SVM.
    """

    model = load_classifier()

    # Load and preprocess image
    image = load_image(image_path)

    resized, grayscale = preprocess_image(image)

    # Extract descriptors
    keypoints, descriptors = extract_descriptors(
        grayscale,
        method="SIFT"
    )

    if descriptors is None or len(descriptors) == 0:
        raise ValueError(
            "No SIFT features found in test image."
        )

    # Generate visual histogram
    histogram = generate_histogram(
        descriptors,
        model["kmeans"]
    )

    # Predict
    prediction = model["svm"].predict(
        histogram.reshape(1, -1)
    )[0]

    probabilities = model["svm"].predict_proba(
        histogram.reshape(1, -1)
    )[0]

    confidence = float(np.max(probabilities)) * 100

    return {
        "predicted_class": str(prediction),
        "confidence": round(confidence, 2),
        "keypoints": len(keypoints),
        "class_scores": {
            str(name): round(float(score) * 100, 2)
            for name, score in zip(
                model["svm"].classes_,
                probabilities
            )
        }
    }


def predict_by_matching(image_path):
    """
    Predict image class using descriptor matching.
    """

    database = load_feature_database()

    # Load test image
    image = load_image(image_path)

    resized, grayscale = preprocess_image(image)

    method = database["detector"]

    query_keypoints, query_descriptors = extract_descriptors(
        grayscale,
        method
    )

    if query_descriptors is None:
        raise ValueError(
            "No descriptors found in query image."
        )

    class_scores = {
        class_name: []
        for class_name in database["class_names"]
    }

    # Compare query with every database image
    for record in database["records"]:

        matches = match_descriptors(
            query_descriptors,
            record["descriptors"],
            method=method,
            matcher_type="BF"
        )

        good_matches = apply_ratio_test(matches)

        # Normalize by query descriptor count
        score = (
            len(good_matches) / len(query_descriptors)
        ) * 100

        class_scores[record["label"]].append(score)

    # Average top 3 image scores per class
    final_scores = {}

    for class_name, scores in class_scores.items():

        top_scores = sorted(
            scores,
            reverse=True
        )[:3]

        final_scores[class_name] = (
            float(np.mean(top_scores))
            if top_scores
            else 0.0
        )

    predicted_class = max(
        final_scores,
        key=final_scores.get
    )

    return {
        "predicted_class": predicted_class,
        "matching_score": round(
            final_scores[predicted_class], 2
        ),
        "class_scores": {
            name: round(score, 2)
            for name, score in final_scores.items()
        },
        "keypoints": len(query_keypoints)
    }