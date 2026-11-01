import cv2
import joblib

from pathlib import Path

from config import (
    FEATURE_DATABASE_PATH,
    DATABASE_DETECTOR,
    MAX_DESCRIPTORS_PER_IMAGE,
    MIN_DESCRIPTORS_REQUIRED
)

from dataset_loader import load_dataset

from image_utils import (
    load_image,
    preprocess_image
)

from descriptor_extractor import extract_descriptors


def build_feature_database(method=DATABASE_DETECTOR):
    """
    Extract and store descriptors from all dataset images.
    """

    dataset, class_names = load_dataset()

    feature_database = []

    print("\nBuilding Feature Database...")

    for index, item in enumerate(dataset):

        try:

            # Load image
            image = load_image(item["image_path"])

            # Preprocessing
            resized, grayscale = preprocess_image(image)

            # Extract descriptors
            keypoints, descriptors = extract_descriptors(
                grayscale,
                method
            )

            # Skip images without enough features
            if (
                descriptors is None
                or len(descriptors) < MIN_DESCRIPTORS_REQUIRED
            ):

                print(
                    "Skipping image:",
                    item["image_name"]
                )

                continue

            # Limit descriptors per image
            descriptors = descriptors[
                :MAX_DESCRIPTORS_PER_IMAGE
            ]

            # Store image information
            record = {

                "image_name": item["image_name"],

                "image_path": item["image_path"],

                "label": item["label"],

                "class_index": item["class_index"],

                "keypoint_count": len(keypoints),

                "descriptors": descriptors,

                "detector": method

            }

            feature_database.append(record)

            print(
                f"[{index + 1}/{len(dataset)}] "
                f"{item['image_name']} -> "
                f"{item['label']} | "
                f"Descriptors: {len(descriptors)}"
            )

        except Exception as error:

            print(
                "Error processing:",
                item["image_name"],
                error
            )

    # Database information
    database = {

        "class_names": class_names,

        "detector": method,

        "total_images": len(feature_database),

        "records": feature_database

    }

    # Save database
    FEATURE_DATABASE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        database,
        FEATURE_DATABASE_PATH
    )

    print("\nFeature Database Created Successfully!")

    print("Total stored images:", len(feature_database))

    print("Database location:", FEATURE_DATABASE_PATH)

    return database


def load_feature_database():
    """
    Load saved feature database.
    """

    if not FEATURE_DATABASE_PATH.exists():

        raise FileNotFoundError(
            "Feature database not found. Build it first."
        )

    return joblib.load(FEATURE_DATABASE_PATH)


def get_descriptors_by_class(
    database,
    class_name
):
    """
    Retrieve descriptors belonging to one class.
    """

    descriptors_list = [

        record["descriptors"]

        for record in database["records"]

        if record["label"] == class_name

    ]

    return descriptors_list


def get_image_descriptor(
    database,
    image_name
):
    """
    Retrieve descriptors for a particular image.
    """

    for record in database["records"]:

        if record["image_name"] == image_name:

            return record["descriptors"]

    return None


def get_class_indices(database):
    """
    Return class name to index mapping.
    """

    return {
        name: index
        for index, name in enumerate(
            database["class_names"]
        )
    }