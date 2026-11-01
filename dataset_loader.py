import cv2

from pathlib import Path

from config import DATASET_DIR, SUPPORTED_FORMATS


def get_class_names(dataset_dir=DATASET_DIR):
    """
    Extract class names from dataset folders.
    """

    dataset_dir = Path(dataset_dir)

    if not dataset_dir.exists():
        raise FileNotFoundError("Dataset folder not found.")

    class_names = sorted([
        folder.name
        for folder in dataset_dir.iterdir()
        if folder.is_dir()
    ])

    return class_names


def load_dataset(dataset_dir=DATASET_DIR):
    """
    Load image paths and their class labels.
    """

    dataset_dir = Path(dataset_dir)

    class_names = get_class_names(dataset_dir)

    dataset = []

    for label in class_names:

        class_folder = dataset_dir / label

        for image_path in sorted(class_folder.iterdir()):

            if image_path.suffix.lower() not in SUPPORTED_FORMATS:
                continue

            image = cv2.imread(str(image_path))

            if image is None:
                print("Skipping invalid image:", image_path)
                continue

            image_name = image_path.stem

            dataset.append({
                "image_path": str(image_path),
                "image_name": image_name,
                "label": label,
                "class_index": class_names.index(label)
            })

    return dataset, class_names


def get_dataset_summary(dataset, class_names):
    """
    Generate class-wise image counts.
    """

    summary = {}

    for class_name in class_names:

        count = sum(
            1
            for item in dataset
            if item["label"] == class_name
        )

        summary[class_name] = count

    return summary


def print_dataset_summary(dataset, class_names):
    """
    Display dataset information.
    """

    summary = get_dataset_summary(
        dataset,
        class_names
    )

    print("\n========== DATASET SUMMARY ==========")

    print("Total images:", len(dataset))

    print("Total classes:", len(class_names))

    print("\nClass Names:")

    for index, class_name in enumerate(class_names):

        print(
            index,
            "->",
            class_name,
            ":",
            summary[class_name],
            "images"
        )

    print("=====================================")