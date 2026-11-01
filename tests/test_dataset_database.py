from dataset_loader import (
    load_dataset,
    print_dataset_summary
)

from descriptor_database import (
    build_feature_database,
    load_feature_database,
    get_descriptors_by_class,
    get_class_indices
)


def test_dataset_database():

    # Step 1: Load dataset
    dataset, class_names = load_dataset()

    # Step 2: Print summary
    print_dataset_summary(
        dataset,
        class_names
    )

    # Step 3: Build descriptor database
    database = build_feature_database(
        method="SIFT"
    )

    # Step 4: Load saved database
    loaded_database = load_feature_database()

    print("\n========== DATABASE TEST ==========")

    print(
        "Total stored images:",
        loaded_database["total_images"]
    )

    print(
        "Available classes:",
        loaded_database["class_names"]
    )

    # Step 5: Retrieve class descriptors
    for class_name in class_names:

        descriptors = get_descriptors_by_class(
            loaded_database,
            class_name
        )

        print(
            class_name,
            "->",
            len(descriptors),
            "images stored"
        )

    # Step 6: Class index mapping
    print("\nClass Index Mapping:")

    print(get_class_indices(loaded_database))

    print("\nDATABASE TEST COMPLETED!")


if __name__ == "__main__":

    test_dataset_database()