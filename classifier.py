
import cv2
import numpy as np
import joblib

from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from math import ceil

from config import (
    VOCABULARY_SIZE,
    MAX_DESCRIPTORS_FOR_KMEANS,
    DESCRIPTORS_PER_IMAGE_FOR_KMEANS,
    SVM_KERNEL,
    SVM_C,
    SVM_GAMMA,
    CLASSIFIER_PATH
)


# Reproducibility
RANDOM_STATE = 42


# ==========================================
# 1. COLLECT TRAINING DESCRIPTORS
# ==========================================

def collect_training_descriptors(records):
    """
    Collect SIFT descriptors from training images.
    """

    descriptor_list = []

    for record in records:

        descriptors = record["descriptors"]

        if descriptors is None or len(descriptors) == 0:
            continue

        # Limit descriptors per image
        descriptors = descriptors[
            :DESCRIPTORS_PER_IMAGE_FOR_KMEANS
        ]

        descriptor_list.append(descriptors)

    if not descriptor_list:
        raise ValueError("No training descriptors found.")

    all_descriptors = np.vstack(descriptor_list)

    all_descriptors = all_descriptors.astype(np.float32)

    # Limit total descriptors
    if len(all_descriptors) > MAX_DESCRIPTORS_FOR_KMEANS:

        rng = np.random.default_rng(RANDOM_STATE)

        indices = rng.choice(
            len(all_descriptors),
            MAX_DESCRIPTORS_FOR_KMEANS,
            replace=False
        )

        all_descriptors = all_descriptors[indices]

    return all_descriptors


# ==========================================
# 2. CREATE VISUAL VOCABULARY USING K-MEANS
# ==========================================

def create_visual_vocabulary(descriptors):
    """
    Generate visual words using MiniBatch K-Means.
    """

    vocabulary_size = min(
        VOCABULARY_SIZE,
        len(descriptors)
    )

    if vocabulary_size < 2:
        raise ValueError(
            "Not enough descriptors to create vocabulary."
        )

    kmeans = MiniBatchKMeans(
        n_clusters=vocabulary_size,
        random_state=RANDOM_STATE,
        batch_size=1024,
        n_init=10
    )

    kmeans.fit(descriptors)

    return kmeans


# ==========================================
# 3. GENERATE BAG OF VISUAL WORDS HISTOGRAM
# ==========================================

def generate_histogram(descriptors, kmeans):
    """
    Convert image descriptors into a visual-word histogram.
    """

    vocabulary_size = kmeans.n_clusters

    histogram = np.zeros(
        vocabulary_size,
        dtype=np.float32
    )

    if descriptors is None or len(descriptors) == 0:
        return histogram

    descriptors = descriptors.astype(np.float32)

    # Assign descriptors to visual words
    visual_words = kmeans.predict(descriptors)

    # Count visual word frequencies
    for word in visual_words:
        histogram[word] += 1

    # L1 normalization
    total = histogram.sum()

    if total > 0:
        histogram /= total

    return histogram


# ==========================================
# 4. CREATE FEATURE MATRIX
# ==========================================

def create_feature_matrix(records, kmeans):
    """
    Generate histograms and labels for all images.
    """

    features = []
    labels = []

    for record in records:

        histogram = generate_histogram(
            record["descriptors"],
            kmeans
        )

        features.append(histogram)

        labels.append(record["label"])

    X = np.asarray(features, dtype=np.float32)

    y = np.asarray(labels)

    return X, y


# ==========================================
# 5. TRAIN BOVW + SVM CLASSIFIER
# ==========================================

def train_bovw_classifier(database):
    """
    Train SVM using Bag of Visual Words features.
    """

    print("\n========== BOVW CLASSIFIER TRAINING ==========")

    # Check detector
    if database["detector"] != "SIFT":
        raise ValueError(
            "BoVW classifier requires SIFT descriptors."
        )

    records = database["records"]

    if not records:
        raise ValueError("Feature database is empty.")

    # Extract labels
    labels = np.asarray([
        record["label"]
        for record in records
    ])

    unique_labels, counts = np.unique(
        labels,
        return_counts=True
    )

    # Minimum dataset requirements
    if len(unique_labels) < 2:
        raise ValueError(
            "At least two different classes are required."
        )

    if np.min(counts) < 2:
        raise ValueError(
            "Each class must contain at least two images."
        )

    # ======================================
    # TRAIN TEST SPLIT
    # ======================================

    test_count = max(
        len(unique_labels),
        ceil(0.20 * len(records))
    )

    if test_count >= len(records):

        raise ValueError(
            "Not enough images for train/test split."
        )

    if len(records) - test_count < len(unique_labels):

        raise ValueError(
            "Not enough training images to represent every class."
        )

    train_records, test_records = train_test_split(
        records,
        test_size=test_count,
        random_state=RANDOM_STATE,
        stratify=labels
    )

    print("Total images:", len(records))

    print("Training images:", len(train_records))

    print("Testing images:", len(test_records))

    # ======================================
    # BUILD VISUAL VOCABULARY
    # ======================================

    print("\nCollecting training descriptors...")

    training_descriptors = collect_training_descriptors(
        train_records
    )

    print(
        "Total descriptors:",
        len(training_descriptors)
    )

    print("\nCreating visual vocabulary...")

    kmeans = create_visual_vocabulary(
        training_descriptors
    )

    print(
        "Vocabulary size:",
        kmeans.n_clusters
    )

    # ======================================
    # GENERATE HISTOGRAMS
    # ======================================

    print("\nGenerating training histograms...")

    X_train, y_train = create_feature_matrix(
        train_records,
        kmeans
    )

    print("Training feature matrix:", X_train.shape)

    print("\nGenerating testing histograms...")

    X_test, y_test = create_feature_matrix(
        test_records,
        kmeans
    )

    print("Testing feature matrix:", X_test.shape)

    # ======================================
    # TRAIN SVM
    # ======================================

    print("\nTraining SVM model...")

    svm = SVC(
        kernel=SVM_KERNEL,
        C=SVM_C,
        gamma=SVM_GAMMA,
        probability=True,
        random_state=RANDOM_STATE
    )

    svm.fit(
        X_train,
        y_train
    )

    # ======================================
    # MODEL EVALUATION
    # ======================================

    train_predictions = svm.predict(X_train)

    test_predictions = svm.predict(X_test)

    train_accuracy = accuracy_score(
        y_train,
        train_predictions
    )

    test_accuracy = accuracy_score(
        y_test,
        test_predictions
    )

    print("\n========== SVM RESULTS ==========")

    print(
        "Training Accuracy:",
        round(train_accuracy * 100, 2),
        "%"
    )

    print(
        "Testing Accuracy:",
        round(test_accuracy * 100, 2),
        "%"
    )

    # ======================================
    # SAVE TRAINED MODEL
    # ======================================

    model_package = {

        "kmeans": kmeans,

        "svm": svm,

        "class_names": list(unique_labels),

        "detector": "SIFT",

        "vocabulary_size": kmeans.n_clusters,

        "training_accuracy": train_accuracy,

        "testing_accuracy": test_accuracy

    }

    CLASSIFIER_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        model_package,
        CLASSIFIER_PATH
    )

    print("\n========== MODEL SAVED ==========")

    print("Model path:", CLASSIFIER_PATH)

    print("Classes:", list(unique_labels))

    print("\nBOVW + SVM TRAINING COMPLETED!")

    return model_package


# ==========================================
# 6. LOAD TRAINED CLASSIFIER
# ==========================================

def load_classifier():
    """
    Load the saved BoVW + SVM model.
    """

    if not CLASSIFIER_PATH.exists():

        raise FileNotFoundError(
            "Trained classifier not found. Train it first."
        )

    model = joblib.load(CLASSIFIER_PATH)

    print("\nClassifier loaded successfully!")

    return model
