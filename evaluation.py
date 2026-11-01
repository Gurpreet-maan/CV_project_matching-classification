
import time
import json
import csv

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from descriptor_database import load_feature_database

from classifier import (
    collect_training_descriptors,
    create_visual_vocabulary,
    create_feature_matrix
)

from config import (
    SVM_KERNEL,
    SVM_C,
    SVM_GAMMA
)

from math import ceil


# ==========================================
# CONFIGURATION
# ==========================================

RANDOM_STATE = 42

OUTPUT_DIR = Path("outputs/evaluation")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================
# 1. LOAD DATASET
# ==========================================

def load_evaluation_data():

    print("\n========== LOADING DATA ==========")

    database = load_feature_database()

    if database["detector"] != "SIFT":
        raise ValueError(
            "This BoVW evaluation currently requires SIFT descriptors."
        )

    records = database["records"]

    if not records:
        raise ValueError("Dataset is empty.")

    labels = np.array([
        record["label"]
        for record in records
    ])

    classes, counts = np.unique(
        labels,
        return_counts=True
    )

    print("Total images:", len(records))
    print("Number of classes:", len(classes))

    for class_name, count in zip(classes, counts):
        print(f"{class_name}: {count} images")

    if len(classes) < 2:
        raise ValueError(
            "At least two classes are required."
        )

    if np.min(counts) < 2:
        raise ValueError(
            "Each class needs at least two images."
        )

    return records, labels


# ==========================================
# 2. TRAIN TEST SPLIT
# ==========================================

def split_dataset(records, labels):

    print("\n========== DATASET SPLITTING ==========")

    classes = np.unique(labels)

    test_count = max(
        len(classes),
        ceil(0.20 * len(records))
    )

    if test_count >= len(records):
        raise ValueError(
            "Not enough images for testing."
        )

    if len(records) - test_count < len(classes):
        raise ValueError(
            "Not enough training images."
        )

    train_records, test_records = train_test_split(
        records,
        test_size=test_count,
        random_state=RANDOM_STATE,
        stratify=labels
    )

    print("Training images:", len(train_records))
    print("Testing images:", len(test_records))

    return train_records, test_records


# ==========================================
# 3. TRAIN BOVW + SVM MODEL
# ==========================================

def train_evaluation_model(train_records):

    print("\n========== MODEL TRAINING ==========")

    start_time = time.perf_counter()

    # Collect SIFT descriptors
    descriptors = collect_training_descriptors(
        train_records
    )

    print("Training descriptors:", len(descriptors))

    # Create visual vocabulary
    kmeans = create_visual_vocabulary(
        descriptors
    )

    print("Vocabulary size:", kmeans.n_clusters)

    # Generate histograms
    X_train, y_train = create_feature_matrix(
        train_records,
        kmeans
    )

    print("Feature matrix:", X_train.shape)

    # Train SVM
    svm = SVC(
        kernel=SVM_KERNEL,
        C=SVM_C,
        gamma=SVM_GAMMA,
        random_state=RANDOM_STATE
    )

    svm.fit(X_train, y_train)

    training_time = time.perf_counter() - start_time

    print("SVM training completed!")

    print(
        "Training time:",
        round(training_time, 4),
        "seconds"
    )

    return kmeans, svm, training_time


# ==========================================
# 4. MODEL PREDICTION
# ==========================================

def evaluate_predictions(test_records, kmeans, svm):

    print("\n========== MODEL PREDICTION ==========")

    X_test, y_test = create_feature_matrix(
        test_records,
        kmeans
    )

    start_time = time.perf_counter()

    predictions = svm.predict(X_test)

    prediction_time = time.perf_counter() - start_time

    average_time = prediction_time / len(X_test)

    print("Testing completed!")

    print("Test samples:", len(X_test))

    print(
        "Total prediction time:",
        round(prediction_time, 6),
        "seconds"
    )

    print(
        "Average prediction time per image:",
        round(average_time, 6),
        "seconds"
    )

    return y_test, predictions, average_time


# ==========================================
# 5. CALCULATE EVALUATION METRICS
# ==========================================

def calculate_metrics(y_test, predictions):

    print("\n========== EVALUATION METRICS ==========")

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    metrics = {

        "accuracy": round(float(accuracy * 100), 2),

        "precision": round(float(precision * 100), 2),

        "recall": round(float(recall * 100), 2),

        "f1_score": round(float(f1 * 100), 2)

    }

    for name, value in metrics.items():

        print(
            name.replace("_", " ").title(),
            ":",
            value,
            "%"
        )

    return metrics


# ==========================================
# 6. CLASSIFICATION REPORT
# ==========================================

def generate_classification_report(y_test, predictions):

    print("\n========== CLASSIFICATION REPORT ==========")

    report = classification_report(
        y_test,
        predictions,
        zero_division=0
    )

    print(report)

    report_path = OUTPUT_DIR / "classification_report.txt"

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "VISIONMATCH AI - CLASSIFICATION REPORT\n\n"
        )

        file.write(report)

    print("Report saved:", report_path)


# ==========================================
# 7. CONFUSION MATRIX
# ==========================================

def generate_confusion_matrix(y_test, predictions, classes):

    print("\n========== CONFUSION MATRIX ==========")

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=classes
    )

    print(matrix)

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=classes
    )

    fig, ax = plt.subplots(figsize=(9, 7))

    display.plot(
        ax=ax,
        cmap="Blues",
        values_format="d",
        xticks_rotation=45
    )

    plt.title("VisionMatch AI - Confusion Matrix")

    plt.tight_layout()

    path = OUTPUT_DIR / "confusion_matrix.png"

    plt.savefig(path, dpi=300)

    plt.show()

    plt.close(fig)

    print("Confusion matrix saved:", path)


# ==========================================
# 8. CLASS-WISE PERFORMANCE CHART
# ==========================================

def generate_performance_chart(y_test, predictions, classes):

    print("\n========== CLASS PERFORMANCE CHART ==========")

    report = classification_report(
        y_test,
        predictions,
        labels=classes,
        target_names=[str(c) for c in classes],
        output_dict=True,
        zero_division=0
    )

    precision_values = []
    recall_values = []
    f1_values = []

    for class_name in classes:

        class_metrics = report[str(class_name)]

        precision_values.append(
            class_metrics["precision"] * 100
        )

        recall_values.append(
            class_metrics["recall"] * 100
        )

        f1_values.append(
            class_metrics["f1-score"] * 100
        )

    x = np.arange(len(classes))

    width = 0.25

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.bar(
        x - width,
        precision_values,
        width,
        label="Precision"
    )

    ax.bar(
        x,
        recall_values,
        width,
        label="Recall"
    )

    ax.bar(
        x + width,
        f1_values,
        width,
        label="F1-Score"
    )

    ax.set_title("Class-wise Performance")

    ax.set_xlabel("Classes")

    ax.set_ylabel("Score (%)")

    ax.set_xticks(x)

    ax.set_xticklabels(classes)

    ax.set_ylim(0, 110)

    ax.legend()

    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    path = OUTPUT_DIR / "class_performance.png"

    plt.savefig(path, dpi=300)

    plt.show()

    plt.close(fig)

    print("Performance chart saved:", path)


# ==========================================
# 9. SAVE METRICS
# ==========================================

def save_metrics(metrics, training_time, average_time):

    results = {

        "project": "VisionMatch AI",

        "algorithm": "SIFT + BoVW + SVM",

        "metrics": metrics,

        "training_time_seconds": round(
            training_time, 6
        ),

        "average_prediction_time_seconds": round(
            average_time, 6
        )

    }

    path = OUTPUT_DIR / "evaluation_metrics.json"

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=4
        )

    print("\nMetrics saved:", path)

    # Also save CSV
    csv_path = OUTPUT_DIR / "evaluation_metrics.csv"

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow(["Metric", "Value"])

        for name, value in metrics.items():

            writer.writerow([name, value])

        writer.writerow([
            "training_time_seconds",
            round(training_time, 6)
        ])

        writer.writerow([
            "average_prediction_time_seconds",
            round(average_time, 6)
        ])

    print("CSV saved:", csv_path)


# ==========================================
# 10. MAIN EVALUATION PIPELINE
# ==========================================

def run_evaluation():

    print("\n")
    print("=" * 55)

    print("       VISIONMATCH AI - MODEL EVALUATION")

    print("=" * 55)

    # Load data
    records, labels = load_evaluation_data()

    # Split dataset
    train_records, test_records = split_dataset(
        records,
        labels
    )

    # Train model
    kmeans, svm, training_time = train_evaluation_model(
        train_records
    )

    # Predict
    y_test, predictions, average_time = evaluate_predictions(
        test_records,
        kmeans,
        svm
    )

    classes = svm.classes_

    # Calculate metrics
    metrics = calculate_metrics(
        y_test,
        predictions
    )

    # Classification report
    generate_classification_report(
        y_test,
        predictions
    )

    # Confusion matrix
    generate_confusion_matrix(
        y_test,
        predictions,
        classes
    )

    # Performance chart
    generate_performance_chart(
        y_test,
        predictions,
        classes
    )

    # Save results
    save_metrics(
        metrics,
        training_time,
        average_time
    )

    print("\n")
    print("=" * 55)

    print("       EVALUATION COMPLETED SUCCESSFULLY!")

    print("=" * 55)

    print("\nAll evaluation files saved in:")

    print(OUTPUT_DIR.resolve())


# ==========================================
# PROGRAM ENTRY POINT
# ==========================================

if __name__ == "__main__":

    run_evaluation()
