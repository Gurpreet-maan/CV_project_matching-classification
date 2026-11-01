from pathlib import Path

# ==============================
# PROJECT DIRECTORIES
# ==============================

BASE_DIR = Path(__file__).resolve().parent

DATASET_DIR = BASE_DIR / "dataset"
INPUT_DIR = BASE_DIR / "input_images"
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = BASE_DIR / "models"

KEYPOINT_DIR = OUTPUT_DIR / "keypoints"
MATCH_DIR = OUTPUT_DIR / "matches"
CLASSIFICATION_DIR = OUTPUT_DIR / "classification_results"
CHART_DIR = OUTPUT_DIR / "comparison_charts"

# ==============================
# IMAGE SETTINGS
# ==============================

IMAGE_SIZE = (512, 512)

SUPPORTED_FORMATS = (".jpg", ".jpeg", ".png", ".bmp")

# ==============================
# FEATURE DETECTION SETTINGS
# ==============================

DEFAULT_DETECTOR = "SIFT"

SIFT_FEATURES = 1000

ORB_FEATURES = 1000

# ==============================
# MATCHING SETTINGS
# ==============================

RATIO_THRESHOLD = 0.75
MIN_MATCH_COUNT = 10

# ==============================
# CLASSIFICATION SETTINGS
# ==============================

RANDOM_STATE = 42
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.10

# ==============================
# CREATE REQUIRED DIRECTORIES
# ==============================

DIRECTORIES = [
    INPUT_DIR,
    OUTPUT_DIR,
    KEYPOINT_DIR,
    MATCH_DIR,
    CLASSIFICATION_DIR,
    CHART_DIR,
    MODEL_DIR
]


def create_directories():

    for directory in DIRECTORIES:
        directory.mkdir(parents=True, exist_ok=True)


# Create folders automatically
create_directories()

# ==============================
# FEATURE DATABASE SETTINGS
# ==============================

FEATURE_DATABASE_PATH = MODEL_DIR / "feature_database.pkl"

MAX_DESCRIPTORS_PER_IMAGE = 500

DATABASE_DETECTOR = "SIFT"

MIN_DESCRIPTORS_REQUIRED = 2

# ==============================
# BOVW CLASSIFICATION SETTINGS
# ==============================

VOCABULARY_SIZE = 100

MAX_DESCRIPTORS_FOR_KMEANS = 20000

DESCRIPTORS_PER_IMAGE_FOR_KMEANS = 200

SVM_KERNEL = "rbf"

SVM_C = 10

SVM_GAMMA = "scale"

CLASSIFIER_PATH = MODEL_DIR / "bovw_svm_model.pkl"

MATCHING_CLASSIFIER_PATH = MODEL_DIR / "matching_classifier.pkl"