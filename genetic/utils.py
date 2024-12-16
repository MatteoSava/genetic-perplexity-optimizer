import random
import numpy as np
import torch

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "/modal/google/gemma-2-9b/"
SUBMISSION_FILE = "data/sample_submission.csv"
OUTPUT_FILE = "submission.csv"


# Set random seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
