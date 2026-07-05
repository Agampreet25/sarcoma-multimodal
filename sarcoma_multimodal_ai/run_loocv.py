import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader

from src.dataset import SarcomaDataset
from src.model import MultimodalBayesianNet
from src.train import train_model
from src.uncertainty import mc_dropout_predict
from src.metrics import compute_metrics
from src.utils import set_seed
from src.visualize import (
    plot_uncertainty_scatter,
    plot_uncertainty_calibration,
    plot_deferral_curve,
)

set_seed(42)

ROOT_DIR = (
    "/Users/agampreet/Desktop/sarcoma_multimodal_ai/"
    "can-img/manifest-MjbMt99Q1553106146386120388/"
    "Soft-tissue-Sarcoma"
)

patient_ids = [f"STS_{i:03d}" for i in range(1, 52)]
# patient_ids = patient_ids[:10]   # TEMPORARY: run on first 10 patients

labels_df = pd.read_csv("labels.csv")

if labels_df["label"].isnull().any():
    raise ValueError("labels.csv contains empty labels.")

labels = dict(zip(labels_df.patient_id, labels_df.label))

missing_labels = [p for p in patient_ids if p not in labels]
if len(missing_labels) > 0:
    raise ValueError(f"Missing labels for patients: {missing_labels}")

print("✅ Labels loaded successfully")
print(labels_df["label"].value_counts())

y_true, y_pred, y_unc = [], [], []

for test_pid in patient_ids:
    print(f"\n=== LOOCV: Testing on {test_pid} ===")

    train_ids = [p for p in patient_ids if p != test_pid]

    train_ds = SarcomaDataset(train_ids, labels, ROOT_DIR)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)

    model = MultimodalBayesianNet()
    train_model(model, train_loader)

    test_ds = SarcomaDataset([test_pid], labels, ROOT_DIR)
    mri, pet, label = test_ds[0]

    mean_p, std_p = mc_dropout_predict(
        model,
        mri.unsqueeze(0),
        pet.unsqueeze(0)
    )

    y_true.append(label.item())
    y_pred.append(mean_p)
    y_unc.append(std_p)

    print(
        f"Prediction: {mean_p:.3f} | "
        f"Uncertainty: {std_p:.3f} | "
        f"GT: {int(label.item())}"
    )

auc, acc = compute_metrics(y_true, y_pred)

print("\n================ FINAL RESULTS ================")
print(f"LOOCV AUC: {auc:.4f}")
print(f"LOOCV Accuracy: {acc:.4f}")
print(f"Mean Predictive Uncertainty: {np.mean(y_unc):.4f}")
print("==============================================")

# ---------------- VISUAL ANALYSIS ----------------
plot_uncertainty_scatter(y_pred, y_unc)
plot_uncertainty_calibration(y_true, y_pred)
plot_deferral_curve(y_true, y_pred, y_unc)
