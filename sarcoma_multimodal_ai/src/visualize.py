import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score

# ---------------------------------------------------
# Basic image sanity (already discussed earlier)
# ---------------------------------------------------
def show_mid_slice(volume, title):
    if hasattr(volume, "detach"):
        volume = volume.detach().cpu().numpy()

    vol = volume.squeeze()
    mid = vol.shape[0] // 2

    plt.figure(figsize=(4, 4))
    plt.imshow(vol[mid], cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.show()


# ---------------------------------------------------
# Uncertainty vs prediction
# ---------------------------------------------------
def plot_uncertainty_scatter(preds, uncs):
    plt.figure(figsize=(5, 4))
    plt.scatter(preds, uncs, alpha=0.7)
    plt.xlabel("Predicted Lung Metastasis Risk")
    plt.ylabel("Predictive Uncertainty (Std)")
    plt.title("Uncertainty vs Prediction")
    plt.grid(True)
    plt.show()


# ---------------------------------------------------
# Uncertainty calibration
# ---------------------------------------------------
def plot_uncertainty_calibration(y_true, y_pred, n_bins=5):
    prob_true, prob_pred = calibration_curve(
        y_true, y_pred, n_bins=n_bins, strategy="uniform"
    )

    plt.figure(figsize=(5, 5))
    plt.plot(prob_pred, prob_true, "o-", label="Model")
    plt.plot([0, 1], [0, 1], "--", label="Perfect calibration")
    plt.xlabel("Mean predicted risk")
    plt.ylabel("Observed frequency")
    plt.title("Risk Calibration Curve")
    plt.legend()
    plt.grid(True)
    plt.show()


# ---------------------------------------------------
# Deferral analysis (coverage vs accuracy)
# ---------------------------------------------------
def plot_deferral_curve(y_true, y_pred, y_unc):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_unc = np.array(y_unc)

    thresholds = np.percentile(y_unc, np.linspace(0, 100, 20))
    coverages = []
    accuracies = []

    for t in thresholds:
        keep = y_unc <= t
        if keep.sum() == 0:
            continue

        acc = accuracy_score(
            y_true[keep],
            np.round(y_pred[keep])
        )

        coverages.append(keep.mean())
        accuracies.append(acc)

    plt.figure(figsize=(5, 4))
    plt.plot(coverages, accuracies, marker="o")
    plt.xlabel("Coverage (Fraction of Patients Kept)")
    plt.ylabel("Accuracy on Kept Patients")
    plt.title("Deferral Curve (Uncertainty-Aware)")
    plt.grid(True)
    plt.show()
