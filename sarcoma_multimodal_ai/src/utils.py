import os
import pydicom

def find_series_by_modality(patient_dir, modality):
    """
    modality: 'MR' or 'PT'
    Returns list of series directories
    """
    series_dirs = set()

    for root, _, files in os.walk(patient_dir):
        for f in files:
            if f.endswith(".dcm"):
                try:
                    dcm = pydicom.dcmread(
                        os.path.join(root, f),
                        stop_before_pixels=True
                    )
                    if dcm.Modality == modality:
                        series_dirs.add(root)
                        break
                except:
                    continue

    return sorted(list(series_dirs))


def set_seed(seed=42):
    import random, numpy as np, torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
