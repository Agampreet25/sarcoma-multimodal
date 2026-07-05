import os
import numpy as np
import torch
import pydicom
from src.utils import find_series_by_modality

class SarcomaDataset(torch.utils.data.Dataset):
    """
    One patient = one sample
    """

    def __init__(self, patient_ids, labels, root_dir):
        self.patient_ids = patient_ids
        self.labels = labels
        self.root_dir = root_dir

    def _load_dicom_volume(self, series_dir):
        slices = []

        for f in sorted(os.listdir(series_dir)):
            if f.endswith(".dcm"):
                dcm = pydicom.dcmread(os.path.join(series_dir, f))
                slices.append(dcm.pixel_array)

        volume = np.stack(slices).astype(np.float32)
        volume = (volume - volume.mean()) / (volume.std() + 1e-6)
        return volume

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        pid = self.patient_ids[idx]
        label = self.labels[pid]

        patient_dir = os.path.join(self.root_dir, pid)

        # 🔍 Auto-detect series
        mri_series = find_series_by_modality(patient_dir, "MR")
        pet_series = find_series_by_modality(patient_dir, "PT")

        if len(mri_series) == 0 or len(pet_series) == 0:
            raise RuntimeError(f"Missing MRI or PET for {pid}")

        # ✔ Safe baseline: use first detected series
        mri_vol = self._load_dicom_volume(mri_series[0])
        pet_vol = self._load_dicom_volume(pet_series[0])

        mri_tensor = torch.tensor(mri_vol).unsqueeze(0)
        pet_tensor = torch.tensor(pet_vol).unsqueeze(0)

        return mri_tensor, pet_tensor, torch.tensor(label, dtype=torch.float32)
