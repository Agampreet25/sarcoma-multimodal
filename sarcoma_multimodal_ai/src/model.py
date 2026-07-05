import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv3d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv3d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool3d(2)
        self.gap = nn.AdaptiveAvgPool3d(1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.gap(x)
        return x.view(x.size(0), -1)


class MultimodalBayesianNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.mri_encoder = Encoder3D()
        self.pet_encoder = Encoder3D()

        self.fc1 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, mri, pet):
        mri_feat = self.mri_encoder(mri)
        pet_feat = self.pet_encoder(pet)

        fused = torch.cat([mri_feat, pet_feat], dim=1)
        x = F.relu(self.fc1(fused))
        x = self.dropout(x)
        return torch.sigmoid(self.fc2(x))
