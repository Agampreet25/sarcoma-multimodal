import torch
import torch.nn as nn
from torch.optim import Adam

def train_model(model, loader, epochs=10):
    model.train()
    optimizer = Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = nn.BCELoss()

    for _ in range(epochs):
        for mri, pet, label in loader:
            optimizer.zero_grad()
            pred = model(mri, pet).squeeze(dim=1)
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()
