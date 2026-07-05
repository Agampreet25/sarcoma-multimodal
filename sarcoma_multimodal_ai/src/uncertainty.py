import numpy as np
import torch

def mc_dropout_predict(model, mri, pet, T=50):
    model.train()  # dropout ON
    preds = []

    with torch.no_grad():
        for _ in range(T):
            preds.append(model(mri, pet).item())

    preds = np.array(preds)
    return preds.mean(), preds.std()
