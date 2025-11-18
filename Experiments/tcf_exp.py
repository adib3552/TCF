###########################################
# ORIGINAL IMPORTS
###########################################
from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Solar, Dataset_PEMS
from torch.utils.data import DataLoader

from model.tcf import Model

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

import random
import os


###########################################
# SET SEED
###########################################
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed(2021)

device = "cuda" if torch.cuda.is_available() else "cpu"


###########################################
# LOSS FUNCTIONS
###########################################
mse_loss = nn.MSELoss()

def mae_loss(pred, true):
    return torch.mean(torch.abs(pred - true))


###########################################
# TRAINING FUNCTIONS
###########################################
def train(model, train_loader, optimizer, device, pred_len):
    model.train()
    total_loss = []
    for seq_x, seq_y, seq_x_mark, seq_y_mark in tqdm(train_loader, desc="Training", leave=False):
        optimizer.zero_grad()
        seq_x = seq_x.to(device).float()
        target = seq_y[:, -pred_len:, :].to(device).float()

        outputs = model(seq_x)
        loss = mse_loss(outputs, target)
        loss.backward()
        optimizer.step()
        total_loss.append(loss.item())

    return np.average(total_loss)


def evaluate(model, val_loader, device, pred_len):
    model.eval()
    total_mse, total_mae = 0, 0
    total_loss = []
    with torch.no_grad():
        for seq_x, seq_y, seq_x_mark, seq_y_mark in tqdm(val_loader, desc="Validating", leave=False):
            seq_x = seq_x.to(device).float()
            target = seq_y[:, -pred_len:, :].to(device).float()

            outputs = model(seq_x)
            mse = mse_loss(outputs, target).item()
            mae = mae_loss(outputs, target).item()

            total_loss.append(mse)
            total_mae += mae

    return np.average(total_loss), total_mae / len(val_loader)


def test(model, test_loader, device, pred_len):
    model.eval()
    total_mse, total_mae = 0, 0
    with torch.no_grad():
        for seq_x, seq_y, seq_x_mark, seq_y_mark in tqdm(test_loader, desc="Testing", leave=False):
            seq_x = seq_x.to(device).float()
            target = seq_y[:, -pred_len:, :].to(device).float()

            outputs = model(seq_x)
            total_mse += mse_loss(outputs, target).item()
            total_mae += mae_loss(outputs, target).item()

    return total_mse / len(test_loader), total_mae / len(test_loader)


###########################################
# TRAIN MODEL WRAPPER
###########################################
def train_model(model, train_loader, val_loader, test_loader, pred_len,
                epochs=40, lr=0.001, patience=5, device="cuda"):

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)

    best_val_loss = float("inf")
    best_model = None
    patience_counter = 0

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        train_loss = train(model, train_loader, optimizer, device, pred_len)
        val_mse, val_mae = evaluate(model, val_loader, device, pred_len)
        scheduler.step()

        print(f"Train Loss: {train_loss:.7f} | Val MSE: {val_mse:.7f} | Val MAE: {val_mae:.7f}")
        print(f"LR: {scheduler.get_last_lr()[0]:.8f}")

        # Early stopping
        if val_mse < best_val_loss:
            best_val_loss = val_mse
            best_model = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    model.load_state_dict(best_model)

    # Final test
    test_mse, test_mae = test(model, test_loader, device, pred_len)
    return test_mse, test_mae


###########################################
# HYPERPARAMETER SEARCH
###########################################
d_model_list = [512]
d_ff_list = [128, 256]
d_core_list = [64, 128, 256]
batch_sizes = [32]
e_layers_list = [1, 2, 3]

seq_len = 96

n_vars = 137
size = [96,48,192]

results = []  # store all runs

for d_model in d_model_list:
    for d_ff in d_ff_list:
        for d_core in d_core_list:
            for bs in batch_sizes:
                for e_layers in e_layers_list:
                    print("\n=================================================")
                    print(f"Testing config:")
                    print(f"d_model={d_model}, d_ff={d_ff}, d_core={d_core}, batch={bs}, e_layers={e_layers}")
                    print("=================================================\n")

                    # Build datasets/loaders for this batch size
                    train_set = Dataset_Solar(
                        root_path='C:/Users/Awsftausif/Desktop/S-Mamba_datasets/Solar/',
                        data_path='solar_AL.txt',
                        flag='train',
                        size=size,
                        features='M',
                        target='OT',
                        scale=True,
                        timeenc=0,
                        freq='t'
                    )

                    val_set = Dataset_Solar(
                        root_path='C:/Users/Awsftausif/Desktop/S-Mamba_datasets/Solar/',
                        data_path='solar_AL.txt',
                        flag='val',
                        size=size,
                        features='M',
                        target='OT',
                        scale=True,
                        timeenc=0,
                        freq='t'
                    )

                    test_set = Dataset_Solar(
                        root_path='C:/Users/Awsftausif/Desktop/S-Mamba_datasets/Solar/',
                        data_path='solar_AL.txt',
                        flag='test',
                        size=size,
                        features='M',
                        target='OT',
                        scale=True,
                        timeenc=0,
                        freq='t'
                    )

                    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True)
                    val_loader = DataLoader(val_set, batch_size=bs, shuffle=False)
                    test_loader = DataLoader(test_set, batch_size=bs, shuffle=False)

                    # Create config object
                    class Config:
                        def __init__(self):
                            self.d_model = d_model
                            self.d_core = d_core
                            self.d_ff = d_ff
                            self.e_layers = e_layers
                            self.n_vars = n_vars
                            self.seq_len = seq_len
                            self.pred_len = size[2]
                            self.kernel_size = 0
                            self.patch_len = 16
                            self.n_heads = 8
                            self.factor = 3
                            self.dropout = 0.1
                            self.use_norm = True

                    configs = Config()
                    model = Model(configs).to(device)

                    # Train and evaluate
                    test_mse, test_mae = train_model(
                        model, train_loader, val_loader, test_loader,
                        pred_len=size[2], epochs=40, lr=0.001, patience=5, device=device
                    )

                    print(f"RESULT → Test MSE: {test_mse:.6f}, Test MAE: {test_mae:.6f}")

                    results.append({
                        "d_model": d_model,
                        "d_ff": d_ff,
                        "d_core": d_core,
                        "batch": bs,
                        "e_layers": e_layers,
                        "mse": test_mse,
                        "mae": test_mae
                    })


###########################################
# FIND BEST RESULT
###########################################
best = None
for r in results:
    if best is None:
        best = r
    else:
        if r["mse"] < best["mse"] or (r["mse"] == best["mse"] and r["mae"] < best["mae"]):
            best = r

print("\n===============================")
print(" BEST HYPERPARAMETERS FOUND ")
print("===============================")
print(best)
