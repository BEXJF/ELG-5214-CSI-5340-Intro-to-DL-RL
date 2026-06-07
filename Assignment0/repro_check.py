"""
Ethan ******
*********
ELG 5214 
Assignment 0
Jan 23, 2026

"""

import os
import random
import math

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from datetime import datetime # Generation timestamp added for plots


# Load config from config.yaml
def load_config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# Reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_deterministic():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True) #throw errors if an op is nondeterministic
    except Exception:
        pass

# I trained on CPU but maybe the testing environment can ultrilize CUDA
def get_device(device_cfg):
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_cfg)


# MLP model
class MLP(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 10)
        )

    def forward(self, x):
        return self.net(x)

# Data
def get_train_loader(batch_size):
    transform = transforms.ToTensor()
    dataset = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=transform
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)


# Training
def accuracy(logits, labels):
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


def train_one_run(run_id, seed, cfg, device, paths):
    set_seed(seed)

    loader = get_train_loader(cfg["batch_size"])
    model = MLP(cfg["hidden_dim"]).to(device)

    optimizer = optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    loss_fn = nn.CrossEntropyLoss()

    total_iters = cfg["epochs"] * len(loader)
    half_iter = total_iters // 2
    step = 0

    losses = []
    accs = []

    log_file = os.path.join(paths["logs"], f"run{run_id}.log")

    with open(log_file, "w") as f:
        f.write("iter,epoch,batch,loss,accuracy\n")

        model.train()
        for epoch in range(cfg["epochs"]):
            for batch_idx, (x, y) in enumerate(loader):
                step += 1

                x = x.to(device)
                y = y.to(device)

                logits = model(x)
                loss = loss_fn(logits, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                acc = accuracy(logits.detach(), y)

                losses.append(loss.item())
                accs.append(acc)

                f.write(f"{step},{epoch+1},{batch_idx+1},{loss.item():.6f},{acc:.6f}\n")

                if step == half_iter:
                    torch.save(
                        model.state_dict(),
                        os.path.join(paths["checkpoints"], f"run{run_id}_half.pth")
                    )

    torch.save(
        model.state_dict(),
        os.path.join(paths["checkpoints"], f"run{run_id}_final.pth")
    )

    return np.array(losses), np.array(accs)


# Plotting diagrams
def compute_mean_std_se(data):
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    se = std / math.sqrt(data.shape[0])
    return mean, std, se


def plot_mean_band(x, mean, band, title, ylabel, band_label, ax):
    ax.plot(x, mean)
    ax.fill_between(x, mean - band, mean + band, alpha=0.2, label=band_label)
    ax.set_title(title)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    ax.legend()


def save_mean_plot(x, mean_loss, band_loss, mean_acc, band_acc, title, band_label, path):
    fig = plt.figure(figsize=(10, 4))

    ax1 = fig.add_subplot(1, 2, 1)
    plot_mean_band(x, mean_loss, band_loss, "Training Loss", "Loss", band_label, ax1)

    ax2 = fig.add_subplot(1, 2, 2)
    plot_mean_band(x, mean_acc, band_acc, "Training Accuracy", "Accuracy", band_label, ax2)

    fig.suptitle(title)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fig.text(0.99, 0.01,
            f"Generated on: {timestamp}",
            ha="right", va="bottom", fontsize=5)
    
    fig.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close(fig)


def save_all_runs_plot(x, loss_runs, acc_runs, path):
    fig = plt.figure(figsize=(10, 4))

    ax1 = fig.add_subplot(1, 2, 1)
    for i in range(loss_runs.shape[0]):
        ax1.plot(x, loss_runs[i], label=f"run{i+1}")
    ax1.set_title("Training Loss")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Loss")
    ax1.legend()

    ax2 = fig.add_subplot(1, 2, 2)
    for i in range(acc_runs.shape[0]):
        ax2.plot(x, acc_runs[i], label=f"run{i+1}")
    ax2.set_title("Training Accuracy")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Accuracy")
    ax2.legend()

    fig.suptitle("All Individual Learning Curves (5 runs)")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fig.text(0.99, 0.01,
        f"Generated on: {timestamp}",
        ha="right", va="bottom", fontsize=5)

    fig.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close(fig)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# Main
def main():

    print("running ......")
    cfg = load_config()
    set_deterministic()
    device = get_device(cfg["device"])

    paths = {
        "logs": "outputs/logs",
        "checkpoints": "outputs/checkpoints",
        "plots": "outputs/plots"
    }

    for p in paths.values():
        os.makedirs(p, exist_ok=True)

    all_losses = []
    all_accs = []

    for run in range(1, cfg["num_runs"] + 1):
        seed = cfg["base_seed"] + run
        print(f"Running experiment {run} with seed {seed}")

        losses, accs = train_one_run(run, seed, cfg, device, paths)
        all_losses.append(losses)
        all_accs.append(accs)

    all_losses = np.stack(all_losses)
    all_accs = np.stack(all_accs)

    x = np.arange(1, all_losses.shape[1] + 1)

    mean_l, std_l, se_l = compute_mean_std_se(all_losses)
    mean_a, std_a, se_a = compute_mean_std_se(all_accs)

    save_mean_plot(
        x, mean_l, std_l, mean_a, std_a,
        "Mean ± Standard Deviation (5 runs)",
        "± 1 SD",
        "outputs/plots/mean_std.png"
    )

    save_mean_plot(
        x, mean_l, se_l, mean_a, se_a,
        "Mean ± Standard Error (5 runs)",
        "± 1 SE",
        "outputs/plots/mean_se.png"
    )

    save_all_runs_plot(
        x, all_losses, all_accs,
        "outputs/plots/all_runs.png"
    )

    print("Training complete. This program export 3 plots, model checkpoints, and output logs as required in Assignment description.")
    print("ELG5214, Assignment 0 by Ethan Xujia Fan.")


if __name__ == "__main__":
    main()
