import json
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from models import EXPERIMENTS, get_model

# ------------------------
# Config
# ------------------------
DATA_DIR = "PokemonData"  # dataset path
CHECKPOINT_DIR = "checkpoints"
RESULTS_DIR = "results"
BATCH_SIZE = 32
NUM_WORKERS = 4
VAL_RATIO = 0.1
TEST_RATIO = 0.1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"Using device: {DEVICE}")


# ------------------------
# Data
# ------------------------
def get_dataloaders(data_dir: str):
    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # Load full dataset with train transform (will override for val/test)
    full_dataset = datasets.ImageFolder(data_dir, transform=train_transform)

    # Split into train / val / test
    n = len(full_dataset)
    n_val = int(n * VAL_RATIO)
    n_test = int(n * TEST_RATIO)
    n_train = n - n_val - n_test
    train_set, val_set, test_set = random_split(
        full_dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42),
    )

    # Apply eval transform to val/test
    val_set.dataset.transform = eval_transform
    test_set.dataset.transform = eval_transform

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    print(f"Dataset: {n_train} train / {n_val} val / {n_test} test")
    print(f"Classes: {len(full_dataset.classes)}")
    return train_loader, val_loader, test_loader, full_dataset.classes


# ------------------------
# Train / Eval
# ------------------------
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += imgs.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        outputs = model(imgs)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return total_loss / total, correct / total, all_preds, all_labels


# ------------------------
# Learning Curve
# ------------------------
def save_learning_curve(name, train_losses, val_losses, train_accs, val_accs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(train_losses) + 1)

    ax1.plot(epochs, train_losses, label="Train")
    ax1.plot(epochs, val_losses, label="Val")
    ax1.set_title(f"{name} — Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()

    ax2.plot(epochs, train_accs, label="Train")
    ax2.plot(epochs, val_accs, label="Val")
    ax2.set_title(f"{name} — Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f"{name}_curve.png")
    plt.savefig(path)
    plt.close()
    print(f"  Saved learning curve: {path}")


# ------------------------
# Main Training Loop
# ------------------------
def train_experiment(name, config, train_loader, val_loader, test_loader):
    print(f"\n{'='*60}")
    print(f"Experiment: {config['description']}")
    print(f"{'='*60}")

    model = get_model(name).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=config["lr"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["epochs"]
    )

    best_val_acc = 0.0
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{name}_best.pth")

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(1, config["epochs"] + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer
        )
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion)
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(
            f"  Epoch [{epoch:02d}/{config['epochs']}] "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
        )

        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  ✓ Best model saved (val_acc: {best_val_acc:.4f})")

    # Load best model and evaluate on test set
    model.load_state_dict(torch.load(checkpoint_path))
    test_loss, test_acc, preds, labels = evaluate(model, test_loader, criterion)
    test_precision = precision_score(labels, preds, average="macro", zero_division=0)
    test_recall = recall_score(labels, preds, average="macro", zero_division=0)

    print(f"\n  Test Results:")
    print(f"    Accuracy  : {test_acc:.4f}")
    print(f"    Precision : {test_precision:.4f}")
    print(f"    Recall    : {test_recall:.4f}")

    # Save learning curve
    save_learning_curve(name, train_losses, val_losses, train_accs, val_accs)

    return {
        "description": config["description"],
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "test_precision": test_precision,
        "test_recall": test_recall,
    }


# ------------------------
# Entry Point
# ------------------------
if __name__ == "__main__":
    train_loader, val_loader, test_loader, classes = get_dataloaders(DATA_DIR)

    all_results = {}
    for name, config in EXPERIMENTS.items():
        result = train_experiment(name, config, train_loader, val_loader, test_loader)
        all_results[name] = result

    # Save all results to JSON
    results_path = os.path.join(RESULTS_DIR, "results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=4)
    print(f"\nAll results saved: {results_path}")

    # Print summary table
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(
        f"{'Model':<25} {'Val Acc':>8} {'Test Acc':>9} {'Precision':>10} {'Recall':>8}"
    )
    print("-" * 60)
    for name, r in all_results.items():
        print(
            f"{name:<25} {r['best_val_acc']:>8.4f} {r['test_acc']:>9.4f} "
            f"{r['test_precision']:>10.4f} {r['test_recall']:>8.4f}"
        )
