# analyze_training.py
# analyze_training.py

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ==== CONFIG ====
LOG_FILE = "training_log_pytorch.csv"  # path to your log CSV
OUT_DIR = "model_analysis"             # folder where plots & summary will be saved


def pick_column(df, preferred, fallback_list):
    """
    Try to get column 'preferred', otherwise look for the first
    existing name in fallback_list. Raise if nothing is found.
    """
    if preferred in df.columns:
        return preferred
    for name in fallback_list:
        if name in df.columns:
            return name
    raise ValueError(f"Could not find any of { [preferred] + fallback_list } in columns: {list(df.columns)}")


def main():
    # 1. Create output folder
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Saving analysis to: {os.path.abspath(OUT_DIR)}")

    # 2. Load training log
    df = pd.read_csv(LOG_FILE)
    print("\nFirst rows of log:")
    print(df.head())
    print("\nColumns:", list(df.columns))

    # ---- Map column names (handles both 'TrainLoss' and 'Train Loss' styles) ----
    epoch_col = pick_column(df, "Epoch", [])

    train_loss_col = pick_column(df, "Train Loss", ["TrainLoss", "train_loss"])
    val_loss_col   = pick_column(df, "Val Loss",   ["ValLoss", "val_loss"])

    train_dice_col = pick_column(df, "Train Dice", ["TrainDice", "train_dice"])
    val_dice_col   = pick_column(df, "Val Dice",   ["ValDice", "val_dice"])

    time_col = None
    for cand in ["Time (s)", "TimeSec", "time_sec"]:
        if cand in df.columns:
            time_col = cand
            break

    # 3. Extract arrays
    epochs     = df[epoch_col].values
    train_loss = df[train_loss_col].values.astype(float)
    val_loss   = df[val_loss_col].values.astype(float)
    train_dice = df[train_dice_col].values.astype(float)
    val_dice   = df[val_dice_col].values.astype(float)

    # 4. Plot Loss curves
    plt.figure()
    plt.plot(epochs, train_loss, marker="o", label="Train loss")
    plt.plot(epochs, val_loss, marker="o", label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    loss_path = os.path.join(OUT_DIR, "loss_curves.png")
    plt.savefig(loss_path, dpi=200)
    plt.close()
    print(f"Saved loss curves to: {loss_path}")

    # 5. Plot Dice curves
    plt.figure()
    plt.plot(epochs, train_dice, marker="o", label="Train Dice")
    plt.plot(epochs, val_dice, marker="o", label="Val Dice")
    plt.xlabel("Epoch")
    plt.ylabel("Dice coefficient")
    plt.title("Training vs Validation Dice")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    dice_path = os.path.join(OUT_DIR, "dice_curves.png")
    plt.savefig(dice_path, dpi=200)
    plt.close()
    print(f"Saved dice curves to: {dice_path}")

    # 6. Optional: epoch time histogram if present
    if time_col is not None:
        times = df[time_col].values.astype(float)
        plt.figure()
        plt.bar(epochs, times)
        plt.xlabel("Epoch")
        plt.ylabel("Time (s)")
        plt.title("Epoch duration")
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        time_path = os.path.join(OUT_DIR, "epoch_times.png")
        plt.savefig(time_path, dpi=200)
        plt.close()
        print(f"Saved epoch time plot to: {time_path}")
    else:
        print("No time column found, skipping epoch time plot.")

    # 7. Summary stats & best epoch
    best_val_loss_idx = int(np.argmin(val_loss))
    best_val_loss_epoch = int(epochs[best_val_loss_idx])
    best_val_loss = float(val_loss[best_val_loss_idx])
    best_val_dice = float(val_dice[best_val_loss_idx])

    best_val_dice_idx = int(np.argmax(val_dice))
    best_dice_epoch = int(epochs[best_val_dice_idx])
    best_dice_value = float(val_dice[best_val_dice_idx])

    summary_lines = []
    summary_lines.append("=== Training Summary ===\n")
    summary_lines.append(f"Num epochs logged: {len(epochs)}\n")
    summary_lines.append(f"Best val loss: {best_val_loss:.4f} at epoch {best_val_loss_epoch}\n")
    summary_lines.append(f"Val Dice at that epoch: {best_val_dice:.4f}\n\n")
    summary_lines.append(f"Best val Dice: {best_dice_value:.4f} at epoch {best_dice_epoch}\n")
    summary_lines.append(f"Train Dice at that epoch: {train_dice[best_val_dice_idx]:.4f}\n\n")

    summary_lines.append("Loss (train/val):\n")
    summary_lines.append(df[[epoch_col, train_loss_col, val_loss_col]].to_string(index=False))
    summary_lines.append("\n\nDice (train/val):\n")
    summary_lines.append(df[[epoch_col, train_dice_col, val_dice_col]].to_string(index=False))
    summary_text = "\n".join(summary_lines)

    summary_path = os.path.join(OUT_DIR, "training_summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary_text)
    print(f"\nSummary written to: {summary_path}")


if __name__ == "__main__":
    main()