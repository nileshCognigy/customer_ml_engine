import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import log_loss, accuracy_score

def run_day2_experiment():
    print("=" * 60)
    print("DAY 2: LOSS FUNCTIONS & OVERFITTING DYNAMICS")
    print("=" * 60)

    # 1. Generate a noisy classification dataset (simulating realistic edge cases)
    X, y = make_classification(
        n_samples=1200,
        n_features=20,
        n_informative=8,
        n_redundant=4,
        flip_y=0.15,  # 15% random label noise (mimics messy customer data)
        random_state=42
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # 2. Track metrics across different tree depths (complexity)
    depths = list(range(1, 16))
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for depth in depths:
        # Train decision tree with increasing complexity
        model = DecisionTreeClassifier(max_depth=depth, random_state=42)
        model.fit(X_train, y_train)

        # Probabilities for log-loss (Cross-Entropy)
        train_probs = model.predict_proba(X_train)
        val_probs = model.predict_proba(X_val)

        # Predictions for accuracy
        train_preds = model.predict(X_train)
        val_preds = model.predict(X_val)

        # Compute Cross-Entropy Log Loss
        train_losses.append(log_loss(y_train, train_probs))
        val_losses.append(log_loss(y_val, val_probs))

        train_accs.append(accuracy_score(y_train, train_preds))
        val_accs.append(accuracy_score(y_val, val_preds))

    # 3. Print the Diagnostic Comparison
    results_df = pd.DataFrame({
        "Depth": depths,
        "Train Acc": np.round(train_accs, 3),
        "Val Acc": np.round(val_accs, 3),
        "Train Loss": np.round(train_losses, 3),
        "Val Loss": np.round(val_losses, 3)
    })
    
    print("\nModel Depth Progression:")
    print(results_df.to_string(index=False))

    # Identify the sweet spot (minimum validation loss)
    optimal_depth = depths[np.argmin(val_losses)]
    print(f"\n--> Sweet Spot (Optimal Regularization): max_depth = {optimal_depth}")
    print(f"--> Overfit Zone: max_depth >= 10 (Notice Train Acc approaches 1.0 while Val Loss explodes!)")

    # 4. Generate Diagnostic Loss Curve Plot
    plt.figure(figsize=(12, 5))

    # Subplot 1: Cross-Entropy Loss Curve
    plt.subplot(1, 2, 1)
    plt.plot(depths, train_losses, label="Training Loss (BCE)", marker="o", color="blue")
    plt.plot(depths, val_losses, label="Validation Loss (BCE)", marker="s", color="red")
    plt.axvline(x=optimal_depth, color="green", linestyle="--", label=f"Optimal Depth ({optimal_depth})")
    plt.title("Cross-Entropy Loss vs. Model Complexity")
    plt.xlabel("Tree Depth (Capacity)")
    plt.ylabel("Log Loss (Lower is Better)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Subplot 2: Accuracy Comparison
    plt.subplot(1, 2, 2)
    plt.plot(depths, train_accs, label="Train Accuracy", marker="o", color="blue")
    plt.plot(depths, val_accs, label="Validation Accuracy", marker="s", color="red")
    plt.axvline(x=optimal_depth, color="green", linestyle="--", label=f"Optimal Depth ({optimal_depth})")
    plt.title("Accuracy: Memorization vs Generalization")
    plt.xlabel("Tree Depth (Capacity)")
    plt.ylabel("Accuracy Score")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_filename = "loss_curve.png"
    plt.savefig(plot_filename, dpi=300)
    print(f"\nDiagnostic plot saved to: {plot_filename}")

if __name__ == "__main__":
    run_day2_experiment()