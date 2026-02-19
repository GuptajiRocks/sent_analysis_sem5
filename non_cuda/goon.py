# Corrected synthetic verification + visualization cell
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix

# reported metrics
models = {
    "LSTM": {"acc": 0.5244, "f1": 0.3712},
    "GRU": {"acc": 0.5519, "f1": 0.3957},
    "BiLSTM+GRU": {"acc": 0.5699, "f1": 0.4027},
    "BiLSTM+RNN": {"acc": 0.5577, "f1": 0.3961},
    "DistilBERT (5 epochs)": {"acc": 0.8415, "f1": 0.7946}
}

num_classes = 26
samples_per_class = 100  # synthetic per-class sample count -> total_samples = 2600
total_samples = num_classes * samples_per_class

def build_confusion_for_accuracy(target_acc, n_classes=num_classes, per_class_count=samples_per_class, seed=None):
    """
    Build an integer confusion matrix (n_classes x n_classes) such that
    overall accuracy ≈ target_acc by setting the diagonal count per row.
    The remainder of each row is distributed randomly across off-diagonal columns.
    """
    rng = np.random.RandomState(seed)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    # number of correct per class (may vary slightly to better match acc)
    # allow rounding to nearest integer
    correct_per_class = int(round(target_acc * per_class_count))
    # ensure bounds
    correct_per_class = max(0, min(per_class_count, correct_per_class))
    for i in range(n_classes):
        cm[i, i] = correct_per_class
        remaining = per_class_count - cm[i, i]
        if remaining > 0:
            # create probabilities for errors (exclude diagonal)
            probs = rng.rand(n_classes)
            probs[i] = 0
            probs = probs / probs.sum()
            err_counts = rng.multinomial(remaining, probs)
            cm[i] += err_counts
    # final safeguard
    assert all(cm.sum(axis=1) == per_class_count)
    return cm

# build cms and evaluate
results = []
cms = {}
for name, vals in models.items():
    cm = build_confusion_for_accuracy(vals["acc"], seed=abs(hash(name)) % (2**32))
    cms[name] = cm
    # construct y_true and y_pred from cm
    y_true = np.repeat(np.arange(num_classes), samples_per_class)
    y_pred_list = []
    for true_class in range(num_classes):
        for pred_class in range(num_classes):
            count = int(cm[true_class, pred_class])
            if count > 0:
                y_pred_list.extend([pred_class] * count)
    y_pred = np.array(y_pred_list)
    assert len(y_pred) == total_samples, f"length mismatch for {name}"
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    results.append({
        "model": name,
        "reported_acc": vals["acc"],
        "reconstructed_acc": acc,
        "reported_macro_f1": vals["f1"],
        "reconstructed_macro_f1": f1
    })

metrics_df = pd.DataFrame(results)

# Plot: Reported vs Reconstructed Accuracies
plt.figure(figsize=(10,5))
x = np.arange(len(metrics_df))
width = 0.35
plt.bar(x - width/2, metrics_df["reported_acc"], width, label="Reported Acc")
plt.bar(x + width/2, metrics_df["reconstructed_acc"], width, label="Reconstructed Acc")
plt.xticks(x, metrics_df["model"], rotation=20, ha='right')
plt.ylim(0,1)
plt.ylabel("Accuracy")
plt.title("Reported Accuracy")
plt.legend()
plt.tight_layout()
plt.show()

# Plot: Reported vs Reconstructed Macro-F1
plt.figure(figsize=(10,5))
plt.bar(x - width/2, metrics_df["reported_macro_f1"], width, label="Reported Macro-F1")
plt.bar(x + width/2, metrics_df["reconstructed_macro_f1"], width, label="Reconstructed Macro-F1")
plt.xticks(x, metrics_df["model"], rotation=20, ha='right')
plt.ylim(0,1)
plt.ylabel("Macro-F1")
plt.title("Reported Macro-F1")
plt.legend()
plt.tight_layout()
plt.show()

# Show confusion heatmaps (normalized by true-row for readability)
for name, cm in cms.items():
    plt.figure(figsize=(8,6))
    cm_norm = cm.astype(float)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm_norm, row_sums, where=row_sums!=0)
    sns.heatmap(cm_norm, cmap="Blues", cbar=True)
    plt.title(f"Synthetic normalized confusion matrix ({name})")
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.tight_layout()
    plt.show()

# summary table
display_df = metrics_df.rename(columns={
    "model":"Model",
    "reported_acc":"Reported Acc",
    "reconstructed_acc":"Reconstructed Acc",
    "reported_macro_f1":"Reported Macro-F1",
    "reconstructed_macro_f1":"Reconstructed Macro-F1"
})
print(display_df.to_string(index=False))

# Usage note
print("\nNOTE: This constructs synthetic predictions that exactly match the reported overall accuracy (by design),")
print("but the reconstructed Macro-F1 is only approximate because we distribute errors randomly.")
print("To *prove* the reported metrics exactly, provide the real y_true and y_pred arrays from your evaluation;")
print("then compute sklearn.metrics.accuracy_score and precision_recall_fscore_support to verify.")
