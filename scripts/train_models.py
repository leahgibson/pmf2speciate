"""
Train RF model on classifying by generation mechanism
"""

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import json


mechanisms = ["Combustion", "Microbial", "Volatilization"]  # "Ash", "Dust",
cat = "generation_mechanism"


def load_data(mechanism, type):
    """
    Loads training or test data

    Parameters:
    mechanism: str
        Name of generation mechanism as it appears in file name
    type: str
        "train" or "test

    Return: pd.DataFrame
    """

    df = pd.read_parquet(f"./data/processed/{mechanism}_{type}.parquet")
    if cat == "generation_mechanism":
        df["generation_mechanism"] = mechanism

    return df


if cat == "generation_mechanism":
    train_dfs = []
    test_dfs = []
    for mechanism in mechanisms:
        train_dfs.append(load_data(mechanism, type="train"))
        test_dfs.append(load_data(mechanism, type="test"))

    train_data = pd.concat(train_dfs)
    test_data = pd.concat(test_dfs)

    X_train = train_data.drop(columns=["source", "generation_mechanism"])
    y_train = train_data["generation_mechanism"]

    X_test = test_data.drop(columns=["source", "generation_mechanism"])
    y_test = test_data["generation_mechanism"]

    print("train value counts", y_train.value_counts())
    print("test value counts", y_test.value_counts())


else:
    train_data = load_data(mechanism=cat, type="train")
    test_data = load_data(mechanism=cat, type="test")

    X_train = train_data.drop(columns=["source"])
    y_train = train_data["source"]

    X_test = test_data.drop(columns=["source"])
    y_test = test_data["source"]

    print("train value counts", y_train.value_counts())
    print("test value counts", y_test.value_counts())


print("Training...")

# Set training params for different models
n_estimators = {
    "generation_mechanism": 250,
    "Combustion": 110,
    "Ash": 60,
    "Dust": 750,
    "Microbial": 55,
    "Volatilization": 100,
}

min_samples_split = {
    "generation_mechanism": 5,
    "Combustion": 5,
    "Ash": 10,
    "Dust": 10,
    "Microbial": 10,
    "Volatilization": 5,
}

min_samples_leaf = {
    "generation_mechanism": 5,
    "Combustion": 3,
    "Ash": 5,
    "Dust": 5,
    "Microbial": 5,
    "Volatilization": 4,
}


rf = RandomForestClassifier(
    n_estimators=n_estimators[cat],
    max_depth=None,
    min_samples_split=min_samples_split[cat],
    min_samples_leaf=min_samples_leaf[cat],
    random_state=42,
    n_jobs=1,
)
rf.fit(X_train, y_train)

print("Evaluating...")

y_pred = rf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy score: {accuracy:.4f}")

train_score = rf.score(X_train, y_train)
print(f"Training score: {train_score:.4f}")

# Cross-validation
cv_scores = cross_val_score(rf, X_train, y_train, cv=5, n_jobs=-1)
print(f"CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

class_report = classification_report(y_test, y_pred, output_dict=True)
print(classification_report(y_test, y_pred))

model_dir = "./src/pmf2speciate/models"
# Save trained model
model_path = os.path.join(model_dir, f"{cat}_random_forest.pkl")
with open(model_path, "wb") as f:
    pickle.dump(rf, f)
print(f"Model saved to: {model_path}")

# Save feature names
feature_cols = X_train.columns
feature_names_path = os.path.join(model_dir, f"{cat}_feature_names.pkl")
with open(feature_names_path, "wb") as f:
    pickle.dump(feature_cols, f)

# Save model metadata and performance
metadata = {
    "model_type": "RandomForestClassifier",
    "category": cat,
    "category_full_name": cat,
    "n_features": len(feature_cols),
    "n_classes": len(rf.classes_),
    "classes": rf.classes_.tolist(),
    "n_train_samples": len(X_train),
    "n_test_samples": len(X_test),
    "train_accuracy": float(train_score),
    "test_accuracy": float(accuracy),
    "cv_mean": float(cv_scores.mean()),
    "cv_std": float(cv_scores.std()),
    "hyperparameters": rf.get_params(),
}
metadata_path = os.path.join(model_dir, f"{cat}_model_summary.json")
with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=2)
print(f"Summary saved to: {metadata_path}")

# Save classification report
class_report_df = pd.DataFrame(class_report).transpose()
class_report_path = os.path.join(model_dir, f"{cat}_classification_report.csv")
class_report_df.to_csv(class_report_path)
print(f"Classification report saved to: {class_report_path}")

cm = confusion_matrix(y_test, y_pred)
cm_normalized = cm / cm.sum(axis=1)[:, np.newaxis]

n_classes = len(rf.classes_)

if n_classes <= 20:
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2f",
        xticklabels=rf.classes_,
        yticklabels=rf.classes_,
        cmap="Blues",
        vmin=0,
        vmax=1,
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

elif n_classes <= 100:
    plt.figure(figsize=(15, 12))
    sns.heatmap(
        cm_normalized,
        annot=False,  # No text annotations
        xticklabels=rf.classes_,
        yticklabels=rf.classes_,
        cmap="Blues",
        vmin=0,
        vmax=1,
        cbar_kws={"label": "Normalized Count"},
    )
    plt.xticks(rotation=90, ha="center", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)

else:
    plt.figure(figsize=(16, 14))
    sns.heatmap(
        cm_normalized,
        annot=False,
        xticklabels=False,  # No x labels
        yticklabels=False,  # No y labels
        cmap="Blues",
        vmin=0,
        vmax=1,
        cbar_kws={"label": "Normalized Count"},
    )

    # Add text indicating number of classes
    plt.text(
        0.02,
        0.98,
        f"{n_classes} classes",
        transform=plt.gca().transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

plt.title(f"Confusion Matrix - {cat.replace('_', ' ').title()}")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()

# Save confusion matrix
cm_path = os.path.join(model_dir, f"{cat}_confusion_matrix.png")
plt.savefig(cm_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"Confusion matrix saved to: {cm_path}")
