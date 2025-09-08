"""
Train Tier 1 RF model for generation mechanism classification
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score
import seaborn as sns
import matplotlib.pyplot as plt

# Load the synthetic samples
df_train = pd.read_parquet("./data/processed/synthetic_samples.parquet")
df_test = pd.read_parquet("./data/processed/test_samples.parquet")

feature_cols = [
    col for col in df_train.columns if col not in ["generation_mechanism", "code"]
]
X_train = df_train[feature_cols]
y_train = df_train["generation_mechanism"]

print("train value counts", y_train.value_counts())

X_test = df_test[feature_cols]
y_test = df_test["generation_mechanism"]

print("test value counts", y_test.value_counts())


print("Training...")
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    min_samples_split=5,
    min_samples_leaf=2,
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

class_report = classification_report(y_test, y_pred)
print(class_report)

cm = confusion_matrix(y_test, y_pred)
cm_normalized = cm / cm.sum(axis=1)[:, np.newaxis]

cm_normalized = cm / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(
    cm_normalized,
    annot=True,
    fmt=".2f",
    xticklabels=rf.classes_,
    yticklabels=rf.classes_,
    cmap="Blues",
    vmin=0,
    vmax=1,
)  # Set scale 0-1
plt.title("Confusion Matrix - Generation Mechanism Classification (Normalized)")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("models/tier1_confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.show()


# Cross-validation
cv_scores = cross_val_score(rf, X_train, y_train, cv=5, n_jobs=-1)
print(f"CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
