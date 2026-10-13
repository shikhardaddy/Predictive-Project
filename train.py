"""
Handwritten Digit Recognition using Linear Discriminant Analysis (LDA)
with a Quadratic Discriminant Analysis (QDA) comparison
Course: Predictive Analytics

Dataset: Optical Recognition of Handwritten Digits (UCI ML Repository)
         8x8 images, 10 classes (digits 0-9). Bundled with scikit-learn.

Pipeline:
  1. Load data & basic EDA (sample images, class balance)
  2. Stratified train/test split
  3. LDA with tuned shrinkage (GridSearchCV)  -> main model
  4. QDA with tuned reg_param (GridSearchCV)  -> comparison model
  5. Evaluate: accuracy, macro F1, per-class report, confusion matrix
  6. LDA as dimensionality reduction: 2-D discriminant projection plot
  7. Save plots, metrics and the trained model to ./outputs
"""

import json
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split

RANDOM_STATE = 42
OUT = "outputs"
os.makedirs(OUT, exist_ok=True)

# ------------------------------------------------------------------ 1. DATA
digits = load_digits()
X, y = digits.data, digits.target
print("Shape:", X.shape, "| classes:", len(np.unique(y)))
print("Class counts:", dict(zip(*np.unique(y, return_counts=True))))

# ------------------------------------------------------------------ 2. EDA
fig, axes = plt.subplots(2, 10, figsize=(14, 3.4))
for d in range(10):
    idx = np.where(y == d)[0][:2]
    for r, i in enumerate(idx):
        axes[r, d].imshow(digits.images[i], cmap="gray_r")
        axes[r, d].set_title(str(d) if r == 0 else "", fontsize=10)
        axes[r, d].axis("off")
plt.suptitle("Sample 8x8 digit images")
plt.tight_layout()
plt.savefig(f"{OUT}/01_sample_digits.png", dpi=150)
plt.close()

plt.figure(figsize=(6, 4))
counts = np.bincount(y)
plt.bar(range(10), counts, color="#4C78A8")
plt.xticks(range(10))
plt.xlabel("Digit")
plt.ylabel("Samples")
plt.title("Class distribution")
plt.tight_layout()
plt.savefig(f"{OUT}/02_class_distribution.png", dpi=150)
plt.close()

mean_img = np.array([digits.images[y == d].mean(axis=0) for d in range(10)])
fig, axes = plt.subplots(1, 10, figsize=(14, 1.9))
for d in range(10):
    axes[d].imshow(mean_img[d], cmap="gray_r")
    axes[d].set_title(str(d))
    axes[d].axis("off")
plt.suptitle("Average image per class")
plt.tight_layout()
plt.savefig(f"{OUT}/03_class_mean_images.png", dpi=150)
plt.close()

# ------------------------------------------------------------------ 3. SPLIT + MODELS
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# LDA: shrinkage regularises the covariance matrix (needs solver lsqr/eigen)
lda_grid = GridSearchCV(
    LinearDiscriminantAnalysis(solver="lsqr"),
    {"shrinkage": [0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]},
    cv=cv, scoring="accuracy", n_jobs=-1,
).fit(X_train, y_train)
lda = lda_grid.best_estimator_
print("\nLDA best shrinkage:", lda_grid.best_params_["shrinkage"], "| CV accuracy:", round(lda_grid.best_score_, 4))

# QDA: pixels that are constant give singular covariances, so reg_param is essential
qda_grid = GridSearchCV(
    QuadraticDiscriminantAnalysis(),
    {"reg_param": [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]},
    cv=cv, scoring="accuracy", n_jobs=-1,
).fit(X_train, y_train)
qda = qda_grid.best_estimator_
print("QDA best reg_param:", qda_grid.best_params_["reg_param"], "| CV accuracy:", round(qda_grid.best_score_, 4))

lda_cv = cross_val_score(lda, X_train, y_train, cv=cv, scoring="accuracy")

# ------------------------------------------------------------------ 4. EVALUATION
def evaluate(model):
    pred = model.predict(X_test)
    return pred, {
        "test_accuracy": float(accuracy_score(y_test, pred)),
        "test_f1_macro": float(f1_score(y_test, pred, average="macro")),
    }

lda_pred, lda_m = evaluate(lda)
qda_pred, qda_m = evaluate(qda)

metrics = {
    "LDA": {"best_shrinkage": lda_grid.best_params_["shrinkage"],
            "cv_accuracy_mean": float(lda_cv.mean()), "cv_accuracy_std": float(lda_cv.std()), **lda_m},
    "QDA": {"best_reg_param": qda_grid.best_params_["reg_param"],
            "cv_accuracy_mean": float(qda_grid.best_score_), **qda_m},
    "n_train": int(len(X_train)),
    "n_test": int(len(X_test)),
}
with open(f"{OUT}/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("\n=== LDA (main model) ===")
print(f"5-fold CV accuracy: {lda_cv.mean():.4f} +/- {lda_cv.std():.4f}")
print(f"Test accuracy: {lda_m['test_accuracy']:.4f} | macro F1: {lda_m['test_f1_macro']:.4f}")
print(classification_report(y_test, lda_pred, digits=3))
print("=== QDA (comparison) ===")
print(f"Test accuracy: {qda_m['test_accuracy']:.4f} | macro F1: {qda_m['test_f1_macro']:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
for ax, pred, title in zip(axes, [lda_pred, qda_pred], ["LDA", "QDA"]):
    ConfusionMatrixDisplay(confusion_matrix(y_test, pred)).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"{title} confusion matrix (test accuracy {accuracy_score(y_test, pred):.3f})")
plt.tight_layout()
plt.savefig(f"{OUT}/04_confusion_matrices.png", dpi=150)
plt.close()

# Shrinkage / reg_param tuning curves
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, grid, key, name in [(axes[0], lda_grid, "param_shrinkage", "LDA shrinkage"),
                            (axes[1], qda_grid, "param_reg_param", "QDA reg_param")]:
    ax.plot(grid.cv_results_[key].astype(float), grid.cv_results_["mean_test_score"], "o-", color="#F58518")
    ax.set_xlabel(name)
    ax.set_ylabel("CV accuracy")
    ax.set_title(f"Tuning {name}")
plt.tight_layout()
plt.savefig(f"{OUT}/05_tuning_curves.png", dpi=150)
plt.close()

# ------------------------------------------------------------------ 5. LDA AS DIMENSIONALITY REDUCTION
proj = LinearDiscriminantAnalysis(n_components=2).fit(X_train, y_train)
Z = proj.transform(X_test)
plt.figure(figsize=(7.5, 6))
sc = plt.scatter(Z[:, 0], Z[:, 1], c=y_test, cmap="tab10", s=22)
plt.colorbar(sc, ticks=range(10), label="Digit")
plt.xlabel("Linear discriminant 1")
plt.ylabel("Linear discriminant 2")
plt.title("Test digits projected onto the first two discriminants")
plt.tight_layout()
plt.savefig(f"{OUT}/06_lda_projection.png", dpi=150)
plt.close()
print("\nExplained variance ratio of first 2 discriminants:", proj.explained_variance_ratio_[:2].round(3))

# ------------------------------------------------------------------ 6. SAVE MODEL
joblib.dump(lda, f"{OUT}/lda_model.joblib")
joblib.dump(qda, f"{OUT}/qda_model.joblib")
print(f"\nAll outputs saved in ./{OUT}/")
