"""
Train an XGBoost classifier to predict product co-purchase likelihood.

Pipeline
--------
1. Load features from X.parquet / y.csv
2. Stratified 5-fold cross-validation to get robust baseline metrics
3. RandomizedSearchCV to tune key hyperparameters
4. Final evaluation on a held-out test set (ROC-AUC, PR-AUC, F1, etc.)
5. Persist model to models/co_purchase_model.json
6. Save evaluation plots to models/
"""
import pathlib, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, RandomizedSearchCV, cross_validate
)
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_score,
    recall_score, f1_score, classification_report, confusion_matrix,
    RocCurveDisplay, PrecisionRecallDisplay,
)

warnings.filterwarnings("ignore")

ROOT   = pathlib.Path(__file__).parent
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)

# ── Data ────────────────────────────────────────────────────────────────────
X = pd.read_parquet(ROOT / "X.parquet")
y = pd.read_csv(ROOT / "y.csv").squeeze()

print(f"Dataset: {X.shape[0]:,} samples × {X.shape[1]} features")
print(f"Label balance: {y.value_counts().to_dict()}\n")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

# ── Cross-validation baseline ────────────────────────────────────────────────
print("── 5-Fold CV baseline ──────────────────────────────────────────────")
base_model = xgb.XGBClassifier(
    n_estimators=200, max_depth=4, learning_rate=0.10,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, eval_metric="logloss",
)
cv_results = cross_validate(
    base_model, X_train, y_train,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring={"roc_auc": "roc_auc", "average_precision": "average_precision"},
    return_train_score=False,
)
print(f"  ROC-AUC : {cv_results['test_roc_auc'].mean():.4f} ± {cv_results['test_roc_auc'].std():.4f}")
print(f"  PR-AUC  : {cv_results['test_average_precision'].mean():.4f} ± {cv_results['test_average_precision'].std():.4f}\n")

# ── Hyperparameter tuning ────────────────────────────────────────────────────
print("── RandomizedSearchCV (50 iterations) ──────────────────────────────")
param_dist = {
    "n_estimators"   : [100, 200, 300, 400],
    "max_depth"      : [3, 4, 5, 6],
    "learning_rate"  : [0.01, 0.05, 0.10, 0.15, 0.20],
    "subsample"      : [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    "gamma"          : [0, 0.1, 0.2, 0.5],
    "min_child_weight": [1, 3, 5],
}
search = RandomizedSearchCV(
    xgb.XGBClassifier(random_state=42, eval_metric="logloss"),
    param_distributions=param_dist,
    n_iter=50,
    scoring="roc_auc",
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    random_state=42,
    n_jobs=-1,
    verbose=0,
)
search.fit(X_train, y_train)
print(f"  Best CV ROC-AUC : {search.best_score_:.4f}")
print(f"  Best params     : {search.best_params_}\n")

# ── Final evaluation on held-out test set ───────────────────────────────────
model = search.best_estimator_
probs = model.predict_proba(X_test)[:, 1]
preds = (probs >= 0.50).astype(int)

roc_auc  = roc_auc_score(y_test, probs)
pr_auc   = average_precision_score(y_test, probs)
f1       = f1_score(y_test, preds)
precision = precision_score(y_test, preds)
recall   = recall_score(y_test, preds)

print("── Test-set results ────────────────────────────────────────────────")
print(f"  ROC-AUC   : {roc_auc:.4f}")
print(f"  PR-AUC    : {pr_auc:.4f}")
print(f"  F1        : {f1:.4f}")
print(f"  Precision : {precision:.4f}")
print(f"  Recall    : {recall:.4f}")
print()
print(classification_report(y_test, preds, target_names=["not co-purchased", "co-purchased"]))

# ── Plots ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("Co-Purchase Prediction — Model Evaluation", fontsize=14, fontweight="bold")

# 1. Confusion matrix
cm = confusion_matrix(y_test, preds)
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=["Not Co-Purchased", "Co-Purchased"],
    yticklabels=["Not Co-Purchased", "Co-Purchased"],
    ax=axes[0, 0],
)
axes[0, 0].set_title("Confusion Matrix")
axes[0, 0].set_xlabel("Predicted")
axes[0, 0].set_ylabel("Actual")

# 2. ROC curve
RocCurveDisplay.from_predictions(y_test, probs, ax=axes[0, 1], name=f"XGBoost (AUC={roc_auc:.3f})")
axes[0, 1].set_title("ROC Curve")
axes[0, 1].plot([0, 1], [0, 1], "k--", linewidth=0.8)

# 3. Precision-Recall curve
PrecisionRecallDisplay.from_predictions(y_test, probs, ax=axes[1, 0], name=f"XGBoost (AP={pr_auc:.3f})")
axes[1, 0].set_title("Precision-Recall Curve")

# 4. Feature importance
importance = pd.Series(model.feature_importances_, index=X.columns).sort_values()
importance.plot(kind="barh", ax=axes[1, 1], color="steelblue")
axes[1, 1].set_title("Feature Importances (gain)")
axes[1, 1].set_xlabel("Importance")

plt.tight_layout()
plot_path = MODELS / "evaluation_plots.png"
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n✓ Plots saved to {plot_path}")

# CV score distribution
cv_df = pd.DataFrame({
    "ROC-AUC": cv_results["test_roc_auc"],
    "PR-AUC" : cv_results["test_average_precision"],
})
fig2, ax = plt.subplots(figsize=(6, 4))
cv_df.plot(kind="box", ax=ax, color="steelblue")
ax.set_title("5-Fold CV Score Distribution")
ax.set_ylabel("Score")
cv_plot_path = MODELS / "cv_scores.png"
plt.tight_layout()
plt.savefig(cv_plot_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"✓ CV plot saved to {cv_plot_path}")

# ── Persist model ────────────────────────────────────────────────────────────
model_path = MODELS / "co_purchase_model.json"
model.save_model(model_path)
print(f"✓ Model saved to {model_path}")
