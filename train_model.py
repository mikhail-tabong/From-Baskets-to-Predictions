import pandas as pd, xgboost as xgb
from sklearn.metrics import roc_auc_score, precision_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load data
X = pd.read_parquet("X.parquet")
y = pd.read_csv("y.csv").squeeze()

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

# Initialize and train model
model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.10,
    subsample=0.9,
    colsample_bytree=0.8,
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate
probs = model.predict_proba(X_test)[:, 1]
preds = (probs >= 0.50).astype(int)

roc = roc_auc_score(y_test, probs)
precision = precision_score(y_test, preds)

print("ROC-AUC:", round(roc, 3))
print("Precision:", round(precision, 3))

# Optional: Plot feature importances
xgb.plot_importance(model, max_num_features=10)
plt.tight_layout()
plt.show()