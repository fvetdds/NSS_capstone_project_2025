import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import (classification_report,roc_auc_score,confusion_matrix,matthews_corrcoef,make_scorer,precision_score,precision_recall_curve,average_precision_score
)
import xgboost as xgb
import joblib

# Load and split data 
df = pd.read_csv("bcsc_concatenated_no_9.csv")
X = df.drop(columns="breast_cancer_history")
y = df["breast_cancer_history"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

# Compute class weights
scale_pos_weight = len(y_train[y_train==0])/len(y_train[y_train==1])
pos_penalty_factor = 10.0
weight_neg = 1.0
weight_pos = scale_pos_weight*pos_penalty_factor
def weighted_logloss(y_true: np.ndarray, y_pred: np.ndarray):
    p = 1.0 / (1.0 + np.exp(-y_pred))
    
    w = np.where(y_true == 1, weight_pos, weight_neg)
    
    grad = w * (p - y_true)
    hess = w * p * (1.0 - p)
    return grad, hess

# Hyperparameter search setup
penalized_clf = xgb.XGBClassifier(
    objective=weighted_logloss,
    eval_metric="auc",
    random_state=42
)

param_dist = {
    "n_estimators":     [100, 300, 500],
    "max_depth":        [3, 5, 7],
    "learning_rate":    [0.01, 0.05, 0.1],
    "subsample":        [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "gamma":            [0, 1, 5],
}

search = RandomizedSearchCV(
    estimator=penalized_clf,
    param_distributions=param_dist,
    n_iter=20,
    scoring=make_scorer(precision_score, pos_label=1),
    cv=5,
    verbose=2,
    random_state=42,
    n_jobs=-1
)
# Run hyperparameter search ---
search.fit(X_train, y_train)
print("Best hyperparameters:", search.best_params_)

# Retrain on full training set ---
best = search.best_estimator_
best.fit(X_train, y_train)

# Compute test‐set probabilities ---
y_prob = best.predict_proba(X_test)[:, 1]

# Precision‐Recall analysis ---
precision, recall, pr_thresholds = precision_recall_curve(y_test, y_prob)
avg_prec = average_precision_score(y_test, y_prob)

# pick threshold at precision ≥ 0.7
target_precision = 0.7
valid = np.where(precision >= target_precision)[0]
if len(valid) > 0:
    best_idx = valid[np.argmax(recall[valid])]
    matched_precision = precision[best_idx]
    matched_recall = recall[best_idx]
    matched_threshold = pr_thresholds[
        best_idx if best_idx < len(pr_thresholds) else -1
    ]
    print(f"At precision ≥ {target_precision:.2f}:")
    print(f"  Precision: {matched_precision:.3f}")
    print(f"  Recall:    {matched_recall:.3f}")
    print(f"  Threshold: {matched_threshold:.3f}")
else:
    print(f"No recall point found where precision ≥ {target_precision:.2f}")
    # plot Precision-Recall curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, marker='.', label=f'XGBoost (AP={avg_prec:.2f})')
plt.scatter(
    [matched_recall],
    [matched_precision],
    color='red',
    label=(
        f'Precision={matched_precision:.2f}\n'
        f'Recall={matched_recall:.2f}\n'
        f'Threshold={matched_threshold:.2f}'
    )
)
plt.xlabel('Recall', fontsize=14)
plt.ylabel('Precision', fontsize=14)
plt.title('Precision-Recall Curve', fontsize=16)
plt.grid(True)
plt.legend()
plt.show()

# --- Final evaluation at chosen threshold ---
best_thresh = matched_threshold
y_pred = (y_prob >= best_thresh).astype(int)

print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred, labels=[0,1])
print("Confusion Matrix:\n", cm)
print("\nTest ROC AUC:", roc_auc_score(y_test, y_prob))
print("Matthews Correlation Coefficient:", matthews_corrcoef(y_test, y_pred))
# --- Feature importances ---
feat_imp_df = pd.DataFrame({
    'feature': X_train.columns,
    'importance': best.feature_importances_
}).sort_values('importance', ascending=False)
print(feat_imp_df)

# Save model & threshold 
os.makedirs("models", exist_ok=True)
joblib.dump(best, os.path.join("models", "bcsc_xgb_model.pkl"))
joblib.dump(best_thresh, os.path.join("models", "threshold.pkl"))
print("Model and threshold saved.")
