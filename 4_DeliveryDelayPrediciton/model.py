import numpy as np
import pandas as pd

from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
)

# =========================
# 1) 读取训练表
# 如果你已经有 model_ready 这个 DataFrame，
# 就把下面这行改成: df = model_ready.copy()
# =========================
df = pd.read_csv("./csv_files/olist_model_ready_late_delivery.csv")

# =========================
# 2) 目标变量
# =========================
target_col = "is_late"
df = df[df[target_col].notna()].copy()
df[target_col] = df[target_col].astype(int)

# =========================
# 3) 删除不该进模型的列
# =========================
drop_cols = [
    target_col,
    "order_id",                    # 纯ID
    "order_status",                # delivered 后通常变成常量
    "order_purchase_timestamp",    # 原始时间戳字符串，先不用
    "order_approved_at",
    "order_estimated_delivery_date",
    "customer_geo_city",           # 与 customer_city 高度重复，可先删掉
    "customer_geo_state",
]

X = df.drop(columns=[c for c in drop_cols if c in df.columns]).copy()
y = df[target_col].copy()

# 删除常量列
nunique = X.nunique(dropna=False)
constant_cols = nunique[nunique <= 1].index.tolist()
if constant_cols:
    X = X.drop(columns=constant_cols)

# =========================
# 4) 识别类别列 / 数值列
# =========================
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = [c for c in X.columns if c not in cat_cols]

# CatBoost 的分类列不要保留 NaN / float 表示，统一转字符串
for c in cat_cols:
    X[c] = X[c].astype("string").fillna("__MISSING__").astype(str)

# 数值列尽量转成 numeric
for c in num_cols:
    X[c] = pd.to_numeric(X[c], errors="coerce")

print("X shape:", X.shape)
print("类别列数量:", len(cat_cols))
print("数值列数量:", len(num_cols))
print("延误比例:", y.mean())

# =========================
# 5) 分层切分：70% train / 15% valid / 15% test
# =========================
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y,
    test_size=0.15,
    random_state=42,
    stratify=y
)

X_train, X_valid, y_train, y_valid = train_test_split(
    X_trainval, y_trainval,
    test_size=0.17647,   # 0.85 * 0.17647 ≈ 0.15
    random_state=42,
    stratify=y_trainval
)

print("Train:", X_train.shape, y_train.shape)
print("Valid:", X_valid.shape, y_valid.shape)
print("Test :", X_test.shape, y_test.shape)

# =========================
# 6) 训练 CatBoost
# =========================
model = CatBoostClassifier(
    iterations=2000,
    learning_rate=0.03,
    depth=8,
    loss_function="Logloss",
    eval_metric="AUC",
    auto_class_weights="Balanced",
    random_state=42,
    verbose=100
)

model.fit(
    X_train,
    y_train,
    cat_features=cat_cols,
    eval_set=(X_valid, y_valid),
    use_best_model=True,
    early_stopping_rounds=100
)

# =========================
# 7) 在验证集上调阈值（按 F1）
# =========================
valid_proba = model.predict_proba(X_valid)[:, 1]

best_threshold = 0.5
best_f1 = -1

for t in np.arange(0.10, 0.91, 0.01):
    pred_t = (valid_proba >= t).astype(int)
    f1_t = f1_score(y_valid, pred_t, zero_division=0)
    if f1_t > best_f1:
        best_f1 = f1_t
        best_threshold = float(t)

print(f"\nBest threshold on valid = {best_threshold:.2f}")
print(f"Best valid F1           = {best_f1:.4f}")

# =========================
# 8) 测试集评估
# =========================
test_proba = model.predict_proba(X_test)[:, 1]
test_pred = (test_proba >= best_threshold).astype(int)

print("\n=== Test Metrics ===")
print("Accuracy :", round(accuracy_score(y_test, test_pred), 4))
print("Precision:", round(precision_score(y_test, test_pred, zero_division=0), 4))
print("Recall   :", round(recall_score(y_test, test_pred, zero_division=0), 4))
print("F1       :", round(f1_score(y_test, test_pred, zero_division=0), 4))
print("ROC-AUC  :", round(roc_auc_score(y_test, test_proba), 4))
print("PR-AUC   :", round(average_precision_score(y_test, test_proba), 4))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, test_pred))

print("\nClassification Report:")
print(classification_report(y_test, test_pred, digits=4, zero_division=0))

# =========================
# 9) 特征重要性
# =========================
fi = pd.DataFrame({
    "feature": X_train.columns,
    "importance": model.get_feature_importance()
}).sort_values("importance", ascending=False)

print("\nTop 20 Feature Importance:")
print(fi.head(20))

# =========================
# 10) 保存结果
# =========================
fi.to_csv("catboost_feature_importance.csv", index=False, encoding="utf-8-sig")
model.save_model("catboost_late_delivery.cbm")

print("\n已保存:")
print("- catboost_feature_importance.csv")
print("- catboost_late_delivery.cbm")