import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
import xgboost as xgb
import pickle
import warnings
warnings.filterwarnings('ignore')

print("Step 1: Loading Phase 1 features...")
df = pd.read_csv('biogrid_features_phase1.csv')
print(f"  Loaded {len(df)} rows x {len(df.columns)} columns")

print("\nStep 2: Creating positive and negative examples...")

feature_cols = [col for col in df.columns if any([
    col.startswith('len_'),
    col.startswith('comp_'),
    col.startswith('mw_'),
    col.startswith('arom_'),
    col.startswith('inst_'),
    col.startswith('gravy_'),
    col.startswith('pi_'),
    col.startswith('diff_'),
    col.startswith('missing_'),
    col.startswith('invalid_')
])]

print(f"  Using {len(feature_cols)} features")

# All current rows are positive (confirmed interactions)
positive_df = df.copy()
positive_df['label'] = 1

print(f"  Positive examples (confirmed interactions): {len(positive_df)}")

# Create negative examples by shuffling protein pairs
# Negative pairs = random combinations that are unlikely to interact
np.random.seed(42)
negative_df = positive_df.copy()

# Shuffle within each feature group to break the interaction pattern
for col in feature_cols:
    if col.startswith('comp_a_'):
        # Shuffle composition features
        neg_col = col.replace('_a_', '_b_')
        if neg_col in feature_cols:
            indices = np.random.permutation(len(negative_df))
            negative_df[col] = negative_df[col].values[indices]
    elif col.startswith('len_'):
        if col == 'len_a':
            indices = np.random.permutation(len(negative_df))
            negative_df[col] = negative_df[col].values[indices]

negative_df['label'] = 0

print(f"  Negative examples (shuffled pairs): {len(negative_df)}")

# Combine positive and negative
df_balanced = pd.concat([positive_df, negative_df], ignore_index=True)
print(f"  Total samples: {len(df_balanced)}")
print(f"  Class balance: {(df_balanced['label'] == 1).sum()} positive, {(df_balanced['label'] == 0).sum()} negative")

X = df_balanced[feature_cols].copy()
y = df_balanced['label'].copy()

print(f"\nStep 3: Preparing training data...")
print(f"  Shape: {X.shape}")
print(f"  Missing values: {X.isna().sum().sum()}")

X = X.fillna(X.mean())

print(f"  Class balance: {y.sum()} positive, {(1-y).sum()} negative")
print(f"  Positive rate: {y.mean():.2%}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  Train: {len(X_train)} rows, Test: {len(X_test)} rows")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nStep 4: Training XGBoost...")
xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss'
)

xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)

y_pred_xgb = xgb_model.predict(X_test)
y_pred_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]

print("  XGBoost training complete")

print("\nStep 5: Training Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)
y_pred_proba_rf = rf_model.predict_proba(X_test)[:, 1]

print("  Random Forest training complete")

print("\nStep 6: Evaluating models...")

def evaluate_model(y_true, y_pred, y_pred_proba, model_name):
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    f1 = f1_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"\n{model_name}:")
    print(f"  ROC-AUC:  {roc_auc:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:   {recall:.4f}")
    print(f"  Confusion Matrix:")
    print(f"    TN={cm[0,0]}, FP={cm[0,1]}")
    print(f"    FN={cm[1,0]}, TP={cm[1,1]}")
    
    return {
        'model': model_name,
        'roc_auc': roc_auc,
        'f1': f1,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'tn': cm[0,0],
        'fp': cm[0,1],
        'fn': cm[1,0],
        'tp': cm[1,1]
    }

results_xgb = evaluate_model(y_test, y_pred_xgb, y_pred_proba_xgb, 'XGBoost')
results_rf = evaluate_model(y_test, y_pred_rf, y_pred_proba_rf, 'Random Forest')

print("\nStep 7: Feature importance...")

xgb_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 XGBoost features:")
print(xgb_importance.head(10).to_string(index=False))

rf_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Random Forest features:")
print(rf_importance.head(10).to_string(index=False))

print("\nStep 8: Saving models and results...")

with open('model_xgboost.pkl', 'wb') as f:
    pickle.dump(xgb_model, f)
print("  Saved model_xgboost.pkl")

with open('model_rf.pkl', 'wb') as f:
    pickle.dump(rf_model, f)
print("  Saved model_rf.pkl")

importance_df = pd.DataFrame({
    'feature': feature_cols,
    'xgboost_importance': xgb_model.feature_importances_,
    'rf_importance': rf_model.feature_importances_
}).sort_values('xgboost_importance', ascending=False)

importance_df.to_csv('feature_importance.csv', index=False)
print("  Saved feature_importance.csv")

with open('phase2_results.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("PHASE 2: BASELINE MODEL EVALUATION\n")
    f.write("="*80 + "\n\n")
    
    f.write("DATA SUMMARY:\n")
    f.write(f"  Total samples: {len(df_balanced)}\n")
    f.write(f"  - Positive (confirmed interactions): {(y == 1).sum()}\n")
    f.write(f"  - Negative (shuffled decoys): {(y == 0).sum()}\n")
    f.write(f"  Training set: {len(X_train)} rows\n")
    f.write(f"  Test set: {len(X_test)} rows\n")
    f.write(f"  Features used: {len(feature_cols)}\n")
    f.write(f"  Class balance (positive): {y.mean():.2%}\n\n")
    
    f.write("XGBOOST RESULTS:\n")
    f.write(f"  ROC-AUC:  {results_xgb['roc_auc']:.4f}\n")
    f.write(f"  F1 Score: {results_xgb['f1']:.4f}\n")
    f.write(f"  Accuracy: {results_xgb['accuracy']:.4f}\n")
    f.write(f"  Precision: {results_xgb['precision']:.4f}\n")
    f.write(f"  Recall:   {results_xgb['recall']:.4f}\n\n")
    
    f.write("RANDOM FOREST RESULTS:\n")
    f.write(f"  ROC-AUC:  {results_rf['roc_auc']:.4f}\n")
    f.write(f"  F1 Score: {results_rf['f1']:.4f}\n")
    f.write(f"  Accuracy: {results_rf['accuracy']:.4f}\n")
    f.write(f"  Precision: {results_rf['precision']:.4f}\n")
    f.write(f"  Recall:   {results_rf['recall']:.4f}\n\n")
    
    f.write("TOP 10 FEATURES (XGBoost):\n")
    for _, row in xgb_importance.head(10).iterrows():
        f.write(f"  {row['feature']}: {row['importance']:.4f}\n")
    f.write("\n")
    
    f.write("TOP 10 FEATURES (Random Forest):\n")
    for _, row in rf_importance.head(10).iterrows():
        f.write(f"  {row['feature']}: {row['importance']:.4f}\n")

print("  Saved phase2_results.txt")

print("\n" + "="*80)
print("Phase 2 Complete!")
print("="*80)
print("\nNOTE: Negative examples were created by shuffling protein pairs.")
print("This is a standard technique for generating decoy (non-interacting) examples.")
print("\nNext Steps:")
print("1. Review phase2_results.txt for model performance")
print("2. Check feature_importance.csv to understand which features matter most")
print("3. If ROC-AUC > 0.70, proceed to Phase 3 (advanced models)")
print("4. If ROC-AUC < 0.70, iterate on Phase 1 features or add Phase 2 features")
print("="*80)
