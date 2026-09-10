import os
import io
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, precision_score
import xgboost as xgb
DATASET_CSV_PATH = r"C:\Users\HARSH\Downloads\SIH 2026\Gemini code\knee_clinical_ml_ready.csv"
MODEL_SAVE_PATH  = r"C:\Users\HARSH\Downloads\SIH 2026\Gemini code\knee_oa_xgboost.json"
FEATURE_COLS = [
    "age",
    "bmi",
    "joint_site",
    "morning_stiffness",
    "pain_score",
    "mechanical_pain_relieved_by_rest",
    "grinding_crepitus",
    "squat_difficulty",
    "stairs_difficulty",
    "sit_to_stand_difficulty"
]
def load_and_preprocess_dataset(csv_path=DATASET_CSV_PATH):
    print("="*65)
    print("  SIH 2026 - KNEE OSTEOARTHRITIS (OA) ML TRAINING PIPELINE")
    print("="*65)
    print(f"[1/4] Loading patient records from: {csv_path}")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset file not found at: {csv_path}")
    df = pd.read_csv(csv_path)
    
    df.columns = [c.strip().lower() for c in df.columns]
   
    df = df[pd.to_numeric(df['patient_id'], errors='coerce').notnull()].copy()
    
    numeric_cols = [
        "age", "bmi", "joint_site", "morning_stiffness", 
        "pain_score", "mechanical_pain_relieved_by_rest", 
        "grinding_crepitus", "squat_difficulty", 
        "stairs_difficulty", "sit_to_stand_difficulty"
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df["age"] = df["age"].fillna(df["age"].median())
    df["bmi"] = df["bmi"].fillna(df["bmi"].median())
    df["joint_site"] = df["joint_site"].fillna(0).astype(int)
    df["morning_stiffness"] = df["morning_stiffness"].fillna(0).astype(int)
    df["pain_score"] = df["pain_score"].fillna(0)
    df["mechanical_pain_relieved_by_rest"] = df["mechanical_pain_relieved_by_rest"].fillna(1).astype(int)
    df["grinding_crepitus"] = df["grinding_crepitus"].fillna(0).astype(int)
    df["squat_difficulty"] = df["squat_difficulty"].fillna(df["pain_score"] * 0.8)
    df["stairs_difficulty"] = df["stairs_difficulty"].fillna(df["pain_score"] * 0.5)
    df["sit_to_stand_difficulty"] = df["sit_to_stand_difficulty"].fillna(df["pain_score"] * 0.5)
    clinical_oa_mask = (
        (df["pain_score"] >= 3.0) & 
        (
            (df["morning_stiffness"] == 1) | 
            (df["grinding_crepitus"] == 1) | 
            ((df["stairs_difficulty"] + df["sit_to_stand_difficulty"]) >= 2.5)
        )
    )
    df["target_oa"] = clinical_oa_mask.astype(int)
    total_records = len(df)
    oa_pos = int(df["target_oa"].sum())
    oa_neg = total_records - oa_pos
    print(f"[DATASET SUMMARY]")
    print(f"  • Total Evaluated Knees : {total_records}")
    print(f"  • Right Knees (0)       : {int((df['joint_site'] == 0).sum())}")
    print(f"  • Left Knees (1)        : {int((df['joint_site'] == 1).sum())}")
    print(f"  • Arthritic Knees (1)   : {oa_pos} ({oa_pos/total_records*100:.1f}%)")
    print(f"  • Healthy / Non-OA (0)  : {oa_neg} ({oa_neg/total_records*100:.1f}%)")
    return df
#training model
def train_xgboost_classifier(df):
    print("\n[2/4] Training XGBoost with 5-Fold Stratified Cross-Validation...")
    X = df[FEATURE_COLS]
    y = df["target_oa"]
    pos_count = int(np.sum(y == 1))
    neg_count = int(np.sum(y == 0))
    scale_weight = float(neg_count) / float(pos_count) if pos_count > 0 else 1.0
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_list = []
    sens_list = []
    spec_list = []
    acc_list = []
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=3,           # Shallow depth prevents overfitting
            learning_rate=0.05,
            scale_pos_weight=scale_weight,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.5,
            eval_metric="logloss",
            random_state=42
        )
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_val)[:, 1]
        preds = (probs >= 0.5).astype(int)
        auc_list.append(roc_auc_score(y_val, probs))
        sens_list.append(recall_score(y_val, preds))
        acc_list.append(accuracy_score(y_val, preds))
        # Specificity
        neg_mask = (y_val == 0)
        spec = np.sum((preds == 0) & neg_mask) / np.sum(neg_mask) if np.sum(neg_mask) > 0 else 0
        spec_list.append(spec)
    print("\n" + "="*55)
    print("5-FOLD CROSS-VALIDATION CLINICAL PERFORMANCE")
    print("="*55)
    print(f"  • Mean ROC-AUC Score   : {np.mean(auc_list):.3f} (± {np.std(auc_list):.3f})")
    print(f"  • Clinical Sensitivity : {np.mean(sens_list)*100:.1f}% (Actual OA detected)")
    print(f"  • Clinical Specificity : {np.mean(spec_list)*100:.1f}% (Healthy correctly cleared)")
    print(f"  • Overall Accuracy     : {np.mean(acc_list)*100:.1f}%")
    print("="*55)
    # Train final model on 100% of data
    print("\n[3/4] Fitting final production model and serializing...")
    final_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        scale_pos_weight=scale_weight,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.5,
        eval_metric="logloss",
        random_state=42
    )
    final_model.fit(X, y)
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    final_model.save_model(MODEL_SAVE_PATH)
    print(f"[SUCCESS] Trained model saved to: {MODEL_SAVE_PATH}")
    # Display Top Feature Importance
    importances = final_model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    print("\nTOP CLINICAL RISK PREDICTORS (Feature Importances):")
    for rank, idx in enumerate(sorted_idx, 1):
        print(f"  {rank:2d}. {FEATURE_COLS[idx]:<32}: {importances[idx]*100:5.1f}%")
    return final_model
def predict_knee_oa_probability(patient_data, model_path=MODEL_SAVE_PATH):
    """
    Takes patient parameters and returns:
      - Probability of Knee OA in percentage (0.0% to 100.0%)
      - Triage Risk Tier
      - Clinical Recommendation
    """
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    # Align input to model features
    row = [patient_data.get(feat, 0) for feat in FEATURE_COLS]
    input_df = pd.DataFrame([row], columns=FEATURE_COLS)
    prob = model.predict_proba(input_df)[0][1]
    percentage = round(prob * 100.0, 1)
    # Joint Site label
    side = "Right Knee" if patient_data.get("joint_site", 0) == 0 else "Left Knee"
    # Triage Stratification
    if percentage >= 70.0:
        tier = "HIGH RISK (Severe / Advanced Osteoarthritis)"
        action = "Priority clinical referral: Order standing weight-bearing radiographs (X-ray) and orthopedic review."
    elif percentage >= 35.0:
        tier = "MODERATE RISK (Early / Developing Osteoarthritis)"
        action = "Prescribe quadriceps strengthening physical therapy, joint unloading exercises, and dietary management."
    else:
        tier = "LOW RISK (Healthy / Unlikely Osteoarthritis)"
        action = "Maintain regular physical activity, joint mobility, and ergonomic posture guidance."
    return {
        "joint_site": side,
        "probability_value": prob,
        "oa_probability_percentage": f"{percentage}%",
        "triage_category": tier,
        "recommended_action": action
    }
if __name__ == "__main__":
    # 1. Load your dataset and train the model
    dataset = load_and_preprocess_dataset()
    trained_model = train_xgboost_classifier(dataset)
    # 2. Test prediction on a sample patient
    test_patient = {
        "age": 19,
        "bmi": 22.5,
        "joint_site": 0,                   # Right Knee
        "morning_stiffness": 0,
        "pain_score": 5,                   # 7/10 pain
        "mechanical_pain_relieved_by_rest": 1,
        "grinding_crepitus": 0,
        "squat_difficulty": 0,
        "stairs_difficulty": 0,
        "sit_to_stand_difficulty": 0
    }
    print("\n" + "="*55)
    print("TESTING PREDICTION FOR A SAMPLE PATIENT:")
    print("="*55)
    result = predict_knee_oa_probability(test_patient)
    print(f"  • Tested Joint         : {result['joint_site']}")
    print(f"  • Probability of OA    : {result['oa_probability_percentage']}")
    print(f"  • Triage Category      : {result['triage_category']}")
    print(f"  • Clinical Guidance    : {result['recommended_action']}")
    print("="*55)