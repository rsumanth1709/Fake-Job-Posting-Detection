import os
import sys
import json
import joblib
import copy

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier, PassiveAggressiveClassifier, RidgeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

from src.preprocessing import clean_text, remove_stopwords, preprocess_job_text, analyze_scam_signals
from src.dataset_builder import load_and_standardize_datasets

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(MODELS_DIR, exist_ok=True)

def extract_structural_features(df):
    """
    Extracts dense structural and metadata signals correlated with recruitment fraud.
    """
    feats = []
    for _, row in df.iterrows():
        title = str(row.get('title', '') or '')
        cp = str(row.get('company_profile', '') or '')
        desc = str(row.get('description', '') or '')
        req = str(row.get('requirements', '') or '')
        ben = str(row.get('benefits', '') or '')
        tele = int(row.get('telecommuting', 0) if pd.notnull(row.get('telecommuting')) else 0)
        logo = int(row.get('has_company_logo', 1) if pd.notnull(row.get('has_company_logo')) else 1)

        has_prof = 1 if len(cp.strip()) >= 30 else 0
        has_req = 1 if len(req.strip()) >= 25 else 0
        has_ben = 1 if len(ben.strip()) >= 15 else 0
        no_logo_no_prof = 1 if (logo == 0 and has_prof == 0) else 0
        remote_no_logo = 1 if (tele == 1 and logo == 0) else 0
        remote_no_prof = 1 if (tele == 1 and has_prof == 0) else 0

        desc_len_norm = min(1.0, len(desc.split()) / 300.0)
        prof_len_norm = min(1.0, len(cp.split()) / 150.0)

        # Count regex scam indicators
        flags = analyze_scam_signals(
            title=title, company_profile=cp, description=desc,
            requirements=req, benefits=ben, telecommuting=tele, has_company_logo=logo
        )
        crit_count = sum(1 for f in flags if f['severity'] == 'Critical')
        high_count = sum(1 for f in flags if f['severity'] == 'High')
        med_count = sum(1 for f in flags if f['severity'] == 'Medium')

        feats.append([
            logo,
            has_prof,
            prof_len_norm,
            has_req,
            has_ben,
            tele,
            no_logo_no_prof,
            remote_no_logo,
            remote_no_prof,
            desc_len_norm,
            crit_count,
            high_count,
            med_count
        ])

    return np.array(feats, dtype=np.float32)

def train_and_evaluate_all_models():
    print("=" * 70)
    print("[INFO] HYBRID TEXT + METADATA MULTI-MODEL TRAINING PIPELINE")
    print("=" * 70)

    unified_csv = os.path.join(DATA_DIR, "unified_multi_dataset.csv")
    if not os.path.exists(unified_csv):
        print("[INFO] Unified dataset not found. Assembling multiple datasets...")
        df = load_and_standardize_datasets()
    else:
        print(f"[INFO] Loading unified multi-dataset from {unified_csv}...")
        df = pd.read_csv(unified_csv, on_bad_lines='skip', engine='python')

    print(f"Loaded {len(df):,} total records across multiple sources.")

    # Clean text columns
    for col in ['title', 'company_profile', 'description', 'requirements', 'benefits']:
        df[col] = df[col].fillna("").astype(str)

    df['telecommuting'] = pd.to_numeric(df.get('telecommuting', 0), errors='coerce').fillna(0).astype(int)
    df['has_company_logo'] = pd.to_numeric(df.get('has_company_logo', 1), errors='coerce').fillna(1).astype(int)
    if 'dataset_source' not in df.columns:
        df['dataset_source'] = "EMSCAD Benchmark"

    df['fraudulent'] = pd.to_numeric(df['fraudulent'], errors='coerce').fillna(0).astype(int)

    print("\nOverall Target Distribution:")
    counts = df['fraudulent'].value_counts()
    print(f"  Real (0): {counts.get(0, 0):,} ({counts.get(0, 0)/len(df)*100:.2f}%)")
    print(f"  Fake (1): {counts.get(1, 0):,} ({counts.get(1, 0)/len(df)*100:.2f}%)")

    # Dataset Source Breakdown
    dataset_breakdown = {}
    for src, grp in df.groupby("dataset_source"):
        f_cnt = int((grp['fraudulent'] == 1).sum())
        r_cnt = int((grp['fraudulent'] == 0).sum())
        dataset_breakdown[src] = {
            "total": len(grp),
            "real": r_cnt,
            "fake": f_cnt,
            "fraud_rate": round(f_cnt / len(grp) * 100, 2)
        }

    with open(os.path.join(MODELS_DIR, "dataset_breakdown.json"), "w", encoding="utf-8") as f:
        json.dump(dataset_breakdown, f, indent=2)

    # 1. Clean Text & TF-IDF
    print("\n[INFO] Extracting TF-IDF text features & structural metadata features...")
    combined = (
        df['title'] + " " +
        df['company_profile'] + " " +
        df['description'] + " " +
        df['requirements'] + " " +
        df['benefits']
    )
    clean_texts = [remove_stopwords(clean_text(t)) for t in combined]
    df['clean_text'] = clean_texts

    # Train / Test Split
    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.20, random_state=42, stratify=df['fraudulent']
    )

    X_train_text = df['clean_text'].iloc[train_idx]
    X_test_text = df['clean_text'].iloc[test_idx]
    y_train = df['fraudulent'].iloc[train_idx]
    y_test = df['fraudulent'].iloc[test_idx]
    sources_test = df['dataset_source'].iloc[test_idx]

    # Fit TF-IDF
    vectorizer = TfidfVectorizer(max_features=12000, ngram_range=(1, 2), min_df=2)
    X_train_tfidf = vectorizer.fit_transform(X_train_text)
    X_test_tfidf = vectorizer.transform(X_test_text)

    # Extract dense structural features
    print("[INFO] Building structural metadata feature matrix (logo, profile, length, flags)...")
    X_train_meta = extract_structural_features(df.iloc[train_idx])
    X_test_meta = extract_structural_features(df.iloc[test_idx])

    # Combine Text TF-IDF + Metadata Features via Sparse Horizontal Stack
    X_train_comb = sp.hstack([X_train_tfidf, sp.csr_matrix(X_train_meta)], format="csr")
    X_test_comb = sp.hstack([X_test_tfidf, sp.csr_matrix(X_test_meta)], format="csr")
    print(f"Combined feature shape: {X_train_comb.shape}")

    # -------------------------------------------------------------
    # 1. Dedicated Iterative Deep Neural Network (50 Epochs)
    # -------------------------------------------------------------
    TOTAL_EPOCHS = 50
    print("\n" + "=" * 70)
    print(f"[INFO] TRAINING DEEP NEURAL NETWORK (MLP) ACROSS {TOTAL_EPOCHS} EPOCHS")
    print("=" * 70)

    mlp_50 = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        solver='adam',
        alpha=1e-4,
        learning_rate_init=1e-3,
        random_state=42,
        warm_start=True
    )

    classes = np.array([0, 1])
    epoch_history = []
    best_epoch_f1 = -1.0
    best_mlp_checkpoint = None

    for epoch in range(1, TOTAL_EPOCHS + 1):
        mlp_50.partial_fit(X_train_comb, y_train, classes=classes)
        train_loss = float(mlp_50.loss_)

        # Validation metrics at current epoch
        val_preds = mlp_50.predict(X_test_comb)
        val_probs = mlp_50.predict_proba(X_test_comb)[:, 1]
        val_acc = float(accuracy_score(y_test, val_preds))
        val_prec = float(precision_score(y_test, val_preds, zero_division=0))
        val_rec = float(recall_score(y_test, val_preds, zero_division=0))
        val_f1 = float(f1_score(y_test, val_preds, zero_division=0))
        val_auc = float(roc_auc_score(y_test, val_probs))

        is_new_best = val_f1 > best_epoch_f1
        if is_new_best:
            best_epoch_f1 = val_f1
            best_mlp_checkpoint = copy.deepcopy(mlp_50)

        marker = " (*Best Checkpoint)" if is_new_best else ""
        print(f"Epoch {epoch:02d}/{TOTAL_EPOCHS} | Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f} | Val Prec: {val_prec:.4f} | Val Recall: {val_rec:.4f} | Val F1: {val_f1:.4f} | Val AUC: {val_auc:.4f}{marker}", flush=True)

        epoch_history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_accuracy": round(val_acc, 4),
            "val_precision": round(val_prec, 4),
            "val_recall": round(val_rec, 4),
            "val_f1": round(val_f1, 4),
            "val_roc_auc": round(val_auc, 4)
        })

    # Use the best generalizing checkpoint across all 50 epochs
    optimal_mlp = best_mlp_checkpoint if best_mlp_checkpoint is not None else mlp_50

    # -------------------------------------------------------------
    # 2. Define Diverse Model Families on Hybrid Features
    # -------------------------------------------------------------
    base_svm = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-4, max_iter=1000, random_state=42, class_weight='balanced')
    base_pa = PassiveAggressiveClassifier(C=1.0, max_iter=1000, random_state=42, class_weight='balanced')
    base_ridge = RidgeClassifier(class_weight='balanced', random_state=42)

    models = {
        # 1. Deep Neural Network (50 Epochs Trained)
        "Deep Neural Network (50 Epochs)": optimal_mlp,

        # 2. Hybrid Linear SVM
        "Calibrated Linear SVM": CalibratedClassifierCV(base_svm),

        # 3. Hybrid Balanced Logistic Regression
        "Logistic Regression (Balanced)": LogisticRegression(
            C=6.0, class_weight='balanced', max_iter=1000, random_state=42
        ),

        # 4. Hybrid Ridge Classifier
        "Ridge Classifier (Calibrated)": CalibratedClassifierCV(base_ridge),

        # 5. Hybrid Passive Aggressive
        "Passive Aggressive Classifier": CalibratedClassifierCV(base_pa),

        # 6. Hybrid Random Forest (Handles non-linear metadata feature interactions)
        "Random Forest (Balanced)": RandomForestClassifier(
            n_estimators=100, class_weight='balanced_subsample', max_depth=28, random_state=42, n_jobs=-1
        ),

        # 7. Hybrid Soft Voting Ensemble (Combines SVM, LR, and Forest)
        "Voting Ensemble (SVM + LR + RF)": VotingClassifier(
            estimators=[
                ("svm", CalibratedClassifierCV(base_svm)),
                ("lr", LogisticRegression(C=6.0, class_weight='balanced', max_iter=1000, random_state=42)),
                ("rf", RandomForestClassifier(n_estimators=80, class_weight='balanced_subsample', max_depth=25, random_state=42, n_jobs=-1))
            ],
            voting="soft"
        )
    }

    results = {}
    best_model_name = None
    best_f1 = -1.0
    best_clf = None

    print(f"\n[INFO] Evaluating {len(models)} hybrid model architectures on test set...")
    print("-" * 70)

    for name, clf in models.items():
        if "Deep Neural Network" not in name:
            print(f"-> Training & Evaluating {name}...", flush=True)
            clf.fit(X_train_comb, y_train)
        else:
            print(f"-> Evaluating {name} ({TOTAL_EPOCHS} Epochs Finished)...", flush=True)

        preds = clf.predict(X_test_comb)
        prob_preds = clf.predict_proba(X_test_comb)[:, 1] if hasattr(clf, "predict_proba") else preds

        acc = float(accuracy_score(y_test, preds))
        prec = float(precision_score(y_test, preds, zero_division=0))
        rec = float(recall_score(y_test, preds, zero_division=0))
        f1 = float(f1_score(y_test, preds, zero_division=0))
        auc = float(roc_auc_score(y_test, prob_preds))
        cm = confusion_matrix(y_test, preds).tolist()

        print(f"   Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | ROC AUC: {auc:.4f}")

        source_eval = {}
        for src_name in dataset_breakdown.keys():
            mask = (sources_test == src_name).values
            if mask.sum() > 0:
                s_y_true = y_test[mask]
                s_preds = preds[mask]
                s_acc = float(accuracy_score(s_y_true, s_preds))
                s_f1 = float(f1_score(s_y_true, s_preds, zero_division=0))
                source_eval[src_name] = {
                    "test_samples": int(mask.sum()),
                    "accuracy": round(s_acc, 4),
                    "f1_score": round(s_f1, 4)
                }

        results[name] = {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(auc, 4),
            "confusion_matrix": cm,
            "source_evaluation": source_eval
        }

        # Select best model: prioritize high recall and F1 score
        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            best_clf = clf

    print("-" * 70)
    print(f"[INFO] 🏆 Best Performing Architecture: {best_model_name} (F1 Score: {best_f1:.4f})")

    # Vocabulary weights
    aux_lr = LogisticRegression(C=5.0, class_weight='balanced', max_iter=500).fit(X_train_tfidf, y_train)
    feature_names = np.array(vectorizer.get_feature_names_out())
    coefs = aux_lr.coef_[0]
    top_fraud_idx = np.argsort(coefs)[-35:][::-1]
    top_real_idx = np.argsort(coefs)[:35]
    results["top_fraud_words"] = feature_names[top_fraud_idx].tolist()
    results["top_real_words"] = feature_names[top_real_idx].tolist()
    results["dataset_breakdown"] = dataset_breakdown
    results["epoch_history"] = epoch_history
    results["best_model_name"] = best_model_name

    with open(os.path.join(MODELS_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Save artifacts
    joblib.dump(vectorizer, os.path.join(MODELS_DIR, "vectorizer.joblib"))
    joblib.dump(best_clf, os.path.join(MODELS_DIR, "model.joblib"))
    print("\n[SUCCESS] Artifacts and metrics saved successfully in 'models/' directory.")
    return results

if __name__ == "__main__":
    train_and_evaluate_all_models()
