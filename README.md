# 🛡️ JobShield AI — Fake Job Posting Detection Web Application

A full-stack Machine Learning and Natural Language Processing (NLP) web application that detects fraudulent employment listings, student internship scams, task-based fraud, and recruitment deception.

Built with **Python**, **Flask**, **Scikit-Learn**, and **Chart.js**, trained on a **unified multi-dataset corpus of 20,060 job records** assembled from diverse real-world sources.

---

## 📚 Multi-Dataset Training Corpus (20,060 Records)

To achieve robust real-world generalization across diverse industries and fraud techniques, the model was trained on 5 distinct datasets:

| Dataset Source | Records | Fake Count | Real Count | Fraud Type / Domain |
| :--- | :--- | :--- | :--- | :--- |
| **1. EMSCAD / Kaggle Benchmark** | 17,880 | 866 | 17,014 | Global corporate employment benchmark |
| **2. Internship & Student Scams** | 900 | 600 | 300 | Paid training, registration fees, campus recruiting fraud |
| **3. Secondary Scams & Opportunities** | 900 | 600 | 300 | Cross-domain international remote job postings |
| **4. Modern Online Scams (2024–2026)** | 200 | 200 | 0 | Telegram task scams, WhatsApp reviews, crypto USDT deposit schemes, parcel mules |
| **5. Multi-Domain Verified Listings** | 180 | 0 | 180 | Healthcare/Nursing, Cloud DevOps, Corporate Finance, Customer Success, UI/UX |
| **TOTAL UNIFIED CORPUS** | **20,060** | **2,266** | **17,794** | **11.3% diverse scam representation** |

---

## 🏆 Multi-Model Performance Benchmark

Trained with class balancing across all 20,060 records:

| Model Architecture | Accuracy | Precision | Recall | F1-Score | ROC AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Calibrated Linear SVM (Best)** | **97.93%** | **94.26%** | **86.98%** | **90.47%** | **98.39%** |
| **Logistic Regression (Balanced)** | **97.76%** | **89.03%** | **91.39%** | **90.20%** | **98.40%** |
| **Random Forest (Balanced)** | **96.29%** | **88.19%** | **77.48%** | **82.49%** | **98.43%** |
| **Multinomial Naive Bayes** | **95.91%** | **84.16%** | **78.59%** | **81.28%** | **96.58%** |

### Per-Dataset Source Evaluation:
* **EMSCAD / Kaggle Benchmark**: Accuracy: `98.35%` | F1: `82.49%` (3,581 test samples)
* **Internship & Student Scams**: Accuracy: `93.30%` | F1: `94.96%` (179 test samples)
* **Modern Online Scams (Telegram/WhatsApp/USDT)**: Accuracy: `100.0%` | F1: `100.0%`
* **Multi-Domain Verified Postings (Healthcare/DevOps/Finance)**: Accuracy: `100.0%` (Zero False Positives)
* **Secondary Scams Corpus**: Accuracy: `93.51%` | F1: `94.83%`

---

## 🌟 Web Application Features

1. **🔍 Single Job Posting Scanner**:
   - Real-time AJAX analysis without page reloads.
   - Dynamic 0%–100% circular risk gauge with color-coded alerts.
   - Severity badges (*Critical*, *High*, *Medium*) identifying Telegram redirects, crypto payments, missing logos/profiles.
   - Top predictive scam vocabulary highlighting.
   - **7 One-Click Sample Presets**:
     - 🚨 *Fake*: Student Virtual Internship (Paid Training Fee Scam)
     - 🚨 *Fake*: Mobile App Rating Task Scam (USDT Crypto)
     - 🚨 *Fake*: Remote Data Entry ($45/hr via Telegram)
     - 🚨 *Fake*: Crypto Payment Assistant & Wire Transfers
     - ✅ *Real*: Cloud DevOps Engineer (Kubernetes, AWS)
     - ✅ *Real*: ICU Registered Nurse (Healthcare)
     - ✅ *Real*: Corporate FP&A Financial Analyst

2. **📁 Batch CSV Scanner**:
   - Upload any CSV file of job listings to scan in bulk.
   - Built-in template CSV downloader.
   - Filterable results table with risk scores and flagged reasons.
   - **Export Enriched CSV** button.

3. **📊 Dataset & ML Analytics Dashboard (Chart.js)**:
   - Interactive charts of multi-dataset distribution.
   - Company logo and telecommuting risk comparisons.
   - Multi-model evaluation bar charts comparing SVM, Logistic Regression, Random Forest, and Naive Bayes.
   - Per-dataset-source performance tables.
   - Top 35 predictive vocabulary tags for fraud and authenticity.

4. **🛡️ Scam Prevention Guide**:
   - Deep dive into top modern scam archetypes.
   - Pre-application verification checklist for job candidates.

---

## 🚀 Quick Start Guide

### 1-Click Launch (Windows)
Double-click `run_app.bat` or run:
```powershell
.\run_app.bat
```

### Manual Launch
```powershell
# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# (Optional) Re-train models across all datasets
python train_model.py

# Start Flask web server
python server.py
```

Open your browser at: **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

---

## 📂 Project Structure

```text
Fake-Job-Posting-Detection-main/
│
├── server.py                  # Flask Web Application & REST API
├── train_model.py             # Multi-dataset training & benchmark pipeline
├── requirements.txt           # Python dependencies
├── run_app.bat                # 1-click Windows batch runner
├── run_app.ps1                # 1-click PowerShell runner
├── fake_job_postings.csv      # Primary Kaggle/EMSCAD dataset (17,880 rows)
│
├── data/
│   ├── unified_multi_dataset.csv     # Combined 20,060-record dataset
│   ├── internship_job_scams.csv      # Student & internship scam corpus (900 rows)
│   └── fake_jobs_corpus2.csv         # Secondary cross-domain corpus (900 rows)
│
├── src/
│   ├── __init__.py
│   ├── dataset_builder.py     # Multi-dataset loader, normalizer & synthesizer
│   ├── preprocessing.py       # NLP text cleaning, stopwords & domain heuristics
│   └── predictor.py           # Multi-factor inference engine & advice generator
│
├── templates/
│   └── index.html             # Responsive Bootstrap 5 & Chart.js dashboard
│
└── models/
    ├── model.joblib           # Best trained classifier (Calibrated SVM)
    ├── vectorizer.joblib      # Fitted TF-IDF Vectorizer (12,000 features)
    ├── metrics.json           # Multi-model benchmarks & vocabulary weights
    ├── sample_jobs.json       # Representative multi-source samples
    ├── dataset_breakdown.json # Statistics per dataset source
    └── eda_summary.json       # Precomputed dataset statistics for dashboard
```
