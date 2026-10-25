import os
import json
import joblib
import numpy as np
import scipy.sparse as sp
from src.preprocessing import preprocess_job_text, analyze_scam_signals
from src.verifier import verify_company_domain, verify_recruiter_contact, run_full_verification

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

class JobFraudPredictor:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.metrics = {}
        self.sample_jobs = []
        self.eda_summary = {}
        self.load_artifacts()

    def load_artifacts(self):
        model_path = os.path.join(MODELS_DIR, "model.joblib")
        vec_path = os.path.join(MODELS_DIR, "vectorizer.joblib")
        metrics_path = os.path.join(MODELS_DIR, "metrics.json")
        samples_path = os.path.join(MODELS_DIR, "sample_jobs.json")
        eda_path = os.path.join(MODELS_DIR, "eda_summary.json")

        if os.path.exists(model_path) and os.path.exists(vec_path):
            try:
                self.model = joblib.load(model_path)
                self.vectorizer = joblib.load(vec_path)
            except Exception as e:
                print(f"Warning loading model artifacts: {e}")

        if os.path.exists(metrics_path):
            try:
                with open(metrics_path, "r", encoding="utf-8") as f:
                    self.metrics = json.load(f)
            except Exception:
                pass

        if os.path.exists(samples_path):
            try:
                with open(samples_path, "r", encoding="utf-8") as f:
                    self.sample_jobs = json.load(f)
            except Exception:
                pass

        if os.path.exists(eda_path):
            try:
                with open(eda_path, "r", encoding="utf-8") as f:
                    self.eda_summary = json.load(f)
            except Exception:
                pass

    def is_ready(self) -> bool:
        return self.model is not None and self.vectorizer is not None

    def predict(
        self,
        title: str = "",
        company_name: str = "",
        company_profile: str = "",
        description: str = "",
        requirements: str = "",
        benefits: str = "",
        telecommuting: int = 0,
        has_company_logo: int = 1,
        has_questions: int = 0,
        employment_type: str = "Full-time",
        required_experience: str = "Not Specified",
        contact_info: str = ""
    ):
        """
        Analyzes a single job posting using a hybrid multi-modal feature union:
        - TF-IDF NLP text representations
        - Dense structural metadata signals (logo, profile length, requirements, remote flag)
        - Live DNS & Corporate Domain Entity Verification
        - Recruiter Communication Channel & Phishing Auditing
        - Domain-knowledge heuristic guardrails
        """
        # 1. Linguistic & Structural Red Flags Analysis
        flags = analyze_scam_signals(
            title=title,
            company_name=company_name,
            company_profile=company_profile,
            description=description,
            requirements=requirements,
            benefits=benefits,
            telecommuting=telecommuting,
            has_company_logo=has_company_logo,
            has_questions=has_questions
        )

        # 1b. Live Entity & Corporate Domain Verification
        if company_name and len(company_name.strip()) >= 3:
            comp_valid, comp_domain, comp_details = verify_company_domain(company_name)
            if comp_valid is False:
                flags.append({
                    "severity": "High",
                    "factor": "Unverified / Non-Existent Corporate Domain",
                    "explanation": comp_details
                })

        # 1c. Recruiter Contact / Application Channel Verification
        if contact_info and len(contact_info.strip()) >= 3:
            contact_valid, contact_details = verify_recruiter_contact(company_name, contact_info)
            if contact_valid is False:
                flags.append({
                    "severity": "Critical",
                    "factor": "High-Risk Recruiter Communication Channel",
                    "explanation": contact_details
                })
            elif contact_valid == "Suspicious":
                flags.append({
                    "severity": "High",
                    "factor": "Recruiter Domain Mismatch",
                    "explanation": contact_details
                })

        verif_summary = run_full_verification(company_name, contact_info)

        # 2. Text preprocessing
        clean_text = preprocess_job_text(
            title=title,
            company_profile=company_profile,
            description=description,
            requirements=requirements,
            benefits=benefits
        )

        ml_prob = 0.0
        top_contributing_words = []

        if self.is_ready() and clean_text:
            # TF-IDF sparse vector
            vec = self.vectorizer.transform([clean_text])

            # Dense structural features
            has_prof = 1 if len(str(company_profile).strip()) >= 30 else 0
            has_req = 1 if len(str(requirements).strip()) >= 25 else 0
            has_ben = 1 if len(str(benefits).strip()) >= 15 else 0
            no_logo_no_prof = 1 if (has_company_logo == 0 and has_prof == 0) else 0
            remote_no_logo = 1 if (telecommuting == 1 and has_company_logo == 0) else 0
            remote_no_prof = 1 if (telecommuting == 1 and has_prof == 0) else 0
            desc_len_norm = min(1.0, len(str(description).split()) / 300.0)
            prof_len_norm = min(1.0, len(str(company_profile).split()) / 150.0)

            crit_count = sum(1 for f in flags if f['severity'] == 'Critical')
            high_count = sum(1 for f in flags if f['severity'] == 'High')
            med_count = sum(1 for f in flags if f['severity'] == 'Medium')

            meta_vec = np.array([[
                has_company_logo,
                has_prof,
                prof_len_norm,
                has_req,
                has_ben,
                telecommuting,
                no_logo_no_prof,
                remote_no_logo,
                remote_no_prof,
                desc_len_norm,
                crit_count,
                high_count,
                med_count
            ]], dtype=np.float32)

            # Combined Sparse Matrix
            try:
                X_comb = sp.hstack([vec, sp.csr_matrix(meta_vec)], format="csr")
                if hasattr(self.model, "predict_proba"):
                    probs = self.model.predict_proba(X_comb)[0]
                    ml_prob = float(probs[1])
                else:
                    pred_val = int(self.model.predict(X_comb)[0])
                    ml_prob = 0.9 if pred_val == 1 else 0.1
            except Exception:
                # Fallback if model was trained on text only
                if hasattr(self.model, "predict_proba"):
                    probs = self.model.predict_proba(vec)[0]
                    ml_prob = float(probs[1])

            # Extract matching high-risk vocabulary terms
            fraud_vocab = set(self.metrics.get("top_fraud_words", []))
            input_words = set(clean_text.split())
            detected_fraud_words = list(input_words.intersection(fraud_vocab))
            top_contributing_words = detected_fraud_words[:10]

        # 3. Dynamic Composite Risk Calculation with Calibrated Guardrails
        crit_count = sum(1 for f in flags if f['severity'] == 'Critical')
        high_count = sum(1 for f in flags if f['severity'] == 'High')
        med_count = sum(1 for f in flags if f['severity'] == 'Medium')
        low_count = sum(1 for f in flags if f['severity'] == 'Low')

        # High-risk vocabulary match boost
        vocab_match_count = len(top_contributing_words)
        vocab_boost = min(15.0, vocab_match_count * 3.5)

        # Baseline risk starts from the trained ML probability (0.0 to 100.0)
        ml_score = ml_prob * 100.0

        # Heuristic risk index
        heuristic_score = (crit_count * 45.0) + (high_count * 25.0) + (med_count * 8.0) + (low_count * 2.0) + vocab_boost
        heuristic_score = min(100.0, heuristic_score)

        if crit_count == 0 and high_count == 0:
            # Clean posting: combine ML probability, vocabulary matches, and minor flags
            final_risk = ml_score + vocab_boost + (med_count * 6.0) + (low_count * 1.5)
        else:
            # Confirmed scam triggers present: Blend ML with aggressive heuristic signals
            final_risk = (ml_score * 0.35) + (heuristic_score * 0.65)

        # Enforce Minimum Floor Guardrails for confirmed Critical & High scam signals:
        if crit_count >= 1:
            # Payment, fake checks, Telegram/WhatsApp interviews, consumer emails (@gmail), upfront fees
            final_risk = max(final_risk, 85.0 + (crit_count * 4.0))
        elif high_count >= 2:
            # Multiple high-severity patterns
            final_risk = max(final_risk, 65.0)
        elif high_count == 1:
            # Single high-severity pattern
            final_risk = max(final_risk, 45.0)

        final_risk = float(np.clip(final_risk, 0.5, 99.8))

        # 4. Final Verdict Categorization (Calibrated to Prevent Low-Risk False Negatives)
        if final_risk >= 48.0:
            verdict = "Fraudulent / High Scam Risk"
            risk_level = "High"
            badge_color = "#e53e3e"
            verdict_icon = "🚨"
            verdict_summary = "This job posting exhibits strong patterns commonly associated with fraudulent employment scams."
        elif final_risk >= 20.0:
            verdict = "Suspicious / Moderate Caution"
            risk_level = "Moderate"
            badge_color = "#dd6b20"
            verdict_icon = "⚠️"
            verdict_summary = "Certain aspects of this posting raise caution. Independent verification of the employer is recommended."
        else:
            verdict = "Legitimate / Authentic Job Posting"
            risk_level = "Low"
            badge_color = "#38a169"
            verdict_icon = "✅"
            verdict_summary = "The job listing matches typical language, structure, and attributes of genuine corporate postings."

        # 5. Tailored Safety Recommendations
        recommendations = []
        if any(f['severity'] == 'Critical' for f in flags):
            recommendations.append("⛔ NEVER send money, purchase equipment, or cash a check from a prospective employer.")
        if any('Telegram' in f['factor'] or 'Whatsapp' in f['factor'] for f in flags):
            recommendations.append("💬 Insist on communicating via verified corporate email domains (e.g. @company.com) rather than messaging apps.")
        if any('Email' in f['explanation'] for f in flags):
            recommendations.append("📧 Be wary of recruiters using free email services (@gmail, @yahoo). Check official corporate directory.")
        if not has_company_logo or not (company_profile and len(str(company_profile).strip()) >= 30):
            recommendations.append("🔍 Search the company's official website and LinkedIn page to verify if this exact position is listed on their Careers portal.")
        if telecommuting:
            recommendations.append("💼 For remote jobs, request an official written offer letter and video interview before sharing sensitive personal identification (SSN/ID).")
        recommendations.append("🛡️ Legitimate companies will never demand payment for background checks, training materials, or onboarding software.")

        return {
            "verdict": verdict,
            "risk_level": risk_level,
            "risk_score": round(final_risk, 1),
            "ml_probability": round(ml_prob * 100.0, 1),
            "heuristic_score": round(heuristic_score, 1),
            "badge_color": badge_color,
            "verdict_icon": verdict_icon,
            "verdict_summary": verdict_summary,
            "verification": verif_summary,
            "flags": flags,
            "contributing_words": top_contributing_words,
            "recommendations": recommendations,
            "clean_text_preview": clean_text[:200] + ("..." if len(clean_text) > 200 else "")
        }
