import os
import io
import csv
import json
import pandas as pd
from flask import Flask, render_template, request, jsonify, Response, send_file, send_from_directory
from src.predictor import JobFraudPredictor

app = Flask(__name__)
predictor = JobFraudPredictor()

# Curated Presets for 1-Click Evaluation across multiple dataset domains
PRESETS = {
    "fake_internship": {
        "label": "🚨 Fake: Student Internship (Paid Training Fee Scam)",
        "title": "Software Engineering Virtual Internship - 2026 Batch",
        "company_name": "NextGen Campus Technologies",
        "company_profile": "",
        "description": "Exclusive internship program for university students and fresh graduates. Work on real-time client projects from home. To secure your slot, candidates must pay a non-refundable $120 registration and cloud lab kit fee via PayPal or crypto. Interviews conducted via WhatsApp group chat.",
        "requirements": "Basic programming knowledge in Python or Java. Enrollment in university. Must complete fee payment prior to orientation.",
        "benefits": "Letter of recommendation, guaranteed placement referral, flexible schedule.",
        "telecommuting": 1,
        "has_company_logo": 0,
        "employment_type": "Temporary",
        "required_experience": "Entry level"
    },
    "fake_task_app": {
        "label": "🚨 Fake: Mobile App Rating Task Scam (USDT Crypto)",
        "title": "Remote App Optimization Specialist ($300-$800 Daily)",
        "company_name": "Digital AppBoost Global Ltd",
        "company_profile": "",
        "description": "We are hiring remote clerks to review and rate mobile apps on our partner platform. Earn $300 to $800 daily by completing simple rating tasks from home. Work 1-2 hours a day. Payouts sent via USDT/crypto. Members must purchase task VIP credits to unlock high-yield commission tiers.",
        "requirements": "Smartphone, Telegram account, crypto wallet (Binance/TrustWallet) for receiving daily withdrawals.",
        "benefits": "Instant crypto withdrawal, daily bonuses, no experience needed.",
        "telecommuting": 1,
        "has_company_logo": 0,
        "employment_type": "Part-time",
        "required_experience": "Entry level",
        "contact_info": "@AppBoost_Manager (Telegram)"
    },
    "fake_telegram": {
        "label": "🚨 Fake: Remote Data Entry $45/hr (Telegram Contact)",
        "title": "Data Entry Specialist - Immediate Remote Start",
        "company_name": "Global Asset Management Services",
        "company_profile": "",
        "description": "We are seeking reliable home-based Data Entry clerks. Compensation is $45 per hour with guaranteed weekly payouts. Hours are completely flexible. Interviews will be conducted immediately via Telegram app @HR_Recruiter_Official. Selected candidates will receive a cashier check to purchase home office equipment from our approved vendor.",
        "requirements": "Must have basic computer and reliable internet. No previous experience needed. High school diploma or equivalent.",
        "benefits": "Flexible schedule, weekly wire transfer, health coverage after 90 days.",
        "telecommuting": 1,
        "has_company_logo": 0,
        "employment_type": "Full-time",
        "required_experience": "Entry level",
        "contact_info": "@HR_Recruiter_Official (Telegram)"
    },
    "fake_crypto": {
        "label": "🚨 Fake: Crypto Payment Assistant / Wire Transfer",
        "title": "Cryptocurrency Transaction Processing Assistant",
        "company_name": "Apex Wealth Solutions",
        "company_profile": "A fast growing international confidential client financial firm.",
        "description": "Assist our executive management with processing regional payments, crypto conversions (Bitcoin, USDT) and daily wire transfers via Cash App / Western Union. Earn 10% commission per transaction plus $800 weekly base pay.",
        "requirements": "Must have an active personal bank account in good standing. Immediate availability.",
        "benefits": "High commission rates, work from home, weekly direct deposit.",
        "telecommuting": 1,
        "has_company_logo": 0,
        "employment_type": "Part-time",
        "required_experience": "Not Specified",
        "contact_info": "recruiter.crypto@gmail.com"
    },
    "real_engineer": {
        "label": "✅ Real: Cloud DevOps Engineer (Tech)",
        "title": "Senior Cloud Infrastructure Engineer (Kubernetes, AWS)",
        "company_name": "Datastride Cloud Platforms Inc",
        "company_profile": "Datastride is an enterprise Kubernetes management platform serving over 400 global enterprises. Founded in 2017, we are backed by leading venture capital firms.",
        "description": "We are seeking a Senior Cloud Engineer to manage our multi-region AWS and GCP Kubernetes infrastructure. You will design automated Terraform modules, optimize CI/CD deployment pipelines with ArgoCD, and ensure 99.99% uptime for high-throughput distributed database workloads.",
        "requirements": "5+ years of production DevOps/SRE experience. Deep expertise with Kubernetes, Terraform, Docker, and Linux kernel fundamentals. Proficiency in Go or Python. Bachelor's in CS or equivalent practical experience.",
        "benefits": "$165,000 - $190,000 salary, equity options, 401(k) 5% match, comprehensive healthcare, $3,000 annual education stipend.",
        "telecommuting": 1,
        "has_company_logo": 1,
        "employment_type": "Full-time",
        "required_experience": "Mid-Senior level",
        "contact_info": "https://boards.greenhouse.io/datastride/jobs/48201"
    },
    "real_nurse": {
        "label": "✅ Real: ICU Registered Nurse (Healthcare)",
        "title": "Registered Nurse (RN) - Intensive Care Unit (ICU)",
        "company_name": "Memorial Regional Health System",
        "company_profile": "Memorial Regional Health is a premier non-profit 600-bed hospital network recognized by the American Nurses Credentialing Center with Magnet designation.",
        "description": "Provide direct comprehensive nursing care to critically ill patients in our 24-bed medical ICU. Monitor hemodynamics, administer IV titrations, manage mechanical ventilators, and coordinate care with interdisciplinary medical teams.",
        "requirements": "Active state RN license. Current BLS and ACLS certifications. Minimum 2 years of acute care or ICU experience. BSN preferred.",
        "benefits": "Competitive hourly rate ($44-$58/hr), shift differential, $15,000 sign-on bonus, pension plan, tuition assistance.",
        "telecommuting": 0,
        "has_company_logo": 1,
        "employment_type": "Full-time",
        "required_experience": "Associate",
        "contact_info": "https://memorialhealth.myworkdayjobs.com/en-US/careers/job/1029"
    },
    "real_finance": {
        "label": "✅ Real: Senior Financial Analyst (Corporate FP&A)",
        "title": "Senior Corporate Financial Analyst (FP&A)",
        "company_name": "Vanguard Logistics Group",
        "company_profile": "Vanguard Logistics is a publicly traded transportation logistics provider operating across 42 countries with $2.4B in annual revenue.",
        "description": "Lead the annual budgeting and monthly forecasting process for our supply chain division. Build complex financial models in Excel and Adaptive Insights, analyze variance against operational targets, and present quarterly business reviews to executive management.",
        "requirements": "Bachelor's degree in Finance, Accounting, or Economics. 4+ years FP&A experience. Advanced Excel modeling, SQL, and PowerBI skills. CPA or CFA is a plus.",
        "benefits": "Annual bonus program, employee stock purchase plan (ESPP), 401k match, hybrid work schedule (3 days in office).",
        "telecommuting": 0,
        "has_company_logo": 1,
        "employment_type": "Full-time",
        "required_experience": "Mid-Senior level",
        "contact_info": "careers@vanguardlogistics.com"
    }
}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/presets", methods=["GET"])
def get_presets():
    return jsonify(PRESETS)

@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        title = data.get("title", "")
        company_name = data.get("company_name", "")
        desc = data.get("description", "")
        print(f"\n[SCAN_REQUEST] Title: '{title}' | Company: '{company_name}' | Desc snippet: '{desc[:90]}...' | Logo: {data.get('has_company_logo')}", flush=True)

        result = predictor.predict(
            title=title,
            company_name=company_name,
            company_profile=data.get("company_profile", ""),
            description=desc,
            requirements=data.get("requirements", ""),
            benefits=data.get("benefits", ""),
            telecommuting=int(data.get("telecommuting", 0)),
            has_company_logo=int(data.get("has_company_logo", 1)),
            has_questions=int(data.get("has_questions", 0)),
            employment_type=data.get("employment_type", "Full-time"),
            required_experience=data.get("required_experience", "Not Specified"),
            contact_info=data.get("contact_info", "")
        )
        print(f"[SCAN_RESULT] Verdict: {result['verdict']} | Risk Score: {result['risk_score']}% | ML Prob: {result['ml_probability']}% | Flags: {len(result['flags'])}", flush=True)
        return jsonify({"success": True, "data": result})
    except Exception as e:
        print(f"[SCAN_ERROR] {e}", flush=True)
        return jsonify({"success": False, "error": str(e)}), 400

@app.route("/api/batch-scan", methods=["POST"])
def batch_scan():
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"success": False, "error": "Empty filename"}), 400

    try:
        content = file.read().decode("utf-8", errors="ignore")
        df = pd.read_csv(io.StringIO(content), on_bad_lines="skip")
        
        results = []
        for idx, row in df.iterrows():
            res = predictor.predict(
                title=str(row.get("title", "")),
                company_name=str(row.get("company_name", "")),
                company_profile=str(row.get("company_profile", "")),
                description=str(row.get("description", "")),
                requirements=str(row.get("requirements", "")),
                benefits=str(row.get("benefits", "")),
                telecommuting=int(row.get("telecommuting", 0) if pd.notnull(row.get("telecommuting")) else 0),
                has_company_logo=int(row.get("has_company_logo", 1) if pd.notnull(row.get("has_company_logo")) else 1),
                contact_info=str(row.get("contact_info", "")) if pd.notnull(row.get("contact_info")) else ""
            )
            results.append({
                "job_id": row.get("job_id", idx + 1),
                "title": row.get("title", f"Job #{idx+1}"),
                "company_name": row.get("company_name", "N/A"),
                "risk_score": res["risk_score"],
                "risk_level": res["risk_level"],
                "verdict": res["verdict"],
                "flags_count": len(res["flags"]),
                "flags_detail": "; ".join([f"{f['factor']} ({f['severity']})" for f in res["flags"]])
            })

        total = len(results)
        fake_count = sum(1 for r in results if r["risk_level"] == "High")
        moderate_count = sum(1 for r in results if r["risk_level"] == "Moderate")
        legit_count = sum(1 for r in results if r["risk_level"] == "Low")

        summary = {
            "total": total,
            "fake_count": fake_count,
            "moderate_count": moderate_count,
            "legit_count": legit_count,
            "fake_percentage": round(fake_count / total * 100, 1) if total else 0
        }

        return jsonify({
            "success": True,
            "summary": summary,
            "results": results
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

@app.route("/api/download-template", methods=["GET"])
def download_template():
    sample_csv = (
        "job_id,title,company_name,company_profile,description,requirements,benefits,telecommuting,has_company_logo\n"
        "101,Software Engineer,Acme Cloud,Acme is an enterprise database provider.,Develop backend services in Python.,3+ years Python experience.,Health insurance and 401k.,0,1\n"
        "102,Remote Data Entry Clerk,FastCash Logistics,,Earn $60/hr from home. Contact HR on Telegram @ScamJob. Cashier check provided for equipment.,No experience needed.,Weekly wire transfer.,1,0\n"
    )
    return Response(
        sample_csv,
        mimetype="text/csv",
        headers={"Content-disposition": "attachment; filename=job_postings_template.csv"}
    )

@app.route("/api/analytics", methods=["GET"])
def get_analytics():
    return jsonify({
        "success": True,
        "eda": predictor.eda_summary,
        "metrics": predictor.metrics
    })

@app.route("/favicon.ico")
def favicon():
    return send_from_directory(
        os.path.join(app.root_path, "static"),
        "favicon.ico",
        mimetype="image/vnd.microsoft.icon"
    )

if __name__ == "__main__":
    print("[INFO] Starting JobShield AI Flask Web Application on http://127.0.0.1:5000...")
    app.run(host="0.0.0.0", port=5000, debug=False)
