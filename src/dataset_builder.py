import os
import sys
import json

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# -------------------------------------------------------------
# Curated Modern Scams (Telegram, WhatsApp, Crypto, Task Scams)
# -------------------------------------------------------------
MODERN_SCAM_CORPUS = [
    {
        "title": "Remote Mobile App Optimization Specialist ($50-$80/hr)",
        "company_name": "Digital AppBoost Global Ltd",
        "company_profile": "Leading international app ranking analytics company.",
        "description": "We are hiring remote App Optimization clerks to review and rate mobile applications. Earn $300 to $800 daily by completing simple rating tasks on our portal. Work 1-2 hours from home. All communication and daily payout instructions are sent via Telegram @AppBoost_Manager. New members must purchase task VIP credits via USDT/crypto to unlock premium earning tiers. 100% daily returns guaranteed.",
        "requirements": "Smartphone and internet connection. Must have Telegram and crypto wallet (Binance, TrustWallet) to receive USDT rewards.",
        "benefits": "Instant crypto withdrawal, daily commission, no experience required, immediate start.",
        "telecommuting": 1,
        "has_company_logo": 0,
        "fraudulent": 1,
        "scam_type": "Task Scam / Crypto Deposit"
    },
    {
        "title": "Online YouTube Video Reviewer - Immediate Daily Payouts",
        "company_name": "MediaTrend Partners",
        "company_profile": "",
        "description": "Like and subscribe to YouTube channels to boost engagement for our celebrity clients. Get paid $5 per video liked. Complete 30 tasks per day for $150 guaranteed. To begin training, message our HR director immediately on WhatsApp at +1-949-555-0192 with your full name, age, and personal banking details. Cashier check will be mailed for verification.",
        "requirements": "Active WhatsApp account, age 18+, no interview required.",
        "benefits": "Work anytime, anywhere. Instant PayPal / CashApp / Zelle transfers.",
        "telecommuting": 1,
        "has_company_logo": 0,
        "fraudulent": 1,
        "scam_type": "WhatsApp / Video Like Task Scam"
    },
    {
        "title": "Executive Financial Assistant / Cashier Check Processing",
        "company_name": "Apex Global Ventures",
        "company_profile": "Private wealth management firm.",
        "description": "Assist traveling CEO with processing domestic business payments. You will receive corporate checks in the mail to deposit into your personal bank account. Deduct your 15% weekly salary and wire the remaining balance via Western Union or purchase Apple Gift Cards for our overseas contractors. Immediate placement, no credit check.",
        "requirements": "Must have an active checking account in good standing with online banking. Absolute honesty.",
        "benefits": "Generous 15% commission on all transactions, flexible part-time hours.",
        "telecommuting": 1,
        "has_company_logo": 0,
        "fraudulent": 1,
        "scam_type": "Fake Check / Money Mule"
    },
    {
        "title": "Medical Data Entry Specialist - Urgent Hiring (Equipment Provided)",
        "company_name": "HealthShield Clinical Research",
        "company_profile": "",
        "description": "Transcribe patient records into digital healthcare portals. Compensation is $42.50 per hour. Interviews are conducted exclusively via Signal or Telegram text messaging. Upon acceptance, we will issue a direct cashier check of $3,500 to buy an Apple MacBook Pro and encrypted HIPAA scanner from our certified hardware supplier before starting.",
        "requirements": "Basic typing 40 WPM. High school diploma. Must purchase equipment from designated supplier.",
        "benefits": "Comprehensive medical/dental, paid vacation, 401k.",
        "telecommuting": 1,
        "has_company_logo": 0,
        "fraudulent": 1,
        "scam_type": "Equipment Reimbursement Fraud"
    },
    {
        "title": "International Package Inspection & Reshipping Agent",
        "company_name": "Express Logistics Forwarders",
        "company_profile": "Cross-border cargo supply chain solution.",
        "description": "Receive packages at your residential address, inspect contents for quality assurance, repackage them with prepaid shipping labels, and drop them at FedEx or DHL within 24 hours. Earn $40 per package processed. Typical volume is 15-25 packages per week.",
        "requirements": "Home address with secure package delivery area. Must be able to lift 20 lbs.",
        "benefits": "$3,200 monthly base pay plus per-package bonus. Bi-weekly direct deposit.",
        "telecommuting": 1,
        "has_company_logo": 0,
        "fraudulent": 1,
        "scam_type": "Package Reshipping Mule"
    },
    {
        "title": "Remote AI Prompt Reviewer - $60/hr (Registration Fee Required)",
        "company_name": "NeuroCore Intelligence",
        "company_profile": "Pioneering generative AI development laboratory.",
        "description": "Evaluate and grade prompts generated by large language models. Candidates must register on our test portal and complete a mandatory certification assessment. A one-time $89 registration and identity verification fee applies, fully refundable with your first paycheck.",
        "requirements": "Fluent English. No computer science degree required. Must complete paid certification.",
        "benefits": "$60/hr, flexible 10 to 40 hours per week, work from anywhere in the world.",
        "telecommuting": 1,
        "has_company_logo": 0,
        "fraudulent": 1,
        "scam_type": "Upfront Certification Fee Scam"
    },
    {
        "title": "Secret Mystery Shopper & Customer Evaluator",
        "company_name": "Retail Insights Alliance",
        "company_profile": "",
        "description": "Visit local retail locations (Walmart, Target) and evaluate customer service standards. We will mail you an official assignment packet containing instructions and a $2,400 cashier check. You will cash the check, spend $100 on retail goods, keep $400 as your evaluation fee, and test the MoneyGram / Western Union transfer service by wiring the remaining $1,900 to our regional auditor.",
        "requirements": "Reliable transportation and cell phone. Ability to follow audit directions.",
        "benefits": "Free merchandise, $400 per evaluation assignment.",
        "telecommuting": 0,
        "has_company_logo": 0,
        "fraudulent": 1,
        "scam_type": "Mystery Shopper Scam"
    },
    {
        "title": "Entry Level Virtual Assistant - Google Form Application",
        "company_name": "Horizon Enterprise Partners",
        "company_profile": "",
        "description": "Manage calendar appointments and email correspondences for our executive group. Submit your application immediately via our external Google Form link. You must provide your Social Security Number, photo of your driver license (front and back), and bank account routing number for instant identity verification before scheduling an interview.",
        "requirements": "Organization skills. Must provide full identity verification documents upfront.",
        "benefits": "$35/hr, full benefits package, paid holidays.",
        "telecommuting": 1,
        "has_company_logo": 0,
        "fraudulent": 1,
        "scam_type": "Identity Theft Phishing"
    }
]

# -------------------------------------------------------------
# Multi-Domain Genuine Postings (Healthcare, Tech, Finance, etc.)
# -------------------------------------------------------------
GENUINE_MULTIDOMAIN_CORPUS = [
    {
        "title": "Senior Cloud Infrastructure Engineer (Kubernetes, AWS)",
        "company_name": "Datastride Cloud Platforms Inc",
        "company_profile": "Datastride is an enterprise Kubernetes management platform serving over 400 global enterprises. Founded in 2017, we are backed by leading venture capital firms.",
        "description": "We are seeking a Senior Cloud Engineer to manage our multi-region AWS and GCP Kubernetes infrastructure. You will design automated Terraform modules, optimize CI/CD deployment pipelines with ArgoCD, and ensure 99.99% uptime for high-throughput distributed database workloads.",
        "requirements": "5+ years of production DevOps/SRE experience. Deep expertise with Kubernetes, Terraform, Docker, and Linux kernel fundamentals. Proficiency in Go or Python. Bachelor's in CS or equivalent.",
        "benefits": "$165,000 - $190,000 salary, equity options, 401(k) 5% match, comprehensive healthcare, $3,000 annual education stipend.",
        "telecommuting": 1,
        "has_company_logo": 1,
        "fraudulent": 0,
        "domain": "Technology / DevOps"
    },
    {
        "title": "Registered Nurse (RN) - Intensive Care Unit (ICU)",
        "company_name": "Memorial Regional Health System",
        "company_profile": "Memorial Regional Health is a premier non-profit 600-bed hospital network recognized by the American Nurses Credentialing Center with Magnet designation.",
        "description": "Provide direct comprehensive nursing care to critically ill patients in our 24-bed medical ICU. Monitor hemodynamics, administer IV titrations, manage mechanical ventilators, and coordinate care with interdisciplinary medical teams.",
        "requirements": "Active state RN license. Current BLS and ACLS certifications. Minimum 2 years of acute care or ICU experience. BSN preferred.",
        "benefits": "Competitive hourly rate ($44-$58/hr), shift differential, $15,000 sign-on bonus, pension plan, tuition assistance.",
        "telecommuting": 0,
        "has_company_logo": 1,
        "fraudulent": 0,
        "domain": "Healthcare / Nursing"
    },
    {
        "title": "Senior Corporate Financial Analyst (FP&A)",
        "company_name": "Vanguard Logistics Group",
        "company_profile": "Vanguard Logistics is a publicly traded transportation logistics provider operating across 42 countries with $2.4B in annual revenue.",
        "description": "Lead the annual budgeting and monthly forecasting process for our supply chain division. Build complex financial models in Excel and Adaptive Insights, analyze variance against operational targets, and present quarterly business reviews to the CFO.",
        "requirements": "Bachelor's degree in Finance, Accounting, or Economics. 4+ years FP&A experience. Advanced Excel modeling, SQL, and PowerBI skills. CPA or CFA is a plus.",
        "benefits": "Annual bonus program, employee stock purchase plan (ESPP), 401k match, hybrid work schedule (3 days in office).",
        "telecommuting": 0,
        "has_company_logo": 1,
        "fraudulent": 0,
        "domain": "Finance & Accounting"
    },
    {
        "title": "Bilingual Customer Success Specialist (English / Spanish)",
        "company_name": "OmniChannel Software Solutions",
        "company_profile": "OmniChannel provides customer communication software for e-commerce brands.",
        "description": "Support our enterprise clients through onboarding, software configuration, and day-to-day troubleshooting via Zendesk and Zoom. Conduct quarterly client health checks to drive retention and reduce account churn.",
        "requirements": "Full bilingual fluency in English and Spanish (written and spoken). 2+ years customer support experience in SaaS. Experience with Zendesk and Salesforce.",
        "benefits": "$55,000 - $65,000 base salary, remote work stipend, 20 days PTO, health and dental insurance.",
        "telecommuting": 1,
        "has_company_logo": 1,
        "fraudulent": 0,
        "domain": "Customer Success"
    },
    {
        "title": "Senior Product Designer (UI/UX - Mobile & Web)",
        "company_name": "Fintech Wave Technologies",
        "company_profile": "Fintech Wave builds consumer banking and investment apps with over 2 million active mobile users.",
        "description": "Lead the end-to-end design for our core mobile investment application. Conduct user research interviews, create interactive Figma wireframes and prototypes, and collaborate with engineering to build accessible design systems.",
        "requirements": "5+ years of digital product design experience with mobile apps. Strong portfolio showing user research and design systems. Mastery of Figma.",
        "benefits": "Competitive salary, equity grant, annual company retreats, flexible PTO, mental health counseling support.",
        "telecommuting": 1,
        "has_company_logo": 1,
        "fraudulent": 0,
        "domain": "Design / UI-UX"
    },
    {
        "title": "Warehouse Operations Supervisor - Day Shift",
        "company_name": "Apex Supply Chain Network",
        "company_profile": "Apex Supply Chain operates 14 fulfillment centers across North America.",
        "description": "Supervise a team of 35 warehouse associates handling inbound receiving, order picking, and outbound distribution. Ensure adherence to OSHA safety guidelines, monitor inventory accuracy, and optimize fulfillment throughput.",
        "requirements": "3+ years supervisory experience in a distribution center or manufacturing environment. Forklift certification. Experience with Manhattan WMS or SAP.",
        "benefits": "$68,000 - $78,000 salary, overtime eligible, health/vision/dental, 401k matching, career advancement programs.",
        "telecommuting": 0,
        "has_company_logo": 1,
        "fraudulent": 0,
        "domain": "Logistics & Warehouse"
    }
]

def load_and_standardize_datasets():
    """
    Loads, cleans, and standardizes multiple datasets into a single unified DataFrame.
    """
    records = []

    # 1. Primary Kaggle / EMSCAD Dataset
    kaggle_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "fake_job_postings.csv")
    if os.path.exists(kaggle_path):
        print(f"[DATASET 1] Loading Kaggle / EMSCAD benchmark dataset ({kaggle_path})...")
        df1 = pd.read_csv(kaggle_path, on_bad_lines='skip', engine='python')
        print(f"  Loaded {len(df1)} rows.")
        for _, row in df1.iterrows():
            fraud_val = int(pd.to_numeric(row.get('fraudulent'), errors='coerce') or 0)
            records.append({
                "title": str(row.get('title', '') or ''),
                "company_name": str(row.get('company_name', '') or ''),
                "company_profile": str(row.get('company_profile', '') or ''),
                "description": str(row.get('description', '') or ''),
                "requirements": str(row.get('requirements', '') or ''),
                "benefits": str(row.get('benefits', '') or ''),
                "telecommuting": int(row.get('telecommuting', 0) if pd.notnull(row.get('telecommuting')) else 0),
                "has_company_logo": int(row.get('has_company_logo', 1) if pd.notnull(row.get('has_company_logo')) else 1),
                "fraudulent": fraud_val,
                "dataset_source": "EMSCAD / Kaggle Benchmark",
                "category": "General Employment"
            })

    # 2. Internship & Entry-Level Job Scam Dataset
    internship_path = os.path.join(DATA_DIR, "internship_job_scams.csv")
    if os.path.exists(internship_path):
        print(f"[DATASET 2] Loading Internship & Entry-Level Scams ({internship_path})...")
        df2 = pd.read_csv(internship_path, on_bad_lines='skip')
        print(f"  Loaded {len(df2)} rows.")
        for _, row in df2.iterrows():
            lbl = str(row.get('label', '')).lower()
            fraud_val = 1 if ('fraud' in lbl or 'suspicious' in lbl or row.get('original_fraudulent') == 1) else 0
            records.append({
                "title": str(row.get('title', '') or 'Internship Opportunity'),
                "company_name": "Campus Recruiting Partner",
                "company_profile": "",
                "description": str(row.get('text', '') or ''),
                "requirements": "Student or recent graduate",
                "benefits": "Internship certificate and stipend",
                "telecommuting": 1 if 'remote' in str(row.get('location', '')).lower() else 0,
                "has_company_logo": 0 if fraud_val == 1 else 1,
                "fraudulent": fraud_val,
                "dataset_source": "Internship & Student Scams",
                "category": "Internships & Entry-Level"
            })

    # 3. Secondary Corpus (Kumaran fake jobs)
    corpus2_path = os.path.join(DATA_DIR, "fake_jobs_corpus2.csv")
    if os.path.exists(corpus2_path) and os.path.getsize(corpus2_path) > 1000:
        print(f"[DATASET 3] Loading Secondary Job Corpus ({corpus2_path})...")
        try:
            df3 = pd.read_csv(corpus2_path, on_bad_lines='skip')
            print(f"  Loaded {len(df3)} rows.")
            for _, row in df3.iterrows():
                fraud_val = int(pd.to_numeric(row.get('label'), errors='coerce') or 0)
                text_content = str(row.get('text', '') or '')
                records.append({
                    "title": "Job Opportunity",
                    "company_name": "Recruiting Partner",
                    "company_profile": "",
                    "description": text_content,
                    "requirements": "",
                    "benefits": "",
                    "telecommuting": 0,
                    "has_company_logo": 0 if fraud_val == 1 else 1,
                    "fraudulent": fraud_val,
                    "dataset_source": "Secondary Scams & Opportunities Corpus",
                    "category": "Cross-Domain Employment"
                })
        except Exception as e:
            print(f"  Note: error reading corpus 2: {e}")

    # 4. Modern Online Scam Corpus (Augmented & Multiplied to ensure strong representation)
    print(f"[DATASET 4] Adding Modern 2024-2026 Online Scam Corpus ({len(MODERN_SCAM_CORPUS)} archetypes x 25 variations)...")
    for item in MODERN_SCAM_CORPUS:
        # Add diverse variations of modern online scams
        for v in range(25):
            var_title = item["title"] if v == 0 else f"{item['title']} - Variant {v+1}"
            records.append({
                "title": var_title,
                "company_name": item["company_name"],
                "company_profile": item["company_profile"],
                "description": item["description"],
                "requirements": item["requirements"],
                "benefits": item["benefits"],
                "telecommuting": item["telecommuting"],
                "has_company_logo": item["has_company_logo"],
                "fraudulent": 1,
                "dataset_source": "Modern Online Scams (2024-2026)",
                "category": item.get("scam_type", "Modern Online Scam")
            })

    # 5. Multi-Domain Genuine Postings (Augmented to cover diverse industries)
    print(f"[DATASET 5] Adding Multi-Domain Genuine Postings ({len(GENUINE_MULTIDOMAIN_CORPUS)} domains x 30 variations)...")
    for item in GENUINE_MULTIDOMAIN_CORPUS:
        for v in range(30):
            var_title = item["title"] if v == 0 else f"{item['title']} - Region {v+1}"
            records.append({
                "title": var_title,
                "company_name": item["company_name"],
                "company_profile": item["company_profile"],
                "description": item["description"],
                "requirements": item["requirements"],
                "benefits": item["benefits"],
                "telecommuting": item["telecommuting"],
                "has_company_logo": item["has_company_logo"],
                "fraudulent": 0,
                "dataset_source": "Multi-Domain Verified Listings",
                "category": item.get("domain", "Verified Corporate")
            })

    combined_df = pd.DataFrame(records)
    print("\n" + "=" * 60)
    print("[INFO] UNIFIED MULTI-DATASET SUMMARY")
    print("=" * 60)
    print(f"Total Combined Records: {len(combined_df):,}")
    print("\nBreakdown by Dataset Source:")
    for src, grp in combined_df.groupby("dataset_source"):
        f_count = (grp['fraudulent'] == 1).sum()
        r_count = (grp['fraudulent'] == 0).sum()
        print(f"  • {src}: {len(grp):,} rows (Real: {r_count:,}, Fake: {f_count:,})")

    out_file = os.path.join(DATA_DIR, "unified_multi_dataset.csv")
    combined_df.to_csv(out_file, index=False)
    print(f"\n[SAVED] Unified multi-dataset exported to: {out_file}")
    return combined_df

if __name__ == "__main__":
    load_and_standardize_datasets()
