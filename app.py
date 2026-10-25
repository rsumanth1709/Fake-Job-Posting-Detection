import os
import io
import json
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from src.predictor import JobFraudPredictor
from src.preprocessing import clean_text, remove_stopwords

# Page Configuration
st.set_page_config(
    page_title="JobShield AI | Fake Job Posting Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        padding: 2rem 2.5rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .main-header h1 {
        color: white !important;
        font-size: 2.3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .main-header p {
        color: #e0e7ff;
        font-size: 1.05rem;
        margin-bottom: 0;
    }
    .metric-card {
        background-color: #f8fafc;
        border-radius: 10px;
        padding: 1.25rem;
        border: 1px solid #e2e8f0;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: 0.25rem;
    }
    .badge-critical {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 0.25rem 0.6rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-high {
        background-color: #ffedd5;
        color: #9a3412;
        padding: 0.25rem 0.6rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-medium {
        background-color: #fef9c3;
        color: #854d0e;
        padding: 0.25rem 0.6rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .verdict-box {
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        border-left: 6px solid;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 8px 8px 0 0;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_predictor():
    return JobFraudPredictor()

predictor = get_predictor()

# Curated Presets for Quick 1-Click Evaluation
PRESETS = {
    "Select a preset...": None,
    "🚨 Fake: Remote Data Entry $45/hr (Telegram Contact)": {
        "title": "Data Entry Specialist - Immediate Remote Start",
        "company_name": "Global Asset Management Services",
        "company_profile": "",
        "description": "We are seeking reliable home-based Data Entry clerks. Compensation is $45 per hour with guaranteed weekly payouts. Hours are completely flexible. Interviews will be conducted immediately via Telegram app @HR_Recruiter_Official. Selected candidates will receive a cashier check to purchase home office equipment from our approved vendor.",
        "requirements": "Must have basic computer and reliable internet. No previous experience needed. High school diploma or equivalent.",
        "benefits": "Flexible schedule, weekly wire transfer, health coverage after 90 days.",
        "telecommuting": 1,
        "has_company_logo": 0,
        "employment_type": "Full-time",
        "required_experience": "Entry level"
    },
    "🚨 Fake: Crypto Financial Assistant / Wire Transfer": {
        "title": "Cryptocurrency Transaction Processing Assistant",
        "company_name": "Apex Wealth Solutions",
        "company_profile": "A fast growing international confidential client financial firm.",
        "description": "Assist our executive management with processing regional payments, crypto conversions (Bitcoin, USDT) and daily wire transfers via Cash App / Western Union. Earn 10% commission per transaction plus $800 weekly base pay.",
        "requirements": "Must have an active personal bank account in good standing. Immediate availability.",
        "benefits": "High commission rates, work from home, weekly direct deposit.",
        "telecommuting": 1,
        "has_company_logo": 0,
        "employment_type": "Part-time",
        "required_experience": "Not Specified"
    },
    "✅ Real: Senior Software Engineer (Legitimate Tech)": {
        "title": "Senior Backend Software Engineer (Python / Distributed Systems)",
        "company_name": "Novatech Cloud Solutions",
        "company_profile": "Novatech is an enterprise cloud infrastructure company founded in 2014, serving Fortune 500 enterprises with mission-critical database scaling solutions.",
        "description": "We are looking for a Senior Backend Engineer to design and implement highly scalable microservices in Python and Go. You will collaborate with cross-functional teams to build resilient distributed systems and participate in code reviews and architecture design sessions.",
        "requirements": "5+ years professional experience with Python or Go. Solid knowledge of PostgreSQL, Redis, Docker, and Kubernetes. Strong CS fundamentals in algorithms and system design. BS/MS in Computer Science or equivalent practical experience.",
        "benefits": "Competitive salary, 401(k) matching, comprehensive health/dental/vision insurance, 4 weeks paid vacation, continuous learning budget.",
        "telecommuting": 0,
        "has_company_logo": 1,
        "employment_type": "Full-time",
        "required_experience": "Mid-Senior level"
    },
    "✅ Real: Digital Marketing Manager (Legitimate Agency)": {
        "title": "Digital Marketing & Performance Manager",
        "company_name": "BrightPeak Media Group",
        "company_profile": "BrightPeak Media is an award-winning digital agency with offices in New York and London, helping international consumer brands scale acquisition.",
        "description": "Manage multi-channel paid acquisition campaigns across Google Ads, Meta, and LinkedIn. Analyze campaign performance metrics (CAC, ROAS, LTV) and collaborate with our creative team to produce high-converting ad assets.",
        "requirements": "3+ years leading paid search and social campaigns with monthly budgets over $50k. Proficiency in Google Analytics 4, Looker Studio, and Tag Manager. Strong analytical mindset and written communication skills.",
        "benefits": "Health insurance, hybrid work flexibility, gym membership, annual performance bonus.",
        "telecommuting": 0,
        "has_company_logo": 1,
        "employment_type": "Full-time",
        "required_experience": "Associate"
    }
}

# App Header
st.markdown("""
<div class="main-header">
    <div style="display: flex; align-items: center; justify-content: space-between;">
        <div>
            <h1>🛡️ JobShield AI</h1>
            <p>Enterprise Machine Learning & NLP System for Detecting Fraudulent Job Postings & Employment Scams</p>
        </div>
        <div style="background: rgba(255,255,255,0.15); padding: 8px 16px; border-radius: 8px; font-size: 0.9rem;">
            NLP Engine: <b>TF-IDF + Balanced Ensemble</b>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Main Navigation Tabs
tab_single, tab_batch, tab_analytics, tab_guide = st.tabs([
    "🔍 Job Posting Scanner",
    "📁 Batch File Scanner (CSV)",
    "📊 Dataset & ML Analytics",
    "🛡️ Scam Prevention Guide"
])

# ==========================================
# TAB 1: SINGLE JOB SCANNER
# ==========================================
with tab_single:
    st.markdown("### 📋 Analyze Individual Job Listing")
    st.write("Paste the details of any job posting below to evaluate its legitimacy score and detect fraudulent patterns.")

    # Preset selection
    col_preset, col_clear = st.columns([4, 1])
    with col_preset:
        preset_choice = st.selectbox(
            "⚡ Quick Load Sample Postings (Test with 1-click):",
            options=list(PRESETS.keys()),
            index=0
        )
    with col_clear:
        st.write("")
        st.write("")
        if st.button("🔄 Reset Form", use_container_width=True):
            st.session_state["job_form_data"] = {}
            st.rerun()

    selected_preset = PRESETS.get(preset_choice) or {}

    with st.form("job_analysis_form"):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            title_input = st.text_input(
                "Job Title *",
                value=selected_preset.get("title", ""),
                placeholder="e.g. Remote Data Entry Specialist, Senior Python Engineer..."
            )
            company_input = st.text_input(
                "Company Name",
                value=selected_preset.get("company_name", ""),
                placeholder="e.g. Global Tech Inc, Apex Solutions..."
            )
            profile_input = st.text_area(
                "Company Profile / Background",
                value=selected_preset.get("company_profile", ""),
                placeholder="About the company, history, industry presence... (Scammers often omit this!)",
                height=90
            )
            desc_input = st.text_area(
                "Job Description *",
                value=selected_preset.get("description", ""),
                placeholder="Main duties, day-to-day responsibilities, how the interview is conducted...",
                height=150
            )

        with col2:
            st.markdown("#### ⚙️ Job Attributes & Metadata")
            telecommuting_val = st.checkbox(
                "🏠 Fully Remote / Telecommuting",
                value=bool(selected_preset.get("telecommuting", False))
            )
            has_logo_val = st.checkbox(
                "🏢 Company Logo Present on Listing",
                value=bool(selected_preset.get("has_company_logo", True))
            )
            emp_type_val = st.selectbox(
                "Employment Type",
                ["Full-time", "Part-time", "Contract", "Temporary", "Other"],
                index=["Full-time", "Part-time", "Contract", "Temporary", "Other"].index(
                    selected_preset.get("employment_type", "Full-time")
                ) if selected_preset.get("employment_type") in ["Full-time", "Part-time", "Contract", "Temporary", "Other"] else 0
            )
            exp_val = st.selectbox(
                "Required Experience",
                ["Entry level", "Associate", "Mid-Senior level", "Director", "Executive", "Not Specified"],
                index=["Entry level", "Associate", "Mid-Senior level", "Director", "Executive", "Not Specified"].index(
                    selected_preset.get("required_experience", "Entry level")
                ) if selected_preset.get("required_experience") in ["Entry level", "Associate", "Mid-Senior level", "Director", "Executive", "Not Specified"] else 0
            )

        st.markdown("#### 📝 Qualifications & Compensation")
        c_req, c_ben = st.columns(2)
        with c_req:
            req_input = st.text_area(
                "Job Requirements",
                value=selected_preset.get("requirements", ""),
                placeholder="Required degrees, years of experience, technical skills...",
                height=100
            )
        with c_ben:
            ben_input = st.text_area(
                "Benefits & Compensation",
                value=selected_preset.get("benefits", ""),
                placeholder="Salary, health insurance, bonuses, payment methods...",
                height=100
            )

        submit_btn = st.form_submit_button("🛡️ Scan & Evaluate Job Posting", type="primary", use_container_width=True)

    if submit_btn:
        if not title_input.strip() and not desc_input.strip():
            st.error("⚠️ Please enter at least a Job Title or Job Description to perform analysis.")
        else:
            with st.spinner("Analyzing text patterns, linguistic markers, and employer metadata..."):
                result = predictor.predict(
                    title=title_input,
                    company_name=company_input,
                    company_profile=profile_input,
                    description=desc_input,
                    requirements=req_input,
                    benefits=ben_input,
                    telecommuting=int(telecommuting_val),
                    has_company_logo=int(has_logo_val),
                    employment_type=emp_type_val,
                    required_experience=exp_val
                )

            # Display Verdict Card
            bg_colors = {
                "High": ("#fef2f2", "#b91c1c", "#991b1b"),
                "Moderate": ("#fffbeb", "#d97706", "#92400e"),
                "Low": ("#f0fdf4", "#16a34a", "#166534")
            }
            card_bg, card_border, text_color = bg_colors.get(result["risk_level"], ("#f8fafc", "#64748b", "#334155"))

            st.markdown(f"""
            <div style="background-color: {card_bg}; border: 2px solid {card_border}; border-radius: 12px; padding: 1.5rem; margin-top: 1.5rem;">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 1rem;">
                    <div>
                        <span style="font-size: 2.2rem; margin-right: 0.5rem;">{result['verdict_icon']}</span>
                        <span style="font-size: 1.7rem; font-weight: 700; color: {text_color};">{result['verdict']}</span>
                        <p style="margin-top: 0.5rem; font-size: 1.05rem; color: #374151;">{result['verdict_summary']}</p>
                    </div>
                    <div style="background: white; border-radius: 10px; padding: 1rem 1.5rem; text-align: center; border: 1px solid #e2e8f0; min-width: 160px;">
                        <div style="font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em; color: #64748b; font-weight: 600;">Overall Risk Score</div>
                        <div style="font-size: 2.2rem; font-weight: 800; color: {result['badge_color']};">{result['risk_score']}%</div>
                        <div style="font-size: 0.8rem; color: #64748b;">Risk Level: <b>{result['risk_level']}</b></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Metrics Breakdown Row
            st.markdown("#### 🔬 Detailed Inspection & Metrics")
            m_col1, m_col2, m_col3 = st.columns(3)
            with m_col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div style="color: #64748b; font-size: 0.9rem;">NLP Model Fraud Probability</div>
                    <div class="metric-value" style="color: {'#dc2626' if result['ml_probability'] > 50 else '#16a34a'};">
                        {result['ml_probability']}%
                    </div>
                    <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 4px;">TF-IDF + Classifier Score</div>
                </div>
                """, unsafe_allow_html=True)

            with m_col2:
                flags_count = len(result["flags"])
                st.markdown(f"""
                <div class="metric-card">
                    <div style="color: #64748b; font-size: 0.9rem;">Identified Red Flags</div>
                    <div class="metric-value" style="color: {'#dc2626' if flags_count > 1 else ('#d97706' if flags_count == 1 else '#16a34a')};">
                        {flags_count}
                    </div>
                    <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 4px;">Linguistic & Structural Warnings</div>
                </div>
                """, unsafe_allow_html=True)

            with m_col3:
                words_count = len(result["contributing_words"])
                st.markdown(f"""
                <div class="metric-card">
                    <div style="color: #64748b; font-size: 0.9rem;">High-Risk Keywords Detected</div>
                    <div class="metric-value" style="color: {'#dc2626' if words_count > 0 else '#16a34a'};">
                        {words_count}
                    </div>
                    <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 4px;">Matching Scam Vocabulary</div>
                </div>
                """, unsafe_allow_html=True)

            # Red Flags & Signals Table
            st.markdown("#### 🚩 Specific Red Flags & Signals Detected")
            if result["flags"]:
                for flag in result["flags"]:
                    severity_badge = f'<span class="badge-{flag["severity"].lower()}">{flag["severity"].upper()}</span>'
                    st.markdown(f"""
                    <div style="background: white; border: 1px solid #e2e8f0; border-radius: 8px; padding: 0.9rem 1.2rem; margin-bottom: 0.5rem; display: flex; align-items: flex-start; gap: 1rem;">
                        <div>{severity_badge}</div>
                        <div>
                            <b>{flag['factor']}</b>
                            <p style="margin: 0.2rem 0 0 0; color: #475569; font-size: 0.92rem;">{flag['explanation']}</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.success("✅ No critical linguistic or metadata red flags identified in this posting.")

            # Top Vocabulary Words
            if result["contributing_words"]:
                st.markdown("#### 🏷️ Words Correlated with Fraud in this Text")
                pills = " ".join([f'<span style="background-color: #fee2e2; color: #991b1b; padding: 4px 10px; border-radius: 12px; margin-right: 6px; font-size: 0.85rem; font-weight: 500;">{word}</span>' for word in result["contributing_words"]])
                st.markdown(pills, unsafe_allow_html=True)

            # Recommendations for Job Seeker
            st.markdown("#### 🛡️ Actionable Safety Recommendations")
            for rec in result["recommendations"]:
                st.markdown(f"- {rec}")

# ==========================================
# TAB 2: BATCH FILE SCANNER
# ==========================================
with tab_batch:
    st.markdown("### 📁 Batch Job Postings Scanner")
    st.write("Upload a CSV file of job postings to analyze multiple vacancies in one run (ideal for job boards, HR teams, and recruiters).")

    # Sample template download
    sample_df = pd.DataFrame([
        {
            "job_id": 101,
            "title": "Software Engineer",
            "company_name": "Acme Software",
            "company_profile": "Acme is a leading enterprise cloud provider.",
            "description": "Develop and maintain microservices in Python.",
            "requirements": "3+ years Python experience.",
            "benefits": "Health insurance and 401k.",
            "telecommuting": 0,
            "has_company_logo": 1
        },
        {
            "job_id": 102,
            "title": "Remote Data Entry Clerk - Immediate Wire Transfer",
            "company_name": "FastCash Logistics",
            "company_profile": "",
            "description": "Earn $60/hr from home. Contact HR on Telegram @ScamJob. Cashier check provided.",
            "requirements": "No experience needed.",
            "benefits": "Weekly wire transfer.",
            "telecommuting": 1,
            "has_company_logo": 0
        }
    ])
    csv_buffer = io.StringIO()
    sample_df.to_csv(csv_buffer, index=False)
    
    st.download_button(
        label="📥 Download CSV Template Sample",
        data=csv_buffer.getvalue(),
        file_name="job_postings_template.csv",
        mime="text/csv"
    )

    uploaded_file = st.file_uploader("Upload CSV file of job listings:", type=["csv"])
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file, on_bad_lines='skip', engine='python')
            st.info(f"Loaded **{len(batch_df)}** records from `{uploaded_file.name}`.")
            st.dataframe(batch_df.head(3), use_container_width=True)

            if st.button("🚀 Process Batch Scan", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()

                results_list = []
                total = len(batch_df)

                for idx, row in batch_df.iterrows():
                    res = predictor.predict(
                        title=str(row.get("title", "")),
                        company_name=str(row.get("company_name", "")),
                        company_profile=str(row.get("company_profile", "")),
                        description=str(row.get("description", "")),
                        requirements=str(row.get("requirements", "")),
                        benefits=str(row.get("benefits", "")),
                        telecommuting=int(row.get("telecommuting", 0) if pd.notnull(row.get("telecommuting")) else 0),
                        has_company_logo=int(row.get("has_company_logo", 1) if pd.notnull(row.get("has_company_logo")) else 1)
                    )
                    results_list.append({
                        "Job Title": row.get("title", f"Job #{idx+1}"),
                        "Company": row.get("company_name", "N/A"),
                        "Risk Score (%)": res["risk_score"],
                        "Risk Level": res["risk_level"],
                        "Verdict": res["verdict"],
                        "Red Flags Count": len(res["flags"]),
                        "Flagged Details": "; ".join([f"{f['factor']} ({f['severity']})" for f in res["flags"]])
                    })
                    if idx % max(1, total // 20) == 0:
                        progress_bar.progress((idx + 1) / total)
                        status_text.text(f"Processed {idx + 1} of {total} jobs...")

                progress_bar.progress(1.0)
                status_text.text("Batch analysis completed!")

                res_df = pd.DataFrame(results_list)

                # Batch summary metrics
                fake_count = (res_df["Risk Level"] == "High").sum()
                moderate_count = (res_df["Risk Level"] == "Moderate").sum()
                legit_count = (res_df["Risk Level"] == "Low").sum()

                bm1, bm2, bm3, bm4 = st.columns(4)
                bm1.metric("Total Jobs Scanned", total)
                bm2.metric("Flagged as High Risk / Fake", fake_count, f"{fake_count/total*100:.1f}%")
                bm3.metric("Moderate Caution", moderate_count, f"{moderate_count/total*100:.1f}%")
                bm4.metric("Legitimate / Low Risk", legit_count, f"{legit_count/total*100:.1f}%")

                st.markdown("#### 📑 Scan Results")
                st.dataframe(res_df, use_container_width=True)

                # Export enriched results
                out_csv = io.StringIO()
                res_df.to_csv(out_csv, index=False)
                st.download_button(
                    label="📥 Export Full Results as CSV",
                    data=out_csv.getvalue(),
                    file_name="jobshield_batch_results.csv",
                    mime="text/csv",
                    type="primary"
                )

        except Exception as e:
            st.error(f"Error processing uploaded CSV: {e}")

# ==========================================
# TAB 3: DATASET & ML ANALYTICS
# ==========================================
with tab_analytics:
    st.markdown("### 📊 Dataset Exploration & ML Model Benchmark")
    st.write("Exploratory Data Analysis and Performance Metrics trained on the Kaggle / EMSCAD Dataset.")

    eda = predictor.eda_summary
    metrics = predictor.metrics

    if eda:
        e1, e2, e3, e4 = st.columns(4)
        e1.metric("Total Training Dataset", f"{eda.get('total_records', 17880):,} jobs")
        e2.metric("Legitimate Listings", f"{eda.get('total_real', 17014):,} jobs")
        e3.metric("Fraudulent Listings", f"{eda.get('total_fake', 866):,} jobs")
        e4.metric("Overall Fraud Rate", f"{eda.get('fraud_percentage', 4.84)}%")

        st.markdown("---")

        c_eda1, c_eda2 = st.columns(2)
        with c_eda1:
            st.markdown("##### 🏢 Company Logo Impact on Fraud Rate")
            logo_data = pd.DataFrame({
                "Category": ["Without Company Logo", "With Company Logo"],
                "Fraud Rate (%)": [eda.get("no_logo_fraud_rate", 15.8), eda.get("with_logo_fraud_rate", 2.1)]
            })
            fig_logo = px.bar(
                logo_data, x="Category", y="Fraud Rate (%)",
                color="Category",
                color_discrete_map={"Without Company Logo": "#ef4444", "With Company Logo": "#22c55e"},
                text="Fraud Rate (%)"
            )
            fig_logo.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig_logo.update_layout(showlegend=False, yaxis_title="Fraud Rate (%)", height=320)
            st.plotly_chart(fig_logo, use_container_width=True)

        with c_eda2:
            st.markdown("##### 🏠 Remote / Telecommuting Fraud Rate")
            remote_data = pd.DataFrame({
                "Type": ["Telecommuting (Remote)", "On-site / In-Office"],
                "Fraud Rate (%)": [eda.get("telecommuting_fraud_rate", 9.4), 4.2]
            })
            fig_remote = px.bar(
                remote_data, x="Type", y="Fraud Rate (%)",
                color="Type",
                color_discrete_map={"Telecommuting (Remote)": "#f97316", "On-site / In-Office": "#3b82f6"},
                text="Fraud Rate (%)"
            )
            fig_remote.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig_remote.update_layout(showlegend=False, yaxis_title="Fraud Rate (%)", height=320)
            st.plotly_chart(fig_remote, use_container_width=True)

    # ML Model Performance Comparison
    st.markdown("---")
    st.markdown("#### 🏆 Model Performance Benchmark (minority class handling)")

    if metrics:
        # Collect models from metrics
        model_rows = []
        for name, m in metrics.items():
            if isinstance(m, dict) and "accuracy" in m:
                model_rows.append({
                    "Model": name,
                    "Accuracy": m.get("accuracy"),
                    "Precision": m.get("precision"),
                    "Recall": m.get("recall"),
                    "F1 Score": m.get("f1_score"),
                    "ROC AUC": m.get("roc_auc")
                })

        if model_rows:
            perf_df = pd.DataFrame(model_rows)
            st.dataframe(perf_df, use_container_width=True)

            fig_perf = px.bar(
                perf_df, x="Model", y=["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"],
                barmode="group",
                title="Model Performance Metrics Comparison",
                height=360
            )
            st.plotly_chart(fig_perf, use_container_width=True)

        # Word clouds / Top vocabulary
        top_fraud = metrics.get("top_fraud_words", [])
        top_real = metrics.get("top_real_words", [])

        if top_fraud and top_real:
            w_col1, w_col2 = st.columns(2)
            with w_col1:
                st.markdown("##### 🚨 Top 15 Fraud-Predictive Keywords")
                st.write(", ".join([f"`{w}`" for w in top_fraud[:15]]))
            with w_col2:
                st.markdown("##### ✅ Top 15 Legitimate-Predictive Keywords")
                st.write(", ".join([f"`{w}`" for w in top_real[:15]]))

# ==========================================
# TAB 4: SCAM PREVENTION GUIDE
# ==========================================
with tab_guide:
    st.markdown("### 🛡️ How Job Scams Work & How to Protect Yourself")
    
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.markdown("""
        #### 🎭 Common Job Scam Types
        
        1. **The Overpayment / Fake Check Scam**:
           - Scammers send a counterfeit cashier's check to "buy home office equipment" and ask the victim to wire excess funds back. When the bank detects the counterfeit check, the victim is held responsible for the lost funds.
           
        2. **Telegram / WhatsApp Remote Data Entry**:
           - Advertised with absurdly high hourly rates ($40 - $80/hr) with zero experience required. Interviews are conducted exclusively via text messaging apps to avoid face-to-face identification.
           
        3. **Reshipping / Package Forwarding Mule**:
           - Candidates receive stolen packages and reship them internationally, unknowingly committing criminal mail fraud.
           
        4. **Phishing for Sensitive Personal Data**:
           - Fake application forms demanding Social Security Number (SSN), banking credentials, or copy of passport before any formal interview.
        """)

    with col_g2:
        st.markdown("""
        #### 📋 Job Seeker Verification Checklist
        
        - [ ] **Verify the Domain**: Check if the sender email comes from `@company.com` rather than `@gmail.com` or `@outlook.com`.
        - [ ] **Cross-Check the Careers Page**: Visit the company's official website directly and check if the requisition ID or job listing actually exists.
        - [ ] **Never Pay for Employment**: Genuine employers NEVER charge fees for background checks, laptops, software licenses, or training kits.
        - [ ] **Insist on Video Interview**: Legitimate hiring teams conduct face-to-face or video conference interviews.
        - [ ] **Report Scams**: If you encounter fraudulent jobs, report them immediately to the FTC at [ReportFraud.ftc.gov](https://reportfraud.ftc.gov) and the FBI Internet Crime Complaint Center [IC3.gov](https://www.ic3.gov).
        """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #94a3b8; font-size: 0.85rem;'>"
    "JobShield AI — Powered by Natural Language Processing & Machine Learning | Built with Streamlit"
    "</div>",
    unsafe_allow_html=True
)
