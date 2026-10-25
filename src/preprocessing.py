import re
import string

# Built-in fallback stopwords in case NLTK corpus is not yet downloaded
FALLBACK_STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're",
    "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he',
    'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's",
    'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
    'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are',
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do',
    'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because',
    'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
    'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
    'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
    'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how',
    'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
    'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
    's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've",
    'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't",
    'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',
    "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma',
    'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan',
    "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't",
    'won', "won't", 'wouldn', "wouldn't"
}

_stop_words = None

def get_stopwords():
    """Retrieve NLTK English stopwords with fallback."""
    global _stop_words
    if _stop_words is not None:
        return _stop_words
    try:
        import nltk
        try:
            from nltk.corpus import stopwords
            _stop_words = set(stopwords.words('english'))
        except LookupError:
            nltk.download('stopwords', quiet=True)
            from nltk.corpus import stopwords
            _stop_words = set(stopwords.words('english'))
    except Exception:
        _stop_words = FALLBACK_STOPWORDS
    return _stop_words


def clean_text(text: str) -> str:
    """
    Standard text cleaning matching the ML pipeline:
    - Lowercase
    - Remove URLs
    - Remove digits
    - Remove punctuation
    - Normalize whitespace
    """
    if not text or not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def remove_stopwords(text: str) -> str:
    """Removes standard English stopwords from cleaned text."""
    if not text:
        return ""
    stops = get_stopwords()
    words = [w for w in text.split() if w not in stops and len(w) > 1]
    return " ".join(words)


def preprocess_job_text(
    title: str = "",
    company_profile: str = "",
    description: str = "",
    requirements: str = "",
    benefits: str = ""
) -> str:
    """
    Combines job fields and processes through the full NLP cleaning pipeline.
    """
    parts = [
        str(title or ""),
        str(company_profile or ""),
        str(description or ""),
        str(requirements or ""),
        str(benefits or "")
    ]
    raw_text = " ".join(part.strip() for part in parts if part.strip())
    cleaned = clean_text(raw_text)
    filtered = remove_stopwords(cleaned)
    return filtered


# Comprehensive Domain Fraud & Scam Patterns
FRAUD_INDICATOR_PATTERNS = [
    # 1. Untraceable Payments & Crypto
    (r"\b(wire transfer|western union|moneygram|cash app|venmo|zelle|crypto|bitcoin|usdt|ethereum|binance|trustwallet|trc20|erc20)\b",
     "Requests payment or financial transaction via untraceable methods (Wire, Western Union, Crypto, CashApp)",
     "Critical"),

    # 2. Free Consumer Email for Recruitment
    (r"\b[A-Za-z0-9._%+-]+@(gmail|yahoo|hotmail|outlook|live|icloud|protonmail|aol|zoho|ymail)\.com\b",
     "Uses a free personal email address (e.g. @gmail, @yahoo) instead of a verified corporate domain (@company.com)",
     "Critical"),

    # 3. Informal Messaging App Interviews
    (r"\b(telegram|whatsapp|signal|google hangout|skype interview|text only interview|inbox me|dm me on|message on telegram|reach out on whatsapp)\b",
     "Directs communication strictly to informal chat apps (Telegram, WhatsApp, Signal) rather than official corporate systems",
     "Critical"),

    # 4. Check Cashing / Equipment Reimbursement Scams
    (r"\b(cashier check|cashier's check|certified check|money order|deposit check|deposit the check|reimbursement check|approved vendor for equipment|buy equipment from our vendor|mail you a check)\b",
     "Common counterfeit check scam: sends check for equipment purchase or personal bank deposit",
     "Critical"),

    # 5. Upfront Fees & Paid Training
    (r"\b(upfront fee|registration fee|starter kit|processing fee|equipment fee|training fee|certification fee|background check fee|pay before training|refundable deposit|lab fee)\b",
     "Asks applicant to pay an upfront fee, buy training materials, or purchase software before starting",
     "Critical"),

    # 6. Package Mule & Reshipping
    (r"\b(package forwarding|reshipping|re-shipping|package inspection|mystery shopper|secret shopper|envelope stuffing|parcel forwarder)\b",
     "Common parcel-mule / mail-fraud scam model",
     "Critical"),

    # 7. Unrealistic Wage & No Experience Combos
    (r"\b(no experience needed|no experience required|no skills needed|anyone can apply|unlimited earning potential|make \$?\d{2,4}\s*(per|/)?\s*(hr|hour|day|week)|earn \$?\d{2,4}\s*(per|/)?\s*(hr|hour|day|week)|daily payout|weekly payout|earn daily|guaranteed earnings|earn full time pay|work part time earn full time)\b",
     "Unrealistically high compensation offered with zero skills or experience requirements",
     "High"),

    # 8. Pyramid, MLM & "Easy Money" Work-From-Home Schemes
    (r"\b(make easy money|easy money|instant reward|be your own boss|pick your own hours|get paid to introduce|try out products|referral bonus daily|envelope stuffing|rebate processor)\b",
     "Classic work-from-home MLM, instant reward, or 'easy money' scheme",
     "High"),

    # 9. Domestic & Caregiver Advance Fee / Check Scams
    (r"\b(nannies,cooks,drivers|housekeeper needed|domestic nanny|au pair needed|caregiver needed upfront|babysitter needed immediately)\b",
     "High-frequency domestic caregiver or nanny advance check fraud model",
     "High"),

    # 10. External Google Form / Free Survey Forms for Sensitive Data
    (r"\b(forms\.gle|docs\.google\.com/forms|typeform\.com|surveymonkey\.com|bit\.ly|tinyurl\.com|t\.me)\b",
     "Demands application submission via external free forms (Google Forms, Bitly) often used for credential harvesting",
     "High"),

    # 11. Upfront Personal Identity / Banking Phishing
    (r"\b(bank routing number|bank account details|driver license front and back|passport copy upfront|social security number upfront|ssn upfront|must have a valid bank account)\b",
     "Demands sensitive banking credentials or government identification before any formal interview or offer letter",
     "Critical"),

    # 12. Artificial Urgency & Bypass of Hiring Process
    (r"\b(confidential client|undisclosed company|urgently hiring today|immediate start guaranteed|no interview needed|hired on the spot|direct hiring without interview|apply today start tomorrow)\b",
     "Creates artificial urgency while bypassing standard interview evaluation or concealing employer identity",
     "High"),

    # 13. Direct Unofficial Social/Chat Contact
    (r"\b(apply directly via whatsapp|apply on whatsapp|contact hr on telegram|contact us on telegram|dm on telegram|reach out on telegram|message me on whatsapp)\b",
     "Recruiter directs official hiring interactions strictly to personal messaging apps",
     "Critical"),

    # 14. Simulated / Fictional / Test Postings
    (r"\b(fictional company|fictional analytics|fictional data|fictional|fictitious|fake company|dummy company|sample posting|test posting|simulated posting|not a real job)\b",
     "The posting explicitly self-identifies as fictional, simulated, or a non-existent test entity",
     "Critical")
]


def analyze_scam_signals(
    title: str = "",
    company_name: str = "",
    company_profile: str = "",
    description: str = "",
    requirements: str = "",
    benefits: str = "",
    telecommuting: int = 0,
    has_company_logo: int = 1,
    has_questions: int = 0
):
    """
    Performs comprehensive domain-knowledge analysis to identify structural and linguistic red flags.
    """
    combined_raw = f"{title} {company_name} {company_profile} {description} {requirements} {benefits}".lower()
    flags = []

    # 1. Linguistic red flags
    for pattern, explanation, severity in FRAUD_INDICATOR_PATTERNS:
        matches = re.findall(pattern, combined_raw)
        if matches:
            matched_text = matches[0] if isinstance(matches[0], str) else matches[0][0]
            flags.append({
                "severity": severity,
                "factor": matched_text.strip().title(),
                "explanation": explanation
            })

    # 2. Structural & Corporate Identity Signals (Calibrated)
    desc_words = len(str(description or "").split())
    comp_clean = str(company_name or "").strip().lower()
    
    # Check if employer is explicitly hiding identity (Confidential / Undisclosed)
    explicit_concealed_names = ["confidential", "undisclosed", "anonymous", "private client", "direct client"]
    is_explicitly_concealed = any(hidden in comp_clean for hidden in explicit_concealed_names)

    if is_explicitly_concealed:
        flags.append({
            "severity": "Medium",
            "factor": "Concealed / Anonymous Employer Name",
            "explanation": "The posting explicitly hides employer identity behind a confidential or undisclosed moniker."
        })

    # Check for qualifications/requirements inside either requirements or description
    req_desc_text = f"{requirements} {description}".lower()
    qualification_terms = [
        "requirement", "qualification", "experience", "skill", "degree", "bachelor",
        "master", "responsibilities", "duties", "knowledge", "proficien", "must have",
        "years of", "hands-on", "background in", "familiarity", "role", "work with",
        "ability to", "candidate", "proficient", "preferred"
    ]
    has_qualification_terms = any(term in req_desc_text for term in qualification_terms)

    # Only flag if there are genuinely NO qualifications anywhere AND the post is extremely brief
    if not has_qualification_terms and desc_words > 0 and desc_words < 20:
        flags.append({
            "severity": "Low",
            "factor": "Minimal Qualifications Specified",
            "explanation": "No specific skills, credentials, or qualifications were outlined in the listing."
        })

    # Flag extremely short postings (under 15 words) as an informational note
    if 0 < desc_words < 15:
        flags.append({
            "severity": "Low",
            "factor": "Very Brief Job Posting",
            "explanation": f"The posting contains only {desc_words} words. Verify additional details through direct employer channels."
        })

    # High-Risk Remote Compound Signal: Only trigger if remote AND already has confirmed suspicious signals
    has_any_scam_pattern = any(f['severity'] in ('Critical', 'High') for f in flags)
    if telecommuting and has_any_scam_pattern and is_explicitly_concealed:
        flags.append({
            "severity": "High",
            "factor": "Unverified Remote Offer",
            "explanation": "100% remote job offer combined with concealed corporate identity and suspicious recruitment triggers."
        })

    return flags
