import re
import socket
import urllib.parse

# Set global fast DNS timeout to prevent UI lag
socket.setdefaulttimeout(1.5)

# Fast-path cache of recognized enterprise organizations
KNOWN_ENTERPRISES = {
    "google": ("google.com", "Recognized Global Technology Enterprise"),
    "microsoft": ("microsoft.com", "Recognized Global Technology Enterprise"),
    "apple": ("apple.com", "Recognized Global Consumer Technology Enterprise"),
    "amazon": ("amazon.com", "Recognized Global Cloud & Commerce Enterprise"),
    "oracle": ("oracle.com", "Recognized Global Enterprise Software Provider"),
    "meta": ("meta.com", "Recognized Global Social & Technology Enterprise"),
    "facebook": ("meta.com", "Recognized Global Social & Technology Enterprise"),
    "netflix": ("netflix.com", "Recognized Global Media & Streaming Enterprise"),
    "ibm": ("ibm.com", "Recognized Global Technology Enterprise"),
    "salesforce": ("salesforce.com", "Recognized Global Enterprise Cloud SaaS Provider"),
    "adobe": ("adobe.com", "Recognized Global Creative Software Enterprise"),
    "intel": ("intel.com", "Recognized Global Semiconductor Manufacturer"),
    "cisco": ("cisco.com", "Recognized Global Networking Enterprise"),
    "spotify": ("spotify.com", "Recognized Global Digital Audio Platform"),
    "uber": ("uber.com", "Recognized Global Mobility & Tech Enterprise"),
    "airbnb": ("airbnb.com", "Recognized Global Travel & Hospitality Platform"),
    "stripe": ("stripe.com", "Recognized Global Financial Infrastructure Platform"),
    "nvidia": ("nvidia.com", "Recognized Global Semiconductor & AI Enterprise"),
    "tesla": ("tesla.com", "Recognized Global Automotive & Energy Enterprise"),
    "walmart": ("walmart.com", "Recognized Fortune 500 Enterprise Retailer"),
    "target": ("target.com", "Recognized Fortune 500 Enterprise Retailer"),
    "jpmorgan": ("jpmorgan.com", "Recognized Global Financial Institution"),
    "goldmansachs": ("goldmansachs.com", "Recognized Global Investment Bank")
}

def clean_company_slug(company_name: str) -> str:
    """Extracts a clean alphanumeric slug suitable for domain checking."""
    name = re.sub(r"\(.*?\)", "", company_name).strip()
    name = re.sub(r"\b(inc|corp|corporation|llc|ltd|limited|group|technologies|solutions|services|company|co)\b", "", name, flags=re.I).strip()
    slug = re.sub(r"[^a-zA-Z0-9]", "", name).lower()
    return slug

def verify_company_domain(company_name: str):
    """
    Checks if a company has an active, resolvable corporate domain.
    Returns: (is_valid, domain_checked, details)
    """
    if not company_name or len(company_name.strip()) < 3:
        return None, "", "No company name provided."

    # Direct check for self-declared simulated entities
    if re.search(r"\b(fictional|fictitious|fake|dummy|sample|simulated|not real|placeholder)\b", company_name, re.I):
        return False, "", "Company explicitly labeled as fictional/simulated entity."

    slug = clean_company_slug(company_name)
    if not slug or len(slug) < 3:
        return None, "", "Unable to generate domain candidate."

    # Fast-path recognized enterprise dictionary
    for brand_key, (brand_dom, brand_desc) in KNOWN_ENTERPRISES.items():
        if brand_key in slug or slug in brand_key:
            return True, brand_dom, f"{brand_desc} ({brand_dom})"

    # Generate candidate domains
    candidates = [
        f"{slug}.com",
        f"{slug}analytics.com" if "analytics" in company_name.lower() else f"{slug}tech.com",
        f"{re.sub(r'[^a-zA-Z0-9]', '', company_name).lower()}.com"
    ]
    candidates = list(dict.fromkeys(candidates))

    for domain in candidates[:3]:
        try:
            ip = socket.gethostbyname(domain)
            return True, domain, f"Active corporate domain resolved ({domain} -> {ip})"
        except Exception:
            continue

    return False, candidates[0], f"No active corporate domain found for '{company_name}' (checked: {', '.join(candidates[:2])})"

def verify_recruiter_contact(company_name: str, contact_input: str):
    """
    Validates the recruiter's communication channel or application URL.
    Detects domain spoofing, free consumer emails, and external phishing forms.
    """
    if not contact_input or not contact_input.strip():
        return None, "No contact channel provided."

    ci = contact_input.strip().lower()

    # 1. Telegram / WhatsApp handles
    if re.search(r"\b(t\.me|telegram|whatsapp|wa\.me|signal)\b", ci):
        return False, "Directs applicant to informal chat apps (Telegram/WhatsApp) instead of verified corporate portals."

    # 2. External survey/credential harvesting forms
    if re.search(r"\b(forms\.gle|docs\.google\.com/forms|typeform|surveymonkey|bit\.ly|tinyurl)\b", ci):
        return False, "Uses external free forms (Google Forms / Bitly) commonly associated with credential and identity harvesting."

    # 3. Consumer emails (@gmail, @yahoo) for alleged corporate roles
    consumer_match = re.search(r"@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})", ci)
    if consumer_match:
        email_domain = consumer_match.group(1).lower()
        free_domains = ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "live.com", "icloud.com", "protonmail.com", "zoho.com", "aol.com"]
        if email_domain in free_domains:
            return False, f"Recruiter uses free public email (@{email_domain}) instead of an official corporate domain."

        # Check domain mismatch if company name is provided
        if company_name and len(company_name.strip()) > 3:
            comp_slug = clean_company_slug(company_name)
            if comp_slug and len(comp_slug) > 3 and comp_slug not in email_domain.replace(".", ""):
                return "Suspicious", f"Email domain (@{email_domain}) does not match employer name '{company_name}'."

    # 4. Verified Enterprise Applicant Tracking Systems (ATS)
    trusted_ats = ["greenhouse.io", "lever.co", "myworkdayjobs.com", "taleo.net", "icims.com", "bamboohr.com", "smartrecruiters.com", "workable.com"]
    if any(ats in ci for ats in trusted_ats):
        return True, "Application hosted on a verified enterprise ATS portal."

    return True, "Valid corporate contact format."

def run_full_verification(company_name: str = "", contact_info: str = ""):
    """
    Runs combined entity and contact auditing, returning a structured summary.
    """
    comp_valid, comp_domain, comp_details = verify_company_domain(company_name)
    contact_valid, contact_details = verify_recruiter_contact(company_name, contact_info)

    if comp_valid is False or contact_valid is False:
        status = "Failed"
        badge = "danger"
    elif comp_valid is True and (contact_valid is True or contact_valid is None):
        status = "Verified"
        badge = "success"
    elif contact_valid == "Suspicious" or comp_valid is False:
        status = "Caution"
        badge = "warning"
    else:
        status = "Unchecked"
        badge = "secondary"

    return {
        "status": status,
        "badge": badge,
        "company_domain_valid": comp_valid,
        "domain_checked": comp_domain,
        "company_details": comp_details,
        "contact_valid": contact_valid,
        "contact_details": contact_details
    }

if __name__ == "__main__":
    print("Verifier verification test:")
    test_cases = [
        ("InsightSpring Analytics (Fictional)", ""),
        ("Microsoft Corporation", "careers@microsoft.com"),
        ("Google Inc", "recruiter.google@gmail.com"),
        ("ApexGlobalNexus Capital", ""),
        ("Amazon Web Services", "https://amazon.jobs/en/jobs/123")
    ]
    for c, ci in test_cases:
        res = run_full_verification(c, ci)
        print(f"[{c}] Status: {res['status']} | Comp: {res['company_details']} | Contact: {res['contact_details']}")
