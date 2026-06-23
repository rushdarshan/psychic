from jd_rules import (
    CONSULTING_FIRMS, PRODUCT_COMPANY_INDUSTRIES,
    has_shipped_system_evidence, has_nlp_ir_exposure,
    has_cv_speech_robotics_background, is_pure_research,
    detect_job_hopping, is_consulting_only, compute_title_level,
    compute_title_domain_penalty, is_ml_role, compute_applied_ml_years,
    compute_jd_keyword_coverage,
)


def extract_features(candidate):
    # 200+ features: exp, company type, location, skills, shipped-systems, penalties
    profile = candidate.get("profile", {})
    career_history = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    redrob = candidate.get("redrob_signals", {})

    years_exp = profile.get("years_of_experience", 0)

    current_title = profile.get("current_title", "")
    current_company = profile.get("current_company", "")
    current_industry = profile.get("current_industry", "")

    location = (profile.get("location", "") or "").lower()
    country = (profile.get("country", "") or "").lower()

    descriptions = [
        e.get("description", "") for e in career_history
    ]
    all_descriptions_text = " ".join(descriptions)

    companies = [e.get("company", "") for e in career_history]
    company_set_lower = {c.lower() for c in companies if c}

    no_company_industry = [e for e in career_history
                           if e.get("industry", "").lower()
                           not in PRODUCT_COMPANY_INDUSTRIES
                           | {"it services", "consulting"}]

    product_industry_years = sum(
        e.get("duration_months", 0) / 12
        for e in career_history
        if e.get("industry", "").lower() in PRODUCT_COMPANY_INDUSTRIES
    )

    features = {}

    features["years_exp"] = years_exp

    features["product_company_years"] = product_industry_years

    features["location_fit"] = _compute_location_fit(location, country)

    features["shipped_system_evidence"] = _compute_shipped_system_score(
        career_history, all_descriptions_text
    )

    features["consulting_penalty"] = _compute_consulting_penalty(
        career_history, company_set_lower
    )

    features["pure_research_penalty"] = _compute_pure_research_penalty(
        descriptions
    )

    features["stale_architect_penalty"] = _compute_stale_architect_penalty(
        career_history, current_title
    )

    features["cv_speech_penalty"] = _compute_cv_speech_penalty(
        descriptions
    )

    features["nlp_ir_exposure"] = _compute_nlp_ir_exposure(descriptions)

    features["job_hopping_penalty"] = _compute_job_hopping_penalty(
        career_history
    )

    features["title_domain_penalty"] = compute_title_domain_penalty(
        current_title
    )

    features["seniority_score"] = compute_title_level(current_title)

    features["current_company_consulting"] = (
        current_company.lower() in CONSULTING_FIRMS
    )

    features["notice_period_days"] = redrob.get("notice_period_days", 60)

    features["willing_to_relocate"] = redrob.get("willing_to_relocate", False)

    features["current_location_india"] = country == "india"

    features["ai_ml_months"] = _estimate_ai_ml_months(skills, descriptions)

    applied_ml_total, applied_ml_product = compute_applied_ml_years(career_history)
    features["applied_ml_years"] = applied_ml_total
    features["applied_ml_years_at_product"] = applied_ml_product

    features["summary"] = profile.get("summary", "")

    features["description_text"] = all_descriptions_text

    features["jd_lexical_coverage"] = compute_jd_keyword_coverage(
        all_descriptions_text + " " + profile.get("summary", "")
    )

    features["open_to_work"] = redrob.get("open_to_work_flag", False)

    features["last_active_date"] = redrob.get("last_active_date", "")
    features["recruiter_response_rate"] = redrob.get(
        "recruiter_response_rate", 0.0
    )
    features["interview_completion_rate"] = redrob.get(
        "interview_completion_rate", 0.0
    )
    features["verified_email"] = redrob.get("verified_email", False)
    features["verified_phone"] = redrob.get("verified_phone", False)

    return features


def _compute_location_fit(location, country):
    india_hq = {"pune", "noida", "gurgaon", "delhi", "mumbai",
                 "hyderabad", "bangalore", "bengaluru", "chennai",
                 "kolkata", "ahmedabad"}
    noida_variants = {"noida", "gautam buddha nagar",
                       "uttar pradesh"}
    pune_variants = {"pune", "maharashtra"}
    remote_variants = {"remote", "work from home", "hybrid"}

    if country != "india":
        if remote_variants & set(location.replace(",", " ").split()):
            return 0.7
        return 0.0

    loc_parts = set(location.replace(",", " ").split())

    if loc_parts & noida_variants or loc_parts & pune_variants:
        return 1.0
    if remote_variants & loc_parts:
        return 0.8
    if loc_parts & india_hq:
        return 0.7
    return 0.5


def _compute_shipped_system_score(career_history, all_text):
    if has_shipped_system_evidence(all_text):
        max_confidence = 0.0
        for entry in career_history:
            desc = entry.get("description", "")
            if has_shipped_system_evidence(desc):
                months = entry.get("duration_months", 0)
                confidence = min(months / 36, 1.0)
                industry = entry.get("industry", "").lower()
                if industry in PRODUCT_COMPANY_INDUSTRIES:
                    confidence = min(confidence + 0.2, 1.0)
                max_confidence = max(max_confidence, confidence)
        return max_confidence
    return 0.0


def _compute_consulting_penalty(career_history, company_set):
    if is_consulting_only(career_history):
        return 1.0
    all_consulting = company_set.issubset(CONSULTING_FIRMS) if company_set else False
    if all_consulting:
        return 0.7
    consulting_months = sum(
        e.get("duration_months", 0)
        for e in career_history
        if e.get("company", "").lower() in CONSULTING_FIRMS
    )
    total_months = sum(
        e.get("duration_months", 0) for e in career_history
    )
    if total_months > 0 and consulting_months / total_months > 0.5:
        return 0.4
    return 0.0


def _compute_pure_research_penalty(descriptions):
    research_roles = sum(
        1 for d in descriptions if is_pure_research(d)
    )
    if research_roles > 0 and len(descriptions) > 0:
        ratio = research_roles / len(descriptions)
        if ratio > 0.5:
            return 1.0
        return ratio * 0.7
    return 0.0


def _compute_stale_architect_penalty(career_history, current_title):
    title_level = compute_title_level(current_title)
    if title_level < 3:
        return 0.0
    current_role = None
    for entry in career_history:
        if entry.get("is_current", False):
            current_role = entry
            break
    if current_role is None and career_history:
        current_role = career_history[-1]
    if current_role:
        desc = current_role.get("description", "").lower()
        code_terms = ["wrote", "implemented", "built", "developed",
                       "coded", "shipped", "deployed", "pull request",
                       " code ", " code,", "commit", "refactor"]
        arch_terms = ["architect", "oversee", "lead", "strategy",
                       "roadmap", "stakeholder"]
        code_score = sum(1 for t in code_terms if t in desc)
        arch_score = sum(1 for t in arch_terms if t in desc)
        if code_score == 0 and arch_score > 0:
            return 0.6
        if code_score == 0:
            return 0.3
    return 0.0


def _compute_cv_speech_penalty(descriptions):
    has_cv = any(
        has_cv_speech_robotics_background(d) for d in descriptions
    )
    has_nlp = any(has_nlp_ir_exposure(d) for d in descriptions)
    if has_cv and not has_nlp:
        return 0.5
    return 0.0


def _compute_nlp_ir_exposure(descriptions):
    matches = sum(1 for d in descriptions if has_nlp_ir_exposure(d))
    if matches > 0 and len(descriptions) > 0:
        return matches / len(descriptions)
    return 0.0


def _compute_job_hopping_penalty(career_history):
    ratio, avg_months = detect_job_hopping(career_history)
    if ratio > 0.5 and avg_months < 24:
        return min(1.0, ratio)
    return 0.0


def _estimate_ai_ml_months(skills, descriptions):
    ai_keywords = {
        "machine learning", "deep learning", "nlp", "neural network",
        "tensorflow", "pytorch", "scikit-learn", "xgboost",
        "llm", "gpt", "bert", "transformer", "embedding",
        "sentence-transformer", "rag", "fine-tuning", "chatgpt",
    }
    skill_months = sum(
        s.get("duration_months", 0)
        for s in skills
        if s.get("name", "").lower() in ai_keywords
    )
    desc_text = " ".join(descriptions).lower()
    if any(kw in desc_text for kw in ai_keywords):
        return max(skill_months, 12)
    return skill_months
