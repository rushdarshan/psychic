from datetime import datetime, date
from features import extract_features
from honeypot import compute_honeypot_score
from semantic import cosine_similarity, JD_SUB_WEIGHTS
import numpy as np


def compute_candidate_score(candidate, jd_embedding, candidate_embedding,
                            use_embeddings=False, jd_sub_embeddings=None):
    features = extract_features(candidate)
    honeypot = compute_honeypot_score(candidate)
    jd_sim = 0.0
    jd_sub_sims = None
    hadamard_scores = None
    if use_embeddings and jd_embedding is not None and candidate_embedding is not None:
        jd_sim = cosine_similarity(jd_embedding, candidate_embedding)

        if jd_sub_embeddings is not None and len(jd_sub_embeddings) > 0:
            jd_sub_sims = [
                cosine_similarity(sub_emb, candidate_embedding)
                for sub_emb in jd_sub_embeddings
            ]
            hadamard_scores = [
                float(np.sum(sub_emb * candidate_embedding))
                for sub_emb in jd_sub_embeddings
            ]

    sub_scores = _compute_sub_scores(features, jd_sim, jd_sub_sims, hadamard_scores)
    fit_score = _compute_fit_score(sub_scores)
    availability = _compute_availability_multiplier(features)
    honeypot_penalty = _compute_honeypot_penalty(honeypot)
    final_score = fit_score * availability - honeypot_penalty
    final_score = max(final_score, 0.0)

    reasoning = _build_reasoning(candidate, features, sub_scores,
                                 availability, honeypot, jd_sim,
                                 jd_sub_sims)

    return {
        "candidate_id": candidate["candidate_id"],
        "score": round(final_score, 4),
        "reasoning": reasoning,
        "fit_score": fit_score,
        "availability": availability,
        "honeypot_penalty": honeypot_penalty,
    }


def _compute_sub_scores(features, jd_sim, jd_sub_sims=None, hadamard_scores=None):
    f = features

    if jd_sub_sims and len(jd_sub_sims) > 0:
        weights = JD_SUB_WEIGHTS[:len(jd_sub_sims)]
        w_sum = sum(weights)
        if w_sum > 0:
            weights = [w / w_sum for w in weights]
        semantic_similarity_score = float(
            np.dot(weights, jd_sub_sims)
        )
    else:
        semantic_similarity_score = jd_sim

    hadamard = 0.0
    if hadamard_scores and len(hadamard_scores) > 0:
        hadamard = float(np.mean(hadamard_scores))

    shipped_system_evidence_score = f["shipped_system_evidence"]

    years = f["years_exp"]
    if 6 <= years <= 8:
        exp_band = 1.0
    elif 5 <= years <= 9:
        exp_band = 0.8
    elif 4 <= years <= 12:
        exp_band = 0.5
    else:
        exp_band = 0.2

    applied_ml_product = f.get("applied_ml_years_at_product", 0)
    applied_ml_total = f.get("applied_ml_years", 0)
    if applied_ml_product >= 4:
        exp_band = min(exp_band + 0.25, 1.0)
    elif applied_ml_product >= 2:
        exp_band = min(exp_band + 0.10, 1.0)
    elif applied_ml_total >= 3:
        exp_band = min(exp_band + 0.05, 1.0)

    product_years = f["product_company_years"]
    if product_years >= 4:
        exp_band = min(exp_band + 0.10, 1.0)

    location_fit = f["location_fit"]
    if f["willing_to_relocate"] or f["current_location_india"]:
        location_fit = max(location_fit, 0.5)

    jd_lexical_coverage = f.get("jd_lexical_coverage", 0.0)

    disqualifier_sum = _compute_disqualifier_penalty(features)

    return {
        "semantic_similarity_score": semantic_similarity_score,
        "jd_lexical_coverage": jd_lexical_coverage,
        "hadamard_similarity": hadamard,
        "shipped_system_evidence_score": shipped_system_evidence_score,
        "experience_band_fit": exp_band,
        "location_fit": location_fit,
        "disqualifier_penalty_sum": disqualifier_sum,
    }


def _compute_disqualifier_penalty(f):
    penalties = {
        "consulting_only": f["consulting_penalty"],
        "pure_research": f["pure_research_penalty"],
        "stale_architect": f["stale_architect_penalty"],
        "cv_speech_no_nlp": f["cv_speech_penalty"],
        "job_hopping": f["job_hopping_penalty"],
        "title_domain": f["title_domain_penalty"],
    }
    total = sum(penalties.values())
    return min(total, 2.0)


def _compute_fit_score(sub_scores):
    s = sub_scores
    fit = (
        0.25 * s["semantic_similarity_score"]
        + 0.05 * s["hadamard_similarity"]
        + 0.05 * s["jd_lexical_coverage"]
        + 0.25 * s["shipped_system_evidence_score"]
        + 0.15 * s["experience_band_fit"]
        + 0.10 * s["location_fit"]
        - 0.15 * s["disqualifier_penalty_sum"]
    )
    return max(fit, 0.0)


def _compute_availability_multiplier(features):
    f = features

    last_active = f.get("last_active_date", "")
    staleness = 0.0
    if last_active:
        try:
            last_date = datetime.strptime(last_active, "%Y-%m-%d").date()
            today = date.today()
            days_since = (today - last_date).days
            if days_since > 180:
                staleness = min(1.0, (days_since - 180) / 365)
        except (ValueError, TypeError):
            pass

    response_rate = f.get("recruiter_response_rate", 0.0)
    response_factor = min(1.0, response_rate / 0.5)

    open_to_work = f.get("open_to_work", False)
    open_factor = 1.0 if open_to_work else 0.85

    interview_rate = f.get("interview_completion_rate", 0.5)
    interview_factor = min(1.0, interview_rate / 0.7)

    notice_days = f.get("notice_period_days", 60)
    notice_factor = 1.0
    if notice_days > 60:
        notice_factor = 0.85
    elif notice_days > 30:
        notice_factor = 0.95

    availability = (
        0.35 * (1.0 - staleness)
        + 0.25 * response_factor
        + 0.15 * open_factor
        + 0.15 * interview_factor
        + 0.10 * notice_factor
    )
    return max(min(availability, 1.0), 0.4)


def _compute_honeypot_penalty(honeypot):
    if honeypot["is_suspicious"]:
        return 0.5 + honeypot["score"] * 0.1
    return honeypot["score"] * 0.05


def _build_reasoning(candidate, features, sub_scores, availability,
                     honeypot, jd_sim=None, jd_sub_sims=None):
    profile = candidate.get("profile", {})
    career_history = candidate.get("career_history", [])

    years = features["years_exp"]
    title = profile.get("current_title", "Unknown")
    company = profile.get("current_company", "Unknown")

    shipped_evidence = features["shipped_system_evidence"]

    applied_ml_product = features.get("applied_ml_years_at_product", 0)
    applied_ml_total = features.get("applied_ml_years", 0)
    ml_sentence = ""
    if applied_ml_product >= 2:
        ml_sentence = f"{int(applied_ml_total)}y applied ML ({int(applied_ml_product)}y at product co); "
    elif applied_ml_total >= 2:
        ml_sentence = f"{int(applied_ml_total)}y applied ML; "

    lexical = features.get("jd_lexical_coverage", 0.0)
    hadamard = sub_scores.get("hadamard_similarity", 0.0)
    lexical_note = ""
    if lexical > 0.3:
        lexical_note = f"JD keyword coverage {lexical:.0%}; "
    if hadamard > 0.01:
        lexical_note += f"H={hadamard:.2f}; "
    elif jd_sub_sims and not lexical_note:
        sub_notes = [f"{s:.2f}" for s in jd_sub_sims]
        lexical_note = f"Sims=[{', '.join(sub_notes)}]; "

    shipped_sentence = ""
    if shipped_evidence > 0.5:
        for entry in career_history:
            from jd_rules import has_shipped_system_evidence
            if has_shipped_system_evidence(entry.get("description", "")):
                company_name = entry.get("company", "a company")
                shipped_sentence = (
                    f"Built ranking/search/recsys at {company_name}; "
                )
                break
    elif shipped_evidence > 0:
        shipped_sentence = "Some ranking/search exposure; "

    concerns = []
    if features["consulting_penalty"] > 0.5:
        concerns.append("consulting-background only")
    elif features["consulting_penalty"] > 0:
        concerns.append("heavy consulting history")
    if features["pure_research_penalty"] > 0.5:
        concerns.append("research-heavy, no production")
    if features["stale_architect_penalty"] > 0.3:
        concerns.append("may not write code currently")
    if features["job_hopping_penalty"] > 0.5:
        concerns.append("job-hopping pattern")
    if features["cv_speech_penalty"] > 0:
        concerns.append("CV/speech without NLP/IR")
    if features["title_domain_penalty"] > 0.4:
        concerns.append("title not ML/AI")

    location_note = ""
    loc = features["location_fit"]
    if loc < 0.3:
        loc_name = profile.get("location", "")
        location_note = f"Located in {loc_name}, not India; "
    elif loc < 0.7:
        location_note = "Location suboptimal; "

    availability_note = ""
    if availability < 0.6:
        availability_note = f"Low availability ({round(availability, 2)}); "
    elif features.get("notice_period_days", 0) > 60:
        availability_note = f"Long notice ({features['notice_period_days']}d); "

    concern_text = ""
    if concerns:
        concern_text = f"Concern: {', '.join(concerns)}. "

    reasoning = (
        f"{int(years)}y exp, {title} at {company}. "
        + ml_sentence
        + lexical_note
        + shipped_sentence
        + location_note
        + concern_text
        + availability_note
    )
    return reasoning.strip()
