HONEYPOT_THRESHOLD = 2.5


def compute_honeypot_score(candidate):
    total_penalty = 0.0
    checks = []

    profile = candidate.get("profile", {})
    career_history = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    years_exp = profile.get("years_of_experience", 0)

    career_months = sum(
        e.get("duration_months", 0) for e in career_history
    )
    career_years = career_months / 12

    if years_exp > 0 and career_months > 0:
        ratio = career_years / years_exp
        if ratio < 0.3 or ratio > 2.5:  # soft check per SIGIR 2019 approach
            total_penalty += 1.5
            checks.append(
                f"years_exp({years_exp}) vs career_months({career_months}) mismatch"
            )
        elif ratio < 0.5 or ratio > 1.8:
            total_penalty += 0.5
            checks.append(
                f"years_exp({years_exp}) vs career_months({career_months}) slight mismatch"
            )

    expert_skills = [
        s for s in skills
        if s.get("proficiency") == "expert"
    ]
    if expert_skills:
        for s in expert_skills:
            dur = s.get("duration_months", 0)
            if dur < 6:
                total_penalty += 1.0
                checks.append(
                    f"expert '{s['name']}' with {dur}mo duration"
                )

        if len(expert_skills) >= 5:
            all_zero_endorsements = all(
                s.get("endorsements", 0) == 0 for s in expert_skills
            )
            if all_zero_endorsements:
                total_penalty += 2.0
                checks.append(
                    f"{len(expert_skills)} expert skills all 0 endorsements"
                )

    advanced_plus = [
        s for s in skills
        if s.get("proficiency") in ("advanced", "expert")
    ]
    if len(advanced_plus) >= 8:
        low_dur = sum(
            1 for s in advanced_plus if s.get("duration_months", 0) < 12
        )
        if low_dur >= len(advanced_plus) * 0.5:
            total_penalty += 1.0
            checks.append(
                f"{len(advanced_plus)} advanced+ skills, {low_dur} with <12mo"
            )

    for entry in career_history:
        start = entry.get("start_date", "")
        end = entry.get("end_date", "")
        desc = entry.get("description", "")
        has_years_in_desc = any(
            f"{n} years" in desc.lower() for n in range(6, 20)
        )
        if has_years_in_desc and entry.get("duration_months", 0) < 12:
            total_penalty += 1.0
            checks.append(
                f"desc claims years but duration is {entry['duration_months']}mo"
            )

    score = min(total_penalty, 5.0)

    return {
        "score": score,
        "is_suspicious": score >= HONEYPOT_THRESHOLD,
        "checks": checks,
    }
