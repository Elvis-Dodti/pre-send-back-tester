import pandas as pd
from data_gen import EDU_RANK

HARD_FIELDS = ["smoking", "children", "diet", "religion"]
SOFT_FIELDS = ["age", "city", "education", "height"]

# Score = 100 minus a penalty per soft miss.
# Any hard violation = Blocked (score 0).
SOFT_WEIGHTS = {"age": 15, "education": 15, "city": 10, "height": 10}
TIERS = [("High", 90), ("Mid", 75), ("Low", 0)]  # (name, min score)
TIER_ORDER = ["High", "Mid", "Low", "Blocked"]


def _violation_entry(field, severity, msg):
    return {"field": field, "severity": severity, "message": msg}


def check_pair(c, p):
    """c: client row (dict-like), p: profile row (dict-like).
    Returns list of violations."""
    out = []
    # ---- HARD: deal breakers, block the send ----
    if c["no_smoker"] and p["smoking"] != "never":
        out.append(_violation_entry("smoking",
                      "hard",
                      f"Client wants a non-smoker; profile smokes ({p['smoking']})"))
    if (c["children"] == "yes" and p["wants_children"] == "no") or \
       (c["children"] == "no" and p["wants_children"] == "yes"):
        out.append(_violation_entry("children", "hard", f"Children mismatch: client={c['children']}, profile={p['wants_children']}"))
    if c["veg_only"] and p["diet"] != "veg":
        out.append(_violation_entry("diet", "hard", "Client wants vegetarian; profile is non-veg"))
    rel = c["religions"]
    if rel is not None and not (isinstance(rel, float)) and p["religion"] not in rel:
        out.append(_violation_entry("religion", "hard", f"Religion {p['religion']} not in client's list {rel}"))
    # ---- SOFT: warn only, client behaviour shows these are negotiable ----
    if not (c["age_min"] <= p["age"] <= c["age_max"]):
        out.append(_violation_entry("age", "soft", f"Age {p['age']} outside {c['age_min']}-{c['age_max']}"))
    cities = c["cities"]
    if cities is not None and not (isinstance(cities, float)) and p["city"] not in cities:
        out.append(_violation_entry("city", "soft", f"City {p['city']} not in {cities}"))
    if EDU_RANK[p["education"]] < EDU_RANK[c["edu_min"]]:
        out.append(_violation_entry("education", "soft", f"Education {p['education']} below {c['edu_min']}"))
    hm = c["height_min"]
    if hm is not None and not (isinstance(hm, float) and hm != hm) and p["height_cm"] < hm:
        out.append(_violation_entry("height", "soft", f"Height {p['height_cm']}cm below {int(hm)}cm"))
    return out


def score_violations(vs):
    """Violations -> (score 0-100, tier). Hard violation = (0, 'Blocked')."""
    if any(v["severity"] == "hard" for v in vs):
        return 0, "Blocked"
    score = 100 - sum(SOFT_WEIGHTS[v["field"]] for v in vs)
    for name, cutoff in TIERS:
        if score >= cutoff:
            return score, name


def score_pair(c, p):
    vs = check_pair(c, p)
    score, tier = score_violations(vs)
    return dict(score=score, tier=tier, violations=vs)


def rank_profiles(client, profiles):
    """All profiles scored for one client, best first."""
    rows = []
    for p in profiles.to_dict("records"):
        vs = check_pair(client, p)
        score, tier = score_violations(vs)
        rows.append({**p, "score": score, "tier": tier,
                     "issues": "; ".join(v["message"] for v in vs)})
    out = pd.DataFrame(rows)
    return out.sort_values(["score", "profile_id"], ascending=[False, True]).reset_index(drop=True)


def tier_matrix(clients, profiles):
    """Rows = clients, columns = how many profiles fall in each tier."""
    plist = profiles.to_dict("records")
    rows = []
    for c in clients.to_dict("records"):
        counts = {t: 0 for t in TIER_ORDER}
        for p in plist:
            counts[score_violations(check_pair(c, p))[1]] += 1
        rows.append({"client_id": c["client_id"], "matchmaker": c["matchmaker"], **counts})
    return pd.DataFrame(rows).set_index("client_id")
