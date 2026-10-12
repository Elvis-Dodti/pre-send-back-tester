"""Synthetic data generator. ALL DATA IS MOCKED - no real clients or profiles."""
import random
import pandas as pd

CITIES = ["Mumbai", "Delhi", "Bengaluru", "Pune", "Hyderabad", "Chennai"]
RELIGIONS = ["Hindu", "Muslim", "Christian", "Sikh", "Jain"]
RELIGION_W = [0.60, 0.15, 0.10, 0.08, 0.07]
EDU = ["High school", "Graduate", "Postgraduate"]
EDU_RANK = {e: i for i, e in enumerate(EDU)}

# matchmaker -> how often they screen hard constraints before sending (assumption)
MATCHMAKERS = {"A": 0.95, "B": 0.10, "C": 0.55, "D": 0.30, "E": 0.65}

# Ground-truth reason templates (the free text a client might send back)
TEMPLATES = {
    "smoking": ["He smokes, I did mention I can't be with a smoker.",
                "Smoker. Not okay for me.",
                "Why did you send someone who smokes regularly?"],
    "children": ["Doesn't want kids, which is a dealbreaker for me.",
                 "Wants children, I don't. We discussed this.",
                 "Different views on having children."],
    "diet": ["Non-veg, I need a vegetarian family.",
             "Eats meat, my family is strictly veg.",
             "Diet doesn't match what I asked for."],
    "religion": ["Not from my community.",
                 "I only want someone of my religion.",
                 "Religion mismatch, please check my preferences."],
    "age": ["Too old for me.",
            "Age gap is bigger than I'd like.", "Younger than I wanted."],
    "city": ["Lives too far, I don't want to relocate.", "Different city, long distance won't work.", "Not in my city."],
    "education": ["Education level is lower than I hoped for.", "Not enough qualification for me."],
    "height": ["Too short for me.", "Height is an issue."],
    "other": ["Not feeling it.", "Not my type.", "Hmm no.", "Didn't connect with the photos.", "Not for me.", "Something felt off, can't say what."],
}


def make_profiles(n=300, rng=None):
    rng = rng or random.Random(0)
    rows = []
    for i in range(n):
        rows.append(dict(
            profile_id=f"P{i:04d}",
            age=rng.randint(24, 40),
            city=rng.choice(CITIES),
            religion=rng.choices(RELIGIONS, RELIGION_W)[0],
            smoking=rng.choices(["never", "occasionally", "regularly"], [0.70, 0.18, 0.12])[0],
            wants_children=rng.choices(["yes", "no", "open"], [0.55, 0.12, 0.33])[0],
            diet=rng.choices(["veg", "non-veg"], [0.40, 0.60])[0],
            education=rng.choices(EDU, [0.1, 0.55, 0.35])[0],
            height_cm=rng.randint(150, 190),
        ))
    return pd.DataFrame(rows)


def make_clients(n=50, rng=None):
    rng = rng or random.Random(1)
    mms = list(MATCHMAKERS)
    rows = []
    for i in range(n):
        age_min = rng.randint(24, 32)
        rows.append(dict(
            client_id=f"C{i:03d}",
            matchmaker=mms[i % len(mms)],
            # HARD constraints (deal breakers)
            no_smoker=rng.random() < 0.35,
            children=rng.choices(["yes", "no", "open"], [0.30, 0.07, 0.63])[0],
            veg_only=rng.random() < 0.12,
            religions=(None if rng.random() < 0.7 else
                       sorted(set([rng.choices(RELIGIONS, RELIGION_W)[0]]))),
            # SOFT preferences (standard matrimonial filters)
            age_min=age_min,
            age_max=age_min + rng.randint(4, 10),
            cities=(None if rng.random() < 0.35 else rng.sample(CITIES, rng.randint(1, 3))),
            edu_min=rng.choice(EDU),
            height_min=(None if rng.random() < 0.5 else rng.randint(155, 175)),
            # revealed-preference: prob. a soft violation is tolerated (hidden from checker)
            flex=round(rng.uniform(0.35, 0.95), 2),
        ))
    return pd.DataFrame(rows)


def _violations(c, p):
    """Local copy of the rules used only to simulate client behaviour."""
    from checker import check_pair
    return check_pair(c, p)


def simulate_shares(clients, profiles, n_shares=1000, seed=7):
    rng = random.Random(seed)
    cl = clients.to_dict("records")
    pr = profiles.to_dict("records")
    rows = []
    for k in range(n_shares):
        c = cl[k % len(cl)]
        diligence = MATCHMAKERS[c["matchmaker"]]
        p = rng.choice(pr)
        if rng.random() < diligence:  # matchmaker manually screens deal breakers
            for _ in range(25):
                if not any(v["severity"] == "hard" for v in _violations(c, p)):
                    break
                p = rng.choice(pr)
        viol = _violations(c, p)
        hard = [v for v in viol if v["severity"] == "hard"]
        soft = [v for v in viol if v["severity"] == "soft"]

        reason, accepted = None, False
        if hard:
            accepted = rng.random() < 0.04
            if not accepted:
                reason = rng.choice(hard)["field"] if rng.random() < 0.9 else "other"
        else:
            p_acc = 0.68 * (c["flex"] ** len(soft))
            accepted = rng.random() < p_acc
            if not accepted:
                if soft and rng.random() < 0.35:
                    reason = rng.choice(soft)["field"]
                else:
                    reason = "other"
        feedback = None if accepted else rng.choice(TEMPLATES["other"] if (reason != "other" and rng.random() < 0.25) else TEMPLATES[reason])
        rows.append(dict(
            share_id=k, client_id=c["client_id"], matchmaker=c["matchmaker"],
            profile_id=p["profile_id"], accepted=accepted,
            true_reason=reason, feedback=feedback,
        ))
    return pd.DataFrame(rows)


def generate(seed=7):
    profiles = make_profiles(rng=random.Random(seed))
    clients = make_clients(rng=random.Random(seed + 1))
    shares = simulate_shares(clients, profiles, seed=seed + 2)
    return clients, profiles, shares


if __name__ == "__main__":
    c, p, s = generate()
    print(s.accepted.mean(), len(s))
