"""Backtest: replay past shares through the checker and measure what it would have caught."""
import pandas as pd
from checker import check_pair, score_violations, HARD_FIELDS, TIER_ORDER
from tagger import tag_feedback


def run_backtest(clients, profiles, shares, use_llm=False):
    cdict = clients.set_index("client_id").to_dict("index")
    pdict = profiles.set_index("profile_id").to_dict("index")
    df = shares.copy()

    # 1. checker + score over history
    viols = [check_pair(cdict[r.client_id], pdict[r.profile_id]) for r in df.itertuples()]
    df["hard_fields"] = [[v["field"] for v in vs if v["severity"] == "hard"] for vs in viols]
    df["soft_fields"] = [[v["field"] for v in vs if v["severity"] == "soft"] for vs in viols]
    df["blocked"] = df.hard_fields.str.len() > 0
    df["warned"] = df.soft_fields.str.len() > 0
    scored = [score_violations(vs) for vs in viols]
    df["score"] = [s for s, _ in scored]
    df["tier"] = [t for _, t in scored]

    # 2. structure the free-text feedback
    rej = ~df.accepted
    df["tags"] = [["accepted"]] * len(df)
    df.loc[rej, "tags"] = pd.Series(tag_feedback(df.loc[rej, "feedback"], use_llm), index=df.index[rej])
    stated_fields = set(HARD_FIELDS + ["age", "city", "education", "height"])
    df["preventable"] = rej & df.tags.apply(lambda t: any(x in stated_fields for x in t))
    df["preventable_hard"] = rej & df.tags.apply(lambda t: any(x in HARD_FIELDS for x in t))

    n, acc, nrej = len(df), int(df.accepted.sum()), int(rej.sum())
    blocked = df[df.blocked]
    kept = df[~df.blocked]
    summary = dict(
        shares=n, accepted=acc, acceptance_rate=acc / n,
        rejected=nrej,
        preventable_share_of_rejections=df.preventable.sum() / nrej,
        preventable_rate_of_shares=df.preventable.sum() / n,
        hard_blocked=len(blocked),
        hard_blocked_pct_of_shares=len(blocked) / n,
        rejections_caught=int((blocked.accepted == False).sum()),
        recall_hard_preventable=(df.preventable_hard & df.blocked).sum() / max(df.preventable_hard.sum(), 1),
        false_blocks=int(blocked.accepted.sum()),  # accepted profiles we'd have wrongly blocked
        acceptance_after=kept.accepted.mean(),
        shares_after=len(kept), accepted_after=int(kept.accepted.sum()),
        accept_rate_soft_violators=df[df.warned & ~df.blocked].accepted.mean(),
        accept_rate_clean=df[~df.warned & ~df.blocked].accepted.mean(),
    )
    summary["lift_pts"] = (summary["acceptance_after"] - summary["acceptance_rate"]) * 100
    summary["time_saved_pct"] = summary["hard_blocked_pct_of_shares"]

    by_mm = df.groupby("matchmaker").apply(lambda g: pd.Series(dict(
        shares=len(g), acceptance=g.accepted.mean(), hard_flag_rate=g.blocked.mean(),
        acceptance_after=g[~g.blocked].accepted.mean())), include_groups=False).round(3)

    # 3. does the match score mean anything? acceptance by tier on past shares
    by_tier = (df.groupby("tier").accepted.agg(shares="count", acceptance="mean")
                 .reindex(TIER_ORDER).dropna().round(3))

    # 4. tagger quality vs hidden ground truth (only possible because data is synthetic)
    r = df[rej]
    summary["tagger_accuracy"] = float(r.apply(lambda x: x.true_reason in x.tags, axis=1).mean())
    return df, summary, by_mm, by_tier


if __name__ == "__main__":
    from data_gen import generate
    c, p, s = generate()
    data, sm, mm, bt = run_backtest(c, p, s)
    for k, v in sm.items():
        print(f"{k:34s} {v:.3f}" if isinstance(v, float) else f"{k:34s} {v}")
    print(mm)
    print(bt)
