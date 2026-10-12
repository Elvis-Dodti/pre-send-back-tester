import pandas as pd
import streamlit as st
from data_gen import generate
from backtest import run_backtest
from checker import check_pair, rank_profiles, tier_matrix, score_violations, \
    TIER_ORDER
from tagger import tag_feedback

st.set_page_config(
    page_title="Date Crew - Match suggestions & pre-send checker",
    layout="wide")
st.title("Match suggestions, pre-send checker and backtest")
st.caption(
    "All data is synthetic. Rules are deterministic; the LLM is only used to structure free-text rejections.")

seed = st.sidebar.number_input("Random seed", value=7, step=1)
use_llm = st.sidebar.checkbox(
    "Use Claude for feedback tagging (needs ANTHROPIC_API_KEY)", value=False)


@st.cache_data
def load(seed, use_llm):
    clients, profiles, shares = generate(seed)
    df, summary, by_mm, by_tier = run_backtest(clients, profiles, shares,
                                               use_llm)
    return clients, profiles, df, summary, by_mm, by_tier


@st.cache_data
def load_matrix(seed):
    clients, profiles, _ = generate(seed)
    return tier_matrix(clients, profiles)


clients, profiles, df, s, by_mm, by_tier = load(int(seed), use_llm)
tab0, tab1, tab2, tab3 = st.tabs([
    "Match suggestions",
    "Backtest",
    "Pre-send checker",
    "Feedback tagger"
])

with tab0:
    st.subheader("Pick a client, get ranked profile suggestions")
    cid = st.selectbox("Client", clients.client_id, key="match_client")
    crow = clients.set_index("client_id").loc[cid].to_dict()
    ranked = rank_profiles({**crow, "client_id": cid}, profiles)

    counts = ranked.tier.value_counts().reindex(TIER_ORDER).fillna(0).astype(
        int)
    m = st.columns(4)
    for col, t in zip(m, TIER_ORDER):
        col.metric(t, int(counts[t]))

    with st.expander("Client preferences"):
        st.markdown("**Deal-breakers (hard):** "
                    f"non-smoker only: {crow['no_smoker']}, children: {crow['children']}, "
                    f"veg only: {crow['veg_only']}, religions: {crow['religions'] or 'any'}")
        hm = crow["height_min"]
        hm_txt = "any" if (hm is None or hm != hm) else f"{int(hm)}cm"
        st.markdown("**Preferences (soft):** "
                    f"age {crow['age_min']}-{crow['age_max']}, cities: {crow['cities'] or 'any'}, "
                    f"education min: {crow['edu_min']}, height min: {hm_txt}")

    show_cols = ["profile_id", "score", "age", "city", "religion", "education",
                 "height_cm", "smoking", "wants_children", "diet", "issues"]
    top_n = st.slider("Profiles shown per tier", 5, 30, 10)
    for tier, blurb in [("High",
                         "Safe to send. No deal-breaker issues and at most one minor miss."),
                        ("Mid",
                         "Sendable, but flag the misses to the client."),
                        ("Low",
                         "Several soft misses. Send only if nothing better.")]:
        sub = ranked[ranked.tier == tier]
        st.markdown(f"### {tier} ({len(sub)})")
        st.caption(blurb)
        if sub.empty:
            st.write("None for this client.")
        else:
            st.dataframe(sub[show_cols].head(top_n), width="stretch",
                         hide_index=True)
    with st.expander(
            f"Blocked ({int(counts['Blocked'])}): violates a deal-breaker, do not send"):
        st.dataframe(ranked[ranked.tier == "Blocked"][show_cols].head(50),
                     width="stretch", hide_index=True)

    st.subheader(
        "Does the score mean anything? Acceptance on past shares, by tier")
    st.dataframe(by_tier, width="stretch")
    st.caption(
        "If scoring works, acceptance should fall from High to Blocked.")

    st.subheader("Match matrix: profiles per tier, for every client")
    st.dataframe(load_matrix(int(seed)), width="stretch")

# ---------------------------------------------------------------- Backtest
with tab1:
    c = st.columns(4)
    c[0].metric("Acceptance today", f"{s['acceptance_rate']:.1%}")
    c[1].metric("Acceptance if deal-breakers blocked",
                f"{s['acceptance_after']:.1%}", f"{s['lift_pts']:+.1f} pts")
    c[2].metric("Shares avoided (matchmaker time)",
                f"{s['hard_blocked_pct_of_shares']:.1%}")
    c[3].metric("Preventable rejections",
                f"{s['preventable_share_of_rejections']:.0%} of rejections")

    st.markdown(
        f"**{s['rejections_caught']}** past rejections would have been caught before sending; "
        f"**{s['false_blocks']}** accepted profiles would have been wrongly blocked. "
        f"Tagger accuracy vs. hidden ground truth: **{s['tagger_accuracy']:.0%}**."
    )
    st.subheader("By matchmaker")
    st.dataframe(by_mm, width="stretch")
    st.subheader("Why soft preferences warn but don't block")
    st.write(
        f"Acceptance for profiles violating only soft preferences: **{s['accept_rate_soft_violators']:.1%}** "
        f"vs. **{s['accept_rate_clean']:.1%}** for clean matches. Clients tolerate soft misses far more often than deal-breakers."
    )
    st.subheader("Rejection reasons (structured from free text)")
    tags = df[~df.accepted].tags.explode().value_counts()
    st.bar_chart(tags)
    with st.expander("Raw backtest rows"):
        st.dataframe(df.drop(columns=["tags"]).head(200))

# ---------------------------------------------------------------- Pre-send checker
with tab2:
    cid2 = st.selectbox("Client", clients.client_id, key="check_client")
    pid = st.selectbox("Profile to send", profiles.profile_id)
    crow2 = clients.set_index("client_id").loc[cid2].to_dict()
    prow = profiles.set_index("profile_id").loc[pid].to_dict()
    v = check_pair(crow2, prow)
    score, tier = score_violations(v)
    hard = [x for x in v if x["severity"] == "hard"]
    if hard:
        st.error("BLOCK: " + "; ".join(x["message"] for x in hard))
    elif v:
        st.warning(f"WARN ({tier}, score {score}): " + "; ".join(
            x["message"] for x in v))
    else:
        st.success(f"OK to send (High, score {score})")
    col1, col2 = st.columns(2)
    col1.json(crow2)
    col2.json(prow)

# ---------------------------------------------------------------- Feedback tagger
with tab3:
    txt = st.text_area("Paste a client's free-text rejection",
                       "He smokes and lives too far, not for me.")
    st.write(tag_feedback([txt], use_llm)[0])
