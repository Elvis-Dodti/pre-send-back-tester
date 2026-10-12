# Date Crew: match suggestions + pre-send preference checker (backtest prototype)

Pick a client and get every profile scored and grouped into High / Mid / Low / Blocked.
Replays 1,000 synthetic profile shares through the same rules to measure what would have been caught.

- `data_gen.py`  synthetic clients, profiles, shares, rejection feedback (MOCK DATA)
- `checker.py`   hard constraints (block) vs soft preferences (warn); scoring and tiers; rule-based, no LLM
- `tagger.py`    free-text rejection -> structured tags (keyword baseline; Claude if ANTHROPIC_API_KEY set)
- `backtest.py`  metrics: acceptance lift, shares avoided, false blocks, per-matchmaker table, acceptance by tier
- `app.py`       Streamlit UI: Match suggestions / Backtest / Pre-send checker / Feedback tagger

## Scoring
Any deal-breaker violated (smoking, children, diet, religion) = **Blocked**.
Otherwise score = 100 minus a penalty per soft miss (age 15, education 15, city 10, height 10).
**High** >= 90, **Mid** 75-89, **Low** < 75. Weights are assumptions to be tuned on real history.

## Run
    pip install -r requirements.txt
    python backtest.py          # CLI summary
    streamlit run app.py        # UI
    # optional: ANTHROPIC_API_KEY=... and tick "Use Claude" in the sidebar

## Assumptions
Matchmaker screening diligence, client flexibility and vague-feedback rate are invented to roughly
match the brief (31% acceptance, ~35% of rejections already covered by stated preferences).
Projection removes blocked shares without replacement, so it is conservative on volume and
optimistic if replacements are rejected for other reasons.
