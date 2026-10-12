"""Turn free-text rejection feedback into structured reason tags.
Keyword baseline always works offline. LLM tagger is used if ANTHROPIC_API_KEY is set."""
import json, os, re

TAGS = ["smoking", "children", "diet", "religion", "age", "city", "education",
        "height", "other"]

KEYWORDS = {
    "smoking": r"smok",
    "children": r"\bkids?\b|child|baby|babies",
    "diet": r"\bveg|non-veg|\bmeat\b|diet|\beggs?\b",
    "religion": r"religio|communit|\bhindu|muslim|christian|sikh|jain|church|temple",
    "age": r"\bage\b|too old|older|younger|\byoung",
    "city": r"\bcity\b|relocat|too far|long distance|distance|\bmumbai|delhi|bengaluru|pune|hyderabad|chennai",
    "education": r"educat|qualif|degree",
    "height": r"\btall|\bshort|height",
}


def keyword_tag(text):
    if not text:
        return ["other"]
    t = text.lower()
    tags = [k for k, rx in KEYWORDS.items() if re.search(rx, t)]
    return tags or ["other"]


def llm_tag_batch(texts, model=None):
    """Tags a list of feedback strings with Claude. Returns list[list[str]]."""
    import anthropic
    client = anthropic.Anthropic()
    model = model or os.getenv("TAGGER_MODEL", "claude-haiku-5-5")
    out = []
    for i in range(0, len(texts), 25):
        chunk = texts[i:i + 25]
        numbered = "\n".join(f"{j}: {t}" for j, t in enumerate(chunk))
        msg = client.messages.create(
            model=model, max_tokens=1500,
            system=(
                "You classify matchmaking rejection feedback. Allowed tags: "
                f"{TAGS}. Return ONLY JSON: a list of objects {{\"i\": int, \"tags\": [..]}}. "
                "Use 'other' when the reason is vague or about chemistry/photos."
            ),
            messages=[{"role": "user", "content": numbered}],
        )
        raw = re.sub(r"```json|```", "", msg.content[0].text).strip()
        parsed = {d["i"]: [t for t in d["tags"] if t in TAGS] or ["other"] for
                  d in json.loads(raw)}
        out.extend(parsed.get(j, ["other"]) for j in range(len(chunk)))
    return out


def tag_feedback(texts, use_llm=False):
    texts = list(texts)
    if use_llm and os.getenv("ANTHROPIC_API_KEY"):
        try:
            # dedupe: templates repeat a lot, saves cost
            uniq = sorted({t for t in texts if t})
            mapping = dict(zip(uniq, llm_tag_batch(uniq)))
            return [mapping.get(t, ["other"]) for t in texts]
        except Exception as e:  # fall back, never block the workflow
            print("LLM tagger failed, using keyword baseline:", e)
    return [keyword_tag(t) for t in texts]
