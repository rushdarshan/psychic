import numpy as np

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# 3 sub-spans with preamble stripping applied to span 1


def load_model():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBEDDING_MODEL)
    return model


def embed_texts(model, texts, batch_size=64):
    if not texts:
        return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings.astype(np.float32)


def cosine_similarity(a, b):
    return float(np.dot(a, b))


JD_SUB_SECTION_GROUPS = [
    ["description", "what_we_need"],
    ["key_responsibilities"],
    ["ideal_candidate"],
]

JD_SUB_WEIGHTS = [0.25, 0.30, 0.45]


def build_jd_text(jd_data):
    jd_sections = [
        jd_data.get("description", ""),
        jd_data.get("ideal_candidate", ""),
        jd_data.get("what_we_need", ""),
        jd_data.get("key_responsibilities", ""),
    ]
    return " ".join(s for s in jd_sections if s)


def _strip_jd_preamble(text):
    """Remove location/company blurb prefix from sub-span 1."""
    lines = text.split("\n")
    stripped = [ln for ln in lines if not (
        ln.lower().startswith("location:") or
        ln.lower().startswith("opening:") or
        ln.lower().startswith("department:") or
        ln.lower().startswith("company:") or
        ln.lower().startswith("about") or
        ln.lower().startswith("let's be honest")
    )]
    return " ".join(s for s in stripped if s).strip()


def build_jd_sub_texts(jd_data):
    """Decompose JD text into semantically meaningful sub-spans.

    Returns a list of text strings, one per concept group.
    Empty groups are omitted.
    """
    texts = []
    for i, group in enumerate(JD_SUB_SECTION_GROUPS):
        parts = [jd_data.get(k, "") for k in group]
        text = " ".join(p for p in parts if p).strip()
        if i == 0 and text:
            text = _strip_jd_preamble(text)
        if text:
            texts.append(text)
    return texts


def build_candidate_text(candidate):
    profile = candidate.get("profile", {})
    summary = profile.get("summary", "")
    headline = profile.get("headline", "")
    career_history = candidate.get("career_history", [])
    descriptions = " ".join(
        e.get("description", "") for e in career_history
    )
    text = f"{headline} {summary} {descriptions}"
    return text
