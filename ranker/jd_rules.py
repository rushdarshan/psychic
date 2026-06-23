import re

# Keyword rules and classifiers for Senior AI Engineer JD matching


def _alternation(phrases):
    return re.compile(
        r"\b(?:" + "|".join(re.escape(p) for p in phrases) + r")\b",
        re.IGNORECASE,
    )


RESEARCH_TERMS = [
    "research", "academic", "lab", "phd", "publication",
    "paper", "conference", "journal",
]

PRODUCTION_TERMS = [
    "in production", "deploy", "deployed to", "shipped a",
    "launch", "production system", "production deployment",
    "serving", "latency", "a/b test", "a/b testing",
]

CONSULTING_FIRMS = {
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "hcl", "tech mahindra", "l&t infotech", "ltimindtree", "mindtree",
    "mphasis", "oracle financial services", "sutherland", "genpact",
}

SHIPPED_SYSTEM_KEYWORDS = [
    "ranking", "search", "recommendation", "recommender", "recsys",
    "retrieval", "information retrieval", "semantic search",
    "vector search", "hybrid search", "candidate matching",
    "job recommendation", "relevance", "personalized search",
    "personalized recommendation", "match score", "relevance score",
    "ndcg", "mrr", "map", "learning to rank", "ltr",
]

SHIPPED_SYSTEM_TECH_KEYWORDS = [
    "sentence-transformers", "openai embeddings", "bge", "e5",
    "pinecone", "weaviate", "qdrant", "milvus", "opensearch",
    "elasticsearch", "faiss", "embedding", "vector database",
    "hybrid retrieval", "dense retrieval", "sparse retrieval",
]

NLP_IR_KEYWORDS = [
    "nlp", "natural language", "text classification", "sentiment",
    "named entity", "ner", "text mining", "language model",
    "llm", "gpt", "bert", "transformer", "attention",
    "fine-tuning", "fine tuning", "rag", "retrieval augmented",
    "tokenizer", "corpus", "document", "semantic",
]

CV_SPEECH_ROBOTICS_KEYWORDS = [
    "computer vision", "image classification", "object detection",
    "yolo", "cnn", "convolutional", "image segmentation",
    "speech recognition", "speech to text", "tts", "text to speech",
    "robotics", "slam", "point cloud", "lidar", "autonomous",
]

ML_CAREER_TITLE_TOKENS = [
    "machine learning", "deep learning", "data scientist", "data science",
    "artificial intelligence", "natural language",
    "recommendation", "recsys", "recommender",
    "search", "ranking", "rank",
    "applied ml", "applied ai", "applied machine learning",
    "llm", "gpt", "nlp", "rag",
    "ai engineer", "ml engineer", "nlp engineer",
    "recommendation systems", "information retrieval",
    "relevance", "personalization",
]

_SHIPPED_RE = _alternation(SHIPPED_SYSTEM_KEYWORDS)
_SHIPPED_TECH_RE = _alternation(SHIPPED_SYSTEM_TECH_KEYWORDS)
_NLP_IR_RE = _alternation(NLP_IR_KEYWORDS)
_CV_RE = _alternation(CV_SPEECH_ROBOTICS_KEYWORDS)
_RESEARCH_RE = _alternation(RESEARCH_TERMS)
_PRODUCTION_RE = _alternation(PRODUCTION_TERMS)

ML_TITLE_TOKENS = [
    "machine learning", "deep learning", "data scientist", "data science",
    "artificial intelligence", "natural language",
    "recommendation", "recsys", "recommender",
    "search", "ranking", "rank",
    "applied ml", "applied ai", "applied machine learning",
    "llm", "gpt", "nlp", "rag",
    "ai engineer", "ml engineer", "nlp engineer",
]

TECH_TITLE_TOKENS = [
    "software engineer", "backend", "frontend", "full stack",
    "devops", "qa", "cloud engineer", "data engineer",
    "platform engineer", "infrastructure",
    "mobile developer", "ios", "android", "web developer",
]

OFF_DOMAIN_TITLE_TOKENS = [
    "graphic designer", "accountant", "sales executive",
    "customer support", "content writer", "hr manager",
    "human resources", "marketing manager", "civil engineer",
    "mechanical engineer", "operations manager",
    "business analyst", "project manager", "product manager",
    "retail analyst",
]

_ML_ROLE_COMBINED_RE = _alternation(
    ML_CAREER_TITLE_TOKENS + SHIPPED_SYSTEM_KEYWORDS + NLP_IR_KEYWORDS
)

_ML_TITLE_RE = _alternation(ML_TITLE_TOKENS)
_TECH_TITLE_RE = _alternation(TECH_TITLE_TOKENS)
_OFF_DOMAIN_TITLE_RE = _alternation(OFF_DOMAIN_TITLE_TOKENS)
JD_COVERAGE_KEYWORDS = [
    "machine learning", "deep learning", "nlp", "natural language processing",
    "embedding", "sentence transformer", "retrieval",
    "information retrieval", "ranking", "search",
    "recommendation", "recommender system",
    "llm", "large language model", "gpt", "bert", "transformer",
    "rag", "retrieval augmented", "fine tuning",
    "vector search", "semantic search", "hybrid search",
    "candidate matching", "relevance",
    "learning to rank", "personalization",
]

_JD_COVERAGE_RE = _alternation(JD_COVERAGE_KEYWORDS)

_ML_CAREER_TITLE_RE = _alternation(ML_CAREER_TITLE_TOKENS)

PRODUCT_COMPANY_INDUSTRIES = {
    "software", "internet", "saas", "technology", "e-commerce",
    "fintech", "healthtech", "edtech", "hr tech", "marketplace",
    "social media", "advertising technology", "transportation",
    "conglomerate", "manufacturing",
}


def compute_title_level(title):
    title_lower = title.lower()
    if any(t in title_lower for t in ["principal", "distinguished", "fellow"]):
        return 4
    if any(t in title_lower for t in ["staff", "lead", "architect"]):
        return 3
    if any(t in title_lower for t in ["senior", "sr", "head of", "manager"]):
        return 2
    return 1


def detect_job_hopping(career_history):
    durations = [
        entry.get("duration_months", 0)
        for entry in career_history
        if not entry.get("is_current", False)
    ]
    if not durations:
        return 0, 0.0
    avg = sum(durations) / len(durations)
    short_stints = sum(1 for d in durations if d < 18)
    ratio = short_stints / len(durations) if durations else 0
    return ratio, avg


def is_consulting_only(career_history):
    if not career_history:
        return False
    has_product = any(
        entry.get("industry", "").lower() in PRODUCT_COMPANY_INDUSTRIES
        and entry.get("company", "").lower() not in CONSULTING_FIRMS
        for entry in career_history
    )
    if has_product:
        return False
    all_consulting = all(
        entry.get("company", "").lower() in CONSULTING_FIRMS
        for entry in career_history
    )
    return all_consulting


def _hits(text, pattern):
    return len(pattern.findall(text))


def has_shipped_system_evidence(description):
    return _hits(description, _SHIPPED_RE) + _hits(description, _SHIPPED_TECH_RE) > 0


def has_nlp_ir_exposure(description):
    return _hits(description, _NLP_IR_RE) > 0


def has_cv_speech_robotics_background(description):
    nlp_hits = _hits(description, _NLP_IR_RE)
    cv_hits = _hits(description, _CV_RE)
    return cv_hits > nlp_hits and cv_hits > 0


def is_pure_research(description):
    r_score = _hits(description, _RESEARCH_RE)
    p_score = _hits(description, _PRODUCTION_RE)
    return r_score > 0 and p_score == 0


def compute_title_domain_penalty(title):
    if _hits(title, _ML_TITLE_RE) > 0 or _hits(title, _TECH_TITLE_RE) > 0:
        return 0.0
    if _hits(title, _OFF_DOMAIN_TITLE_RE) > 0:
        return 0.6
    return 0.3


def is_ml_role(entry):
    title = (entry.get("title", "") or "").lower()
    if _hits(title, _ML_CAREER_TITLE_RE) > 0:
        return True
    desc = (entry.get("description", "") or "").lower()
    return _hits(desc, _ML_ROLE_COMBINED_RE) >= 2


def compute_applied_ml_years(career_history):
    total_months = 0
    product_months = 0
    for entry in career_history:
        if is_ml_role(entry):
            duration = entry.get("duration_months", 0)
            total_months += duration
            industry = (entry.get("industry", "") or "").lower()
            if industry in PRODUCT_COMPANY_INDUSTRIES:
                product_months += duration
    return total_months / 12, product_months / 12


def compute_jd_keyword_coverage(text):
    """TF-weighted JD keyword coverage.

    For each JD keyword, count occurrences and apply diminishing-returns
    TF weighting: score = count / (count + 1).  Then average across all
    keywords.  This preserves term-frequency signal (like BM25) without
    full tokenization overhead.
    """
    if not text:
        return 0.0
    text_lower = text.lower()
    total = 0.0
    for kw in JD_COVERAGE_KEYWORDS:
        count = text_lower.count(kw)
        if count > 0:
            total += count / (count + 1)
    return total / len(JD_COVERAGE_KEYWORDS)
