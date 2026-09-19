"""Step 4 — c-TF-IDF cluster labelling with optional LLM refinement."""

from __future__ import annotations

import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize

from run_store import load_config
from utils.logger import get_logger

logger = get_logger(__name__)

try:
    import stopwordsiso as _swiso
    _STOPWORDS_ISO_AVAILABLE = True
except ImportError:
    _STOPWORDS_ISO_AVAILABLE = False


def _clean_text(text: str) -> str:
    """Lowercase, strip URLs/mentions/hashtag sigils, remove non-letter chars.

    Unicode-aware: keeps letters from *any* script (Latin, Cyrillic, Arabic,
    CJK, Devanagari, Thai, ...), not just Latin + French accents. The old
    ASCII-only character whitelist silently emptied out every post written
    in a non-Latin script, even though `_build_stopwords` already loads
    stopword lists for those same languages (ja, ar, ru, ko, hi, th, zh,
    uk, ...) — the two stages disagreed on which languages were supported.
    """
    text = str(text).lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = "".join(ch if (ch.isalpha() or ch.isspace()) else " " for ch in text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Top 25 social-media languages (2024) — all supported by stopwordsiso
_LANGS = [
    "en", "ja", "es", "pt", "ar", "fr", "id", "ru", "tr", "ko",
    "de", "it", "nl", "hi", "th", "pl", "ms", "zh", "vi", "tl",
    "uk", "sv", "fi", "no", "da",
]


_TOKEN_RE = re.compile(r"(?u)\b[^\W\d_]{3,}\b")


def _build_stopwords() -> list:
    if _STOPWORDS_ISO_AVAILABLE:
        sw = [w for w in _swiso.stopwords(_LANGS) if w == w.lower() and _TOKEN_RE.fullmatch(w)]
        logger.info("stopwordsiso: %d stopwords  (%d languages)", len(sw), len(_LANGS))
        return sw
    logger.warning("stopwordsiso not installed — falling back to sklearn english stopwords")
    logger.warning("fix: pip install stopwordsiso")
    return list(CountVectorizer(stop_words="english").get_stop_words())


def _build_vectorizer(min_df: int, ngram_range: tuple) -> CountVectorizer:
    sw = _build_stopwords()
    return CountVectorizer(
        min_df=min_df,
        ngram_range=ngram_range,
        stop_words=sw,
        max_features=50_000,
        # sklearn's default token_pattern (\b\w\w+\b) allows digits,
        # underscores, and 2-char tokens. [^\W\d_] is the standard
        # Unicode-aware trick for "letters only" using stdlib re (matches
        # \w minus digits minus underscore), so this restricts to 3+
        # letters from ANY script — matching what _clean_text now keeps.
        # (Previously hardcoded to Latin + French accents only, which
        # silently zeroed out non-Latin-script languages even though
        # _build_stopwords loads stopword lists for them.)
        token_pattern=_TOKEN_RE.pattern,
    )


def _ctfidf_scores(X) -> np.ndarray:
    """Class-based TF-IDF with square-root term frequency (Grootendorst, 2022); see THIRD_PARTY_NOTICES.md."""
    X   = sp.csr_matrix(X, dtype=np.float64)
    df  = np.maximum(np.asarray(X.sum(axis=0)).ravel(), 1)
    avg = int(X.sum(axis=1).mean())
    idf = np.log(avg / df + 1)
    tf  = normalize(X, norm="l1", axis=1)
    tf.data = np.sqrt(tf.data)
    return (tf @ sp.diags(idf)).toarray()


def _compute_ctfidf(docs_per_group: dict, vectorizer: CountVectorizer, top_n_words: int) -> dict:
    """Compute c-TF-IDF scores; return top_n_words per group."""
    group_ids = list(docs_per_group.keys())
    texts     = [docs_per_group[g] for g in group_ids]

    X     = vectorizer.transform(texts)
    vocab = vectorizer.get_feature_names_out()

    scores_matrix = _ctfidf_scores(X)

    result = {}
    for i, gid in enumerate(group_ids):
        row     = scores_matrix[i]
        top_idx = row.argsort()[::-1][:top_n_words]
        result[gid] = [vocab[j] for j in top_idx if row[j] > 0]
    return result


def _select_representative_posts(
    posts: pd.DataFrame,
    all_embeddings: np.ndarray,
    mask: np.ndarray,
    n_posts: int,
) -> list:
    """Return up to n_posts texts closest to the cluster centroid (cosine sim)."""
    idx = np.where(mask)[0]
    if len(idx) <= n_posts:
        return posts.iloc[idx]["_text_clean"].tolist()

    group_embs    = all_embeddings[idx]
    centroid      = group_embs.mean(axis=0)
    centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-10)
    embs_norm     = group_embs / (np.linalg.norm(group_embs, axis=1, keepdims=True) + 1e-10)
    sims          = embs_norm @ centroid_norm
    top_local     = sims.argsort()[::-1][:n_posts]
    return posts.iloc[idx[top_local]]["_text_clean"].tolist()


def _build_prompt(documents: list, keywords: list) -> str:
    docs_block     = "\n".join(f"- {d}" for d in documents)
    keywords_block = ", ".join(keywords)
    return (
        "I have a topic that contains the following documents:\n"
        f"{docs_block}\n"
        f"The topic is described by the following keywords: {keywords_block}\n"
        "Based on the information above, extract a short but highly descriptive "
        "topic label of at most 5 words. The label must be in English.\n"
        "Make sure it is in the following format:\n"
        "topic: <topic label>"
    )


def _extract_label(text: str) -> str | None:
    """Extract the 'topic: <label>' line. Returns None if the format doesn't
    match, so the caller can treat it as a failed attempt (retry-worthy)
    instead of silently accepting garbage."""
    match = re.search(r"topic:\s*(.+)", text, re.IGNORECASE)
    if match:
        label = match.group(1).strip().strip('"\'').strip()
        if label:
            return label
    return None


# Retry ceiling for _make_llm_labeler — deliberately looser than the
# "at most 5 words" instruction in _build_prompt's prompt (3 words of
# slack), so a label that slightly overshoots isn't retried; only a
# clear miss is.
_MAX_LABEL_WORDS = 8

# Matches "topic: ..." labels that are actually the LLM declining to
# answer (e.g. "N/A - Insufficient valid data provided") rather than a
# real label. These pass the "topic: ..." format check and the word-count
# check just fine, so without this they get accepted as if they were
# genuine labels. Treated as a retry-worthy failure, same as an
# unparsable response.
_REFUSAL_PATTERN = re.compile(
    r"\b("
    r"n/?a"
    r"|insufficient"
    r"|unable to"
    r"|cannot determine"
    r"|not enough (data|information)"
    r"|no (clear |sufficient )?(topic|data|information)"
    r"|unclear"
    r")\b",
    re.IGNORECASE,
)


def _make_llm_labeler(cfg: dict, max_retries: int = 3):
    """Return a labeler closure that calls the LLM and falls back to c-TF-IDF on error.

    One retry loop (max_retries attempts) handles all four failure causes:
      - the LLM call itself raises (HTTP/network error)
      - the LLM responded, but the output doesn't match the expected
        "topic: ..." format
      - the LLM produced a well-formed "topic: ..." line, but the label
        itself is a refusal/non-answer (e.g. "N/A - Insufficient valid
        data provided") rather than an actual topic — see
        _REFUSAL_PATTERN. Without this check such text passes the format
        and word-count checks and gets accepted as if it were a real
        label.
      - the LLM returned a label, but it's too long (> _MAX_LABEL_WORDS
        words) — the model missed the "5 words max" instruction by more
        than the tolerance built into _MAX_LABEL_WORDS

    Only after max_retries attempts fail does it fall back to c-TF-IDF.
    """
    from _llm import get_llm_cfg, llm_call
    ncfg = get_llm_cfg(cfg, "naming")

    def labeler(documents: list, keywords: list, fallback: str) -> str:
        prompt      = _build_prompt(documents, keywords)
        last_reason = None

        for attempt in range(1, max_retries + 1):
            try:
                content = llm_call(ncfg, "", prompt)
            except Exception as e:
                last_reason = f"{type(e).__name__}: {e}"
                logger.warning("attempt %d/%d failed (network): %s", attempt, max_retries, e)
                continue

            label = _extract_label(content)
            if label is None:
                last_reason = "unparsable LLM output"
                logger.warning(
                    "attempt %d/%d failed (LLM up but unparsable output): %r",
                    attempt, max_retries, content[:200],
                )
                continue

            if _REFUSAL_PATTERN.search(label):
                last_reason = f"LLM declined to label ({label!r})"
                logger.warning(
                    "attempt %d/%d failed (LLM returned a refusal, not a label): %r",
                    attempt, max_retries, label,
                )
                continue

            n_words = len(label.split())
            if n_words > _MAX_LABEL_WORDS:
                last_reason = f"label too long ({n_words} words)"
                logger.warning(
                    "attempt %d/%d failed (label too long, %d words > %d): %r",
                    attempt, max_retries, n_words, _MAX_LABEL_WORDS, label,
                )
                continue

            return label

        logger.warning(
            "LLM unavailable after %d attempt(s) (%s) — c-TF-IDF fallback for this cluster",
            max_retries, last_reason,
        )
        return fallback

    return labeler


def _load_embedding_col(posts_df: pd.DataFrame) -> np.ndarray:
    """Extract embedding_raw column into a contiguous float32 ndarray."""
    col = posts_df["embedding_raw"].values
    if isinstance(col[0], np.ndarray):
        return np.stack(col).astype(np.float32)
    if isinstance(col[0], bytes):
        return np.stack([pickle.loads(e) for e in col]).astype(np.float32)
    if isinstance(col[0], (list, tuple)):
        return np.array([np.array(e) for e in col], dtype=np.float32)
    raise ValueError(f"unsupported embedding format: {type(col[0])}")


def run(
    posts_df: pd.DataFrame,
    config_path: str = "config.yaml",
    top_n_candidates: int = 10,
    n_representative_posts: int = 15,
    min_df: int = 2,
    ngram_range: tuple = (1, 2),
    force_skip_llm: bool = None,
) -> pd.DataFrame:
    """Label clusters via c-TF-IDF keywords, optionally refined by LLM.

    Args:
        posts_df: posts with cluster_id and content populated.
        config_path: path to config.yaml.
        top_n_candidates: c-TF-IDF keywords per cluster passed to LLM.
        n_representative_posts: posts sent to LLM per cluster.
        min_df: CountVectorizer min_df.
        ngram_range: CountVectorizer ngram_range.
        force_skip_llm: True → c-TF-IDF only; False → LLM confirmed upstream, skip ping;
            None → standalone mode, ping here.

    Returns:
        posts_df with cluster_label and cluster_topic columns added.
    """
    cfg = load_config(config_path)
    logger.info("%d posts to label", len(posts_df))

    posts_df = posts_df.copy()
    posts_df["_text_clean"] = posts_df["content"].fillna("").apply(_clean_text)

    has_embeddings = "embedding_raw" in posts_df.columns
    if has_embeddings:
        all_embeddings = _load_embedding_col(posts_df)
        logger.info("embedding dim: %dd", all_embeddings.shape[1])
    else:
        all_embeddings = None
        logger.warning("no embeddings — using the first %d posts per cluster (dataframe order) as representative posts", n_representative_posts)

    from _llm import check_llm_available, get_llm_cfg
    _llm_ncfg = get_llm_cfg(cfg, "naming")

    if force_skip_llm is True:
        logger.info("force_skip_llm=True — c-TF-IDF only")
        labeler = None
    elif force_skip_llm is False:
        logger.info("force_skip_llm=False — LLM confirmed available upstream")
        labeler = _make_llm_labeler(cfg)
    elif check_llm_available(_llm_ncfg):
        labeler = _make_llm_labeler(cfg)
    else:
        logger.warning("LLM unavailable (%s) — c-TF-IDF fallback", _llm_ncfg["url"])
        labeler = None

    vectorizer = _build_vectorizer(min_df, ngram_range)
    vectorizer.fit(posts_df["_text_clean"].tolist())
    logger.info("vocabulary: %d terms", len(vectorizer.vocabulary_))

    clustered   = posts_df[posts_df["cluster_id"].notna()].copy()
    cluster_ids = sorted(clustered["cluster_id"].unique())
    logger.info("%d clusters to label", len(cluster_ids))

    cluster_docs = {
        cid: " ".join(clustered[clustered["cluster_id"] == cid]["_text_clean"].tolist())
        for cid in cluster_ids
    }
    ctfidf_clusters = _compute_ctfidf(cluster_docs, vectorizer, top_n_candidates)

    cluster_labels = {}
    cluster_topics = {}
    for cid in cluster_ids:
        mask     = (posts_df["cluster_id"] == cid).values
        keywords = ctfidf_clusters.get(cid, [])
        fallback = " · ".join(keywords[:5]) if keywords else str(cid)

        cluster_topics[cid] = fallback

        if has_embeddings:
            docs = _select_representative_posts(posts_df, all_embeddings, mask, n_representative_posts)
        else:
            docs = posts_df.loc[mask, "_text_clean"].head(n_representative_posts).tolist()

        if not keywords or not docs or labeler is None:
            cluster_labels[cid] = fallback
            logger.info("  [%s] → %s  (c-TF-IDF)", cid, fallback)
            continue

        label = labeler(docs, keywords, fallback)
        cluster_labels[cid] = label
        logger.info("  [%s] → %s", cid, label)

    posts_df["cluster_label"] = posts_df["cluster_id"].map(cluster_labels)
    posts_df["cluster_topic"] = posts_df["cluster_id"].map(cluster_topics)
    posts_df.drop(columns=["_text_clean"], inplace=True)

    out_dir = Path(cfg["storage"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    labels_out = posts_df[["id", "cluster_id", "cluster_label", "cluster_topic"]].copy()
    labels_out.to_parquet(out_dir / "labels.parquet", index=False)
    logger.info("saved: %s/labels.parquet", out_dir)
    logger.info("%d posts with cluster_label", posts_df["cluster_label"].notna().sum())

    return posts_df