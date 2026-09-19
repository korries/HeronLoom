"""Step 1 — Text embedding via Qwen3 (Ollama).

Embeds post content using a locally served Qwen3-Embedding model
(MRL-trained, native 4096-dimensional).
"""

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from run_store import load_config
from utils.logger import get_logger

logger = get_logger(__name__)


def _resolve_instruct_prompt(cfg: dict) -> str:
    """Resolve the clustering-embedding instruct_prompt.

    An explicit `embedding.instruct_prompt` in advanced.yaml always wins,
    regardless of `content_type` — this is the manual escape hatch for a
    user who wants full control. When left blank (same "blank = auto"
    convention already used for field_mapping in config.yaml), the prompt is
    derived automatically from `content_type`, mirroring how nova.py derives
    unit_word: "social media post" for social_media, "document" for
    document content_type.
    """
    ecfg = cfg.get("embedding", {})
    override = ecfg.get("instruct_prompt", "")

    if override:
        logger.info("[Embedding] instruct_prompt: explicit override from config")
        return override

    content_type = cfg.get("content_type", "social_media")
    unit = "social media post" if content_type == "social_media" else "document"
    auto_prompt = f"Identify the topic of this {unit}"
    logger.info(
        "[Embedding] instruct_prompt: auto-derived from content_type=%s -> %r",
        content_type, auto_prompt,
    )
    return auto_prompt


class EmbeddingPipeline:
    """Batched embedding pipeline backed by Ollama and a two-level cache.

    The same pipeline builds two independent embedding stores from one
    config: a clustering store (``embeddings.parquet``, optionally
    instruction-wrapped and MRL-truncated) and a retrieval store
    (``retrieval_output``, always raw text, consumed by the search
    module's document-side index). The two never share a parquet file
    or a config sidecar, so a change to one can never desync the other.

    Attributes:
        cfg: Parsed content of the pipeline's YAML config file.
        batch_size: Number of texts sent to Ollama per request.
        model_name: Name of the Ollama embedding model to call.
        ollama_url: URL of the Ollama ``/api/embed`` endpoint.
        truncate: Whether to apply MRL prefix truncation to raw_dim.
        raw_dim: Target embedding dimensionality after truncation.
        instruct_prompt: Instruction prefix applied to the clustering
            embedding only; the retrieval embedding always ignores it.
            Resolved by _resolve_instruct_prompt(): explicit
            embedding.instruct_prompt in advanced.yaml overrides everything;
            otherwise auto-derived from `content_type`
            ("social_media" | "document").
        num_ctx: Context window passed to the Ollama request options.
        retrieval_output: Filename of the retrieval-only parquet store.
    """

    def __init__(self, config_path: str = "config.yaml") -> None:
        """Load config_path and set up the local embedding cache.

        Args:
            config_path: Path to the pipeline's YAML config. Read once;
                the resolved values are cached as instance attributes.
        """
        self.cfg = load_config(config_path)

        ecfg = self.cfg["embedding"]
        self.batch_size = ecfg.get("batch_size", 32)
        self.model_name = ecfg.get("ollama_model", "qwen3-embedding:8b")
        self.ollama_url = ecfg.get("ollama_url", "http://localhost:11434/api/embed")
        self.truncate = ecfg.get("truncate", True)
        self.raw_dim = ecfg.get("raw_dimensions", 1024)
        self.instruct_prompt = _resolve_instruct_prompt(self.cfg)
        self.num_ctx = ecfg.get("num_ctx", 8192)
        self.retrieval_output = ecfg.get(
            "retrieval_parquet", "embeddings_retrieval.parquet"
        )

        cache_dir = (
            Path(self.cfg.get("storage", {}).get("processed_dir", "data/processed"))
            / "embedding_cache"
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            from diskcache import Cache

            self._local_cache: object | None = Cache(str(cache_dir))
        except ImportError:
            self._local_cache = None
            logger.warning(
                "diskcache not installed — embeddings won't be locally cached "
                "(every post re-embedded via Ollama on every run). pip install diskcache"
            )


    def _cache_key(self, text: str) -> str:
        raw = f"{self.model_name}:{text}"
        return f"emb:{hashlib.sha256(raw.encode()).hexdigest()[:16]}"

    def _get_cache(self, text: str) -> np.ndarray | None:
        if self._local_cache is None:
            return None
        try:
            data = self._local_cache.get(self._cache_key(text))
            return pickle.loads(data) if data is not None else None
        except Exception:
            return None

    def _set_cache(self, text: str, embedding: np.ndarray) -> None:
        if self._local_cache is None:
            return
        try:
            self._local_cache.set(
                self._cache_key(text),
                pickle.dumps(embedding, protocol=pickle.HIGHEST_PROTOCOL),
            )
        except Exception as exc:
            logger.warning("DiskCache write failed — %s", exc)


    @staticmethod
    def _sanitize_text(t) -> str:
        """Coerce to str and strip bytes/chars known to break Ollama's tokenizer.

        Handles NUL bytes and other C0 control characters (aside from
        newline/tab) that can slip in from scraped HTML/JSON sources and
        trigger a 400 from the embed endpoint.
        """
        if not isinstance(t, str):
            t = "" if t is None else str(t)
        t = t.replace("\x00", "")
        return "".join(ch for ch in t if ch in ("\n", "\t") or ord(ch) >= 32)

    def _truncate_and_renorm(self, arr: np.ndarray) -> np.ndarray:
        """Prefix-truncate to raw_dim and L2-renormalize (MRL convention)."""
        arr = arr[..., : self.raw_dim].copy()
        norms = np.linalg.norm(arr, axis=-1, keepdims=True)
        norms = np.where(norms == 0, 1e-10, norms)
        return arr / norms

    def _embed_ollama_request(self, texts: list[str]) -> np.ndarray:
        """Single raw HTTP call to the Ollama embed endpoint (no retry/bisection).

        Raises:
            ConnectionError: Ollama is not reachable at ``ollama_url``.
            RuntimeError: Ollama responded with a non-2xx status or an
                unusable body. The exception message includes Ollama's
                actual response body, unlike a bare ``raise_for_status()``.
        """
        payload = {
            "model": self.model_name,
            "input": texts,
            "options": {"num_ctx": self.num_ctx},
        }
        try:
            r = requests.post(self.ollama_url, json=payload, timeout=120)
        except requests.exceptions.ConnectionError as exc:
            raise ConnectionError(
                f"[Embedding] Cannot reach Ollama at {self.ollama_url} — "
                f"check: ollama run {self.model_name}"
            ) from exc

        if r.status_code != 200:
            # Surface Ollama's actual error body instead of swallowing it
            # behind a generic "400 Client Error: Bad Request" message.
            try:
                body = r.json()
                detail = body.get("error", body)
            except ValueError:
                detail = r.text[:1000]
            raise RuntimeError(
                f"[Embedding] Ollama HTTP {r.status_code} for a batch of "
                f"{len(texts)} text(s): {detail}"
            )

        data = r.json()
        if "embeddings" not in data:
            raise RuntimeError(
                f"Ollama response missing 'embeddings' key: {list(data.keys())}"
            )
        return np.array(data["embeddings"], dtype=np.float32)

    def _embed_ollama_batch(self, texts: list[str]) -> np.ndarray:
        """Call the Ollama embed endpoint for one batch, with fault isolation.

        On failure, the batch is bisected recursively so the exact
        offending text is identified instead of failing the whole batch
        opaquely. A batch of size 1 that fails raises a RuntimeError
        naming the text's length and a preview of its content.

        Returns:
            Float32 array of shape (len(texts), native_dim).

        Raises:
            ConnectionError: Ollama is not reachable at ``ollama_url``.
            RuntimeError: Ollama responded but the call otherwise failed
                (bad payload, missing ``embeddings`` key, offending text
                identified for batches of size 1, etc.).
        """
        try:
            return self._embed_ollama_request(texts)
        except ConnectionError:
            raise
        except Exception as exc:
            if len(texts) == 1:
                t = texts[0]
                raise RuntimeError(
                    f"[Embedding] Ollama rejected this single text "
                    f"(len={len(t)} chars, ~{len(t) // 4} tokens est.): {exc}\n"
                    f"[Embedding] Preview: {t[:300]!r}"
                ) from exc
            mid = len(texts) // 2
            logger.warning(
                "Batch of %d texts failed (%s) — bisecting to isolate the culprit",
                len(texts), exc,
            )
            first = self._embed_ollama_batch(texts[:mid])
            second = self._embed_ollama_batch(texts[mid:])
            return np.concatenate([first, second], axis=0)


    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts with cache-first strategy.

        Args:
            texts: raw text strings to embed.

        Returns:
            Float32 array of shape (n, dim).

        Raises:
            ConnectionError: Ollama is unreachable for any cache-miss text.
            RuntimeError: Ollama returned an unusable response.
        """
        if not texts:
            return np.array([], dtype=np.float32)

        results: list[np.ndarray | None] = [None] * len(texts)
        missing_indices: list[int] = []
        missing_texts: list[str] = []

        for i, text in enumerate(texts):
            cached = self._get_cache(text)
            if cached is not None:
                results[i] = cached
            else:
                missing_indices.append(i)
                missing_texts.append(text)

        n_cached = len(texts) - len(missing_texts)
        n_missing = len(missing_texts)

        if missing_texts:
            n_batches = (n_missing + self.batch_size - 1) // self.batch_size
            print(
                f"[Embedding] cache={n_cached}  ollama={n_missing}  batches={n_batches}  batch_size={self.batch_size}"
            )
            for batch_idx, i in enumerate(range(0, n_missing, self.batch_size)):
                batch_texts = missing_texts[i : i + self.batch_size]
                batch = self._embed_ollama_batch(batch_texts)
                for local_idx, emb in enumerate(batch):
                    global_idx = missing_indices[i + local_idx]
                    self._set_cache(texts[global_idx], emb)
                    results[global_idx] = emb
                done = batch_idx + 1
                pct = done / n_batches
                filled = int(30 * pct)
                bar = "█" * filled + "░" * (30 - filled)
                n_done = min(i + self.batch_size, n_missing)
                print(
                    f"\r  [{bar}] {pct*100:5.1f}%  {n_done}/{n_missing} posts",
                    end="",
                    flush=True,
                )
            print()
        else:
            print(f"[Embedding] cache={n_cached}  ollama=0  (fully cached)")

        out = np.array(results, dtype=np.float32)
        native_dim = out.shape[1]

        if self.truncate and native_dim > self.raw_dim:
            out = self._truncate_and_renorm(out)
            print(
                f"[Embedding] dim: {native_dim}d → {out.shape[1]}d  (MRL prefix truncation + L2-norm)"
            )
        else:
            print(f"[Embedding] dim: {native_dim}d  (no truncation)")

        return out


    def _embedding_config_matches(self, out_dir: Path) -> bool:
        """Return True if embedding_config.json matches current parameters."""
        path = out_dir / "embedding_config.json"
        if not path.exists():
            return False
        with open(path) as f:
            saved = json.load(f)
        return (
            saved.get("model_name") == self.model_name
            and saved.get("truncate") == self.truncate
            and saved.get("raw_dimensions") == self.raw_dim
            and saved.get("instruct_prompt") == self.instruct_prompt
        )

    def _save_embedding_config(self, out_dir: Path) -> None:
        with open(out_dir / "embedding_config.json", "w") as f:
            json.dump(
                {
                    "model_name": self.model_name,
                    "truncate": self.truncate,
                    "raw_dimensions": self.raw_dim,
                    "instruct_prompt": self.instruct_prompt,
                },
                f,
            )


    def process_dataframe(self, posts_df: pd.DataFrame) -> pd.DataFrame:
        """Embed all posts and attach an ``embedding_raw`` column.

        Loads from parquet if ``embeddings.parquet`` exists and its config
        matches. Otherwise runs DiskCache/Ollama.

        Args:
            posts_df: must contain ``id`` and ``content`` columns.

        Returns:
            Input frame with an added ``embedding_raw`` column (list of float32).

        Raises:
            ConnectionError: Ollama is unreachable and no cached or
                parquet embedding is available for at least one post.
            RuntimeError: Ollama returned an unusable response.
        """
        out_dir = Path(self.cfg["storage"]["processed_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        emb_path = out_dir / "embeddings.parquet"

        if emb_path.exists() and self._embedding_config_matches(out_dir):
            try:
                parquet_df = pd.read_parquet(emb_path)
                if "embedding_raw" not in parquet_df.columns:
                    raise ValueError("embedding_raw column missing from parquet")
                merged = posts_df.merge(
                    parquet_df[["id", "embedding_raw"]], on="id", how="left"
                )
                n_missing = merged["embedding_raw"].isna().sum()
                if n_missing > 0:
                    raise ValueError(
                        f"{n_missing}/{len(merged)} posts have no embedding in the "
                        f"parquet (IDs don't match)"
                    )
                posts_df = merged
                sample_emb = posts_df["embedding_raw"].iloc[0]
                emb_dim = len(sample_emb) if hasattr(sample_emb, "__len__") else "?"
                print(
                    f"[Embedding] {len(posts_df)} embeddings loaded from parquet  "
                    f"{emb_dim}D  model={self.model_name}"
                )
                return posts_df
            except Exception as exc:
                logger.warning("parquet load failed (%s) — falling back to Ollama", exc)

        posts_df = self._embed_dataframe(posts_df)
        posts_df[["id", "embedding_raw"]].to_parquet(emb_path, index=False)
        self._save_embedding_config(out_dir)
        return posts_df

    def _embed_dataframe(self, posts_df: pd.DataFrame) -> pd.DataFrame:
        """Embed posts for clustering, wrapping them in instruct_prompt if set.

        Args:
            posts_df: must contain ``id`` and ``content`` columns.

        Returns:
            Copy of posts_df with an added ``embedding_raw`` column.

        Raises:
            ValueError: Ollama returned a different embedding count than
                the number of input posts.
        """
        raw_texts = [self._sanitize_text(t) for t in posts_df["content"].fillna("").tolist()]
        n = len(raw_texts)

        if self.instruct_prompt:
            prefix = f"Instruct: {self.instruct_prompt}\nQuery:"
            texts = [f"{prefix}{t}" for t in raw_texts]
            print(f"[Embedding] model={self.model_name}  n={n}  instruct_prompt=<active>")
        else:
            texts = raw_texts
            print(f"[Embedding] model={self.model_name}  n={n}  instruct_prompt=<none>")

        embeddings = self.embed_texts(texts)
        if embeddings.shape[0] != n:
            raise ValueError(
                f"Embedding count mismatch: got {embeddings.shape[0]}, expected {n}"
            )

        posts_df = posts_df.copy()
        posts_df["embedding_raw"] = list(embeddings.astype(np.float32))
        return posts_df


    def _retrieval_config_matches(self, out_dir: Path) -> bool:
        """Return True if retrieval_embedding_config.json matches current parameters."""
        path = out_dir / "retrieval_embedding_config.json"
        if not path.exists():
            return False
        with open(path) as f:
            saved = json.load(f)
        return (
            saved.get("model_name") == self.model_name
            and saved.get("truncate") == self.truncate
            and saved.get("raw_dimensions") == self.raw_dim
        )

    def _save_retrieval_config(self, out_dir: Path) -> None:
        with open(out_dir / "retrieval_embedding_config.json", "w") as f:
            json.dump(
                {
                    "model_name": self.model_name,
                    "truncate": self.truncate,
                    "raw_dimensions": self.raw_dim,
                    # Retrieval documents are never instruction-wrapped.
                    "instruct_prompt": None,
                },
                f,
            )

    def _embed_dataframe_retrieval(self, posts_df: pd.DataFrame) -> pd.DataFrame:
        """Embed posts for retrieval — raw text, no instruction wrapper.

        Per the official Qwen3-Embedding usage (Qwen/HF convention), the
        document side gets no instruction wrapper, only raw text; only
        the query side is wrapped, with "Instruct: {task}\\nQuery:{query}".
        So this method always embeds raw post content and never applies
        ``instruct_prompt`` — that setting only affects the clustering
        embedding built by ``_embed_dataframe``. The matching query-side
        wrapper lives in the search step's ``embed_query()``.

        Uses the same ``embed_texts`` cache-first path as the clustering
        embedding, so an unmodified post's text hashes to the same cache
        key and is never re-sent to Ollama; a new or edited post gets a
        new hash and is embedded automatically.

        Args:
            posts_df: must contain ``id`` and ``content`` columns.

        Returns:
            DataFrame with just ``id`` and ``embedding`` columns.

        Raises:
            ValueError: Ollama returned a different embedding count than
                the number of input posts.
        """
        raw_texts = [self._sanitize_text(t) for t in posts_df["content"].fillna("").tolist()]
        n = len(raw_texts)
        print(
            f"[Embedding][Retrieval] model={self.model_name}  n={n}  instruct_prompt=<none — document side>"
        )

        embeddings = self.embed_texts(raw_texts)
        if embeddings.shape[0] != n:
            raise ValueError(
                f"Retrieval embedding count mismatch: got {embeddings.shape[0]}, expected {n}"
            )

        out = posts_df[["id"]].copy()
        out["embedding"] = list(embeddings.astype(np.float32))
        return out

    def process_dataframe_retrieval(self, posts_df: pd.DataFrame) -> pd.DataFrame:
        """Build (or reuse) the retrieval-only embedding store.

        Mirrors ``process_dataframe``'s reuse logic exactly (config-match
        check, then a merge-by-id; any post missing from the parquet
        forces a full recompute — cheap in practice because
        ``embed_texts``'s DiskCache still skips every unmodified post).
        Writes/reads its own parquet and config sidecar only; never
        touches ``embeddings.parquet`` or ``embedding_config.json``.

        Args:
            posts_df: must contain ``id`` and ``content`` columns.

        Returns:
            DataFrame with ``id`` and ``embedding`` columns — the store
            the search step consults for dense vector search.

        Raises:
            ConnectionError: Ollama is unreachable and no cached or
                parquet embedding is available for at least one post.
            RuntimeError: Ollama returned an unusable response.
        """
        out_dir = Path(self.cfg["storage"]["processed_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        retrieval_path = out_dir / self.retrieval_output

        if retrieval_path.exists() and self._retrieval_config_matches(out_dir):
            try:
                parquet_df = pd.read_parquet(retrieval_path)
                if "embedding" not in parquet_df.columns:
                    raise ValueError("embedding column missing from retrieval parquet")
                merged = posts_df[["id"]].merge(
                    parquet_df[["id", "embedding"]], on="id", how="left"
                )
                n_missing = merged["embedding"].isna().sum()
                if n_missing > 0:
                    raise ValueError(
                        f"{n_missing}/{len(merged)} posts have no retrieval embedding "
                        f"in the parquet (new posts or edited content)"
                    )
                print(
                    f"[Embedding][Retrieval] {len(merged)} embeddings loaded from parquet  model={self.model_name}"
                )
                return merged
            except Exception as exc:
                logger.warning(
                    "[Retrieval] parquet load failed (%s) — falling back to Ollama", exc
                )

        result = self._embed_dataframe_retrieval(posts_df)
        result.to_parquet(retrieval_path, index=False)
        self._save_retrieval_config(out_dir)
        return result


def run(posts_df: pd.DataFrame, config_path: str = "config.yaml") -> pd.DataFrame:
    """Module entry point called by the pipeline orchestrator.

    Args:
        posts_df: raw posts with at least ``id`` and ``content`` columns.
        config_path: path to ``config.yaml``.

    Returns:
        Input frame extended with an ``embedding_raw`` column.

    Raises:
        ConnectionError: Ollama is unreachable and no cached or parquet
            embedding is available for at least one post.
        RuntimeError: Ollama returned an unusable response.
    """
    return EmbeddingPipeline(config_path).process_dataframe(posts_df)