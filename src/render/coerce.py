"""JSON-safe scalar coercion for posts_df / edges_df values."""

import math


def sf(v, d: float = 0.0) -> float:
    """Coerce to float; falls back to d on None/NaN/inf/invalid."""
    try:
        f = float(v if v is not None else d)
        return d if (math.isnan(f) or math.isinf(f)) else f
    except Exception:
        return d


def si(v, d: int = 0) -> int:
    """Coerce to int; falls back to d on None/NaN/inf/invalid."""
    try:
        f = float(v if v is not None else d)
        return d if (math.isnan(f) or math.isinf(f)) else int(f)
    except Exception:
        return d


def ss(v, d: str = "") -> str:
    """Coerce to str; falls back to d for None / pandas null sentinels."""
    if v is None:
        return d
    s = str(v)
    return d if s in ("nan", "None", "NaT", "<NA>", "") else s
