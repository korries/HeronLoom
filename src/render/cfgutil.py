"""config.yaml section accessors, with logged fallback on missing/invalid keys."""

from utils.logger import get_logger

log = get_logger(__name__)


def cfg_get(section: dict, key: str, default, cast=None):
    """Read section[key]; log and fall back to default if missing/invalid."""
    val = section.get(key)
    if val is None:
        log.debug("Fallback %s -> %s (missing key in config.yaml)", key, default)
        return default
    try:
        return cast(val) if cast is not None else val
    except (TypeError, ValueError):
        log.warning("Fallback %s -> %s (invalid value: %r)", key, default, val)
        return default


def cfg_get_silent(section: dict, key: str, default, cast=None):
    """Like cfg_get, but silent — for keys that are legitimately absent."""
    val = section.get(key)
    if val is None:
        return default
    try:
        return cast(val) if cast is not None else val
    except (TypeError, ValueError):
        return default
