"""Logging configuration — console (INFO+) and rotating file (DEBUG+) handlers.

The file handler is shared across *processes*, not just modules: dashboard.py
(the web server, long-lived) and whichever of pipeline.py / reload.py is
running as a real subprocess of it (see pty_bridge.py / runs_api.py) can
each have logs/YYYY-MM-DD.log open at the same time — e.g. the dashboard's
own `_watch_loop` keeps logging while a `reload.py resume ...` subprocess
it launched is itself shelling out to `pipeline.py`. Three independent OS
processes, one shared file.

Stdlib `RotatingFileHandler` isn't safe for that: each process's handler
tracks "should I rotate?" from its own file-size check, so two processes
can each decide to rotate at once, and a process that still holds the
pre-rotation file descriptor open keeps writing into what's now
`....log.1` — invisible, and eventually deleted once that backup itself
rotates out. `concurrent-log-handler` (`pip install concurrent-log-handler`)
is a drop-in, file-locking replacement built for exactly this multi-process
case. We fall back to the stdlib handler (with a one-time warning) if it
isn't installed, so a missing dependency degrades instead of crashing.
"""

import logging
import os
import sys
import warnings
from datetime import datetime
from logging.handlers import RotatingFileHandler as _StdlibRotatingFileHandler

try:
    from concurrent_log_handler import ConcurrentRotatingFileHandler as _FileHandlerClass
    _CONCURRENT_SAFE = True
except ImportError:  # pragma: no cover - dependency not installed yet
    warnings.warn(
        "concurrent_log_handler not installed (pip install concurrent-log-handler) — "
        "falling back to the stdlib RotatingFileHandler. That handler is NOT safe "
        "when dashboard.py and a pipeline.py/reload.py subprocess it launched both "
        "have logs/*.log open at once, which happens here on every Resume/Relabel/"
        "Sync/New run: rotations can race and log lines can be silently dropped. "
        "Install the package to fix this properly.",
        stacklevel=2,
    )
    _FileHandlerClass = _StdlibRotatingFileHandler
    _CONCURRENT_SAFE = False

# Console color: colorlog (pip install colorlog) is the de-facto standard for
# this in the Python ecosystem — used by Errbot, Pythran, zenlog, and most
# other projects that colorize logging output — rather than a hand-rolled
# ANSI formatter. It also bundles + auto-initializes `colorama` on Windows,
# so there's no separate Windows console-mode handling to write ourselves.
try:
    import colorlog
    _COLORLOG_AVAILABLE = True
except ImportError:  # pragma: no cover - dependency not installed yet
    warnings.warn(
        "colorlog not installed (pip install colorlog) — falling back to "
        "plain, uncolored console output. Install the package to get "
        "per-level console colors.",
        stacklevel=2,
    )
    _COLORLOG_AVAILABLE = False

_HERE      = os.path.dirname(os.path.abspath(__file__))
_ROOT      = os.path.dirname(_HERE)
LOG_DIR    = os.path.join(_ROOT, "logs")
_ROOT_NAME = "heronloom"

# --- NOTICE: a level between INFO (20) and WARNING (30) ---------------------
# For actionable tips/hints that aren't warnings (nothing's wrong) but
# shouldn't blend into routine INFO chatter either — e.g. "you're running
# sequential on 113 clusters, here's how to speed that up". Modeled on
# GitHub Actions' own `::notice::` annotation. colorlog supports custom
# levels out of the box via logging.addLevelName (see its docs).
NOTICE = 25
logging.addLevelName(NOTICE, "NOTICE")


def _notice(self: logging.Logger, message, *args, **kwargs) -> None:
    if self.isEnabledFor(NOTICE):
        self._log(NOTICE, message, args, **kwargs)


logging.Logger.notice = _notice  # available on every Logger, incl. get_logger()'s

# DEBUG/INFO/WARNING/ERROR/CRITICAL are colorlog's own README "Examples"
# block, verbatim (github.com/borntyping/python-colorlog):
#   from colorlog import ColoredFormatter
#   formatter = ColoredFormatter(..., log_colors={
#       "DEBUG": "cyan", "INFO": "green", "WARNING": "yellow",
#       "ERROR": "red", "CRITICAL": "red,bg_white",
#   })
# This is NOT the same as colorlog.default_log_colors (the value colorlog
# falls back to if you pass no log_colors at all) — that one differs on
# two levels: DEBUG is "white" and CRITICAL is "bold_red" there. Checked
# directly against the installed package:
#   >>> import colorlog; colorlog.default_log_colors
#   {'DEBUG': 'white', 'INFO': 'green', 'WARNING': 'yellow',
#    'ERROR': 'red', 'CRITICAL': 'bold_red'}
# We're using the README's cyan / red-on-white instead, since that's the
# pairing most CLI tools that build on colorlog actually reach for.
# "red,bg_white" (comma-joined) is colorlog's real syntax for combining
# a foreground and background code — see escape_codes.parse_colors(),
# which splits on "," and concatenates each piece's escape code.
#
# NOTICE isn't a colorlog level at all (it's the custom one defined
# above), so colorlog has no opinion on its color. The style is borrowed
# from `coloredlogs` (xolox/python-coloredlogs), whose DEFAULT_LEVEL_STYLES
# ships an actual notice level styled magenta:
#   >>> import coloredlogs; coloredlogs.DEFAULT_LEVEL_STYLES['notice']
#   {'color': 'magenta'}
#
# NOT using colorlog's own "purple"/"magenta" name for this (ANSI code 35,
# one of the base 16 terminal colors) — forcing the formatter to emit its
# actual bytes confirmed it does send the textbook-correct \x1b[35m for
# that name, but 35 is one of the 8 base slots that terminal color themes
# routinely reassign, and several ship a redder/pinker hue in that slot —
# which is exactly what was showing up as "red" here. It's a terminal
# theme rendering choice, not a bug in this file, but it makes the base
# color unreliable across machines/themes. "fg_201" instead addresses a
# fixed 256-color-palette slot (xterm 256-color code 201 = RGB 255,0,255,
# pure magenta) that themes essentially never remap, so it reads the same
# regardless of terminal theme. Confirmed present in colorlog's own table:
# colorlog.escape_codes.escape_codes["fg_201"] == "\x1b[38;5;201m".
_LOG_COLORS = {
    "DEBUG":    "cyan",
    "INFO":     "green",
    "NOTICE":   "fg_201",  # fixed 256-color magenta — see note above on why not "purple"
    "WARNING":  "yellow",
    "ERROR":    "red",
    "CRITICAL": "red,bg_white",
}


def setup_logger(name: str = _ROOT_NAME, level: int = logging.DEBUG) -> logging.Logger:
    """Configure and return the named logger with console + rotating-file handlers.

    Idempotent: calling twice with the same name returns the existing logger
    without adding duplicate handlers.

    Args:
        name: logger name (default ``"heronloom"``); sub-loggers inherit via hierarchy.
        level: minimum level for the file handler (default DEBUG);
            console handler is always INFO+.

    Returns:
        Configured Logger instance.
    """
    os.makedirs(LOG_DIR, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # don't bubble to root logger

    if logger.handlers:
        return logger

    # File: full detail — module, function, line
    file_fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Console: compact — time, level, message. colorlog colors the whole
    # line by level (see _LOG_COLORS above); it auto-detects TTY vs
    # redirected/piped output and respects NO_COLOR/FORCE_COLOR itself, so
    # there's nothing extra to check here. Falls back to plain text if the
    # package isn't installed.
    if _COLORLOG_AVAILABLE:
        console_fmt = colorlog.ColoredFormatter(
            "%(log_color)s[%(asctime)s] %(levelname)-8s%(reset)s %(message)s",
            datefmt="%H:%M:%S",
            log_colors=_LOG_COLORS,
            stream=sys.stdout,  # lets colorlog auto-detect TTY vs redirected/piped output
        )
    else:
        console_fmt = logging.Formatter(
            "[%(asctime)s] %(levelname)-8s %(message)s",
            datefmt="%H:%M:%S",
        )

    # Rotating file — DEBUG+, 10 MB, 5 backups. ConcurrentRotatingFileHandler
    # when available (see the module docstring for why); each process that
    # calls setup_logger() creates and locks its own handler instance on the
    # same path — that's the supported usage, not a workaround.
    log_file     = os.path.join(LOG_DIR, datetime.now().strftime("%Y-%m-%d") + ".log")
    file_handler = _FileHandlerClass(
        log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_fmt)

    # Console — UTF-8 safe on Windows
    if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_fmt)
    # Tagged explicitly rather than left to isinstance() in
    # set_console_level() below: logging.FileHandler *is* a StreamHandler
    # subclass, so "isinstance(h, StreamHandler)" alone can't tell the file
    # handler and the console handler apart — and now that file_handler may
    # be ConcurrentRotatingFileHandler rather than the stdlib
    # RotatingFileHandler, "and not isinstance(h, RotatingFileHandler)"
    # (the previous check) would stop excluding it too.
    console_handler._is_console = True

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def set_console_level(level: int, name: str = _ROOT_NAME) -> None:
    """Change the console handler's level on an already-configured logger.

    File handler is untouched — it stays at DEBUG regardless, so nothing is
    ever lost from the log file. This only affects what's echoed to stdout
    for the current run.

    Args:
        level: e.g. logging.DEBUG to show debug output in console.
        name: root logger name whose console handler should be adjusted
            (default ``"heronloom"``).
    """
    logger = logging.getLogger(name)
    for handler in logger.handlers:
        if getattr(handler, "_is_console", False):
            handler.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the ``"heronloom"`` hierarchy.

    Calls setup_logger() automatically if the root logger has no handlers yet
    (module imported before main() runs).

    Args:
        name: typically __name__ of the calling module, e.g. ``"adr_refiner"``.

            Exception — the three entry-point scripts (dashboard.py,
            pipeline.py, reload.py) must NOT use __name__: each can be run
            directly (``python dashboard.py``, ``python pipeline.py``,
            ``python reload.py``), and Python sets __name__ = "__main__" for
            whichever one that is. get_logger(__name__) there would collapse
            all three into the same "heronloom.__main__" logger, so a line in
            logs/*.log couldn't be traced back to which of the three actually
            produced it. They pass a fixed name instead:
            get_logger("dashboard") / get_logger("pipeline") /
            get_logger("reload"). Every other module keeps using __name__ as
            usual — it's only ever imported, never run directly, so there's
            no collision to begin with.

    Returns:
        Logger named ``"heronloom.<name>"``.
    """
    root = logging.getLogger(_ROOT_NAME)
    if not root.handlers:
        setup_logger()
    return logging.getLogger(f"{_ROOT_NAME}.{name}")
