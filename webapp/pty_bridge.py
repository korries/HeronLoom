"""Wrap a subprocess in a pseudo-terminal and bridge it to a WebSocket, so
an interactive CLI command (anything that calls input() / safe_input())
behaves in the browser exactly as it would in a real terminal.

Two backends behind the same four-method interface (start / resize / write
/ attach / close), chosen automatically by platform:

* POSIX — `_PosixPtySession`, the stdlib `pty` module with
  `os.set_blocking(fd, False)` + `loop.add_reader()`.
* Windows — `_WindowsPtySession`, `pywinpty` over the real ConPTY API.
  `read()` blocks, so it's driven from `run_in_executor()` rather than an
  event-driven reader. `close()` shells out to `taskkill /T /F` since
  `PtyProcess.terminate()` only reaches the one process winpty holds a
  handle to, and a wrapped process can itself launch children
  (`reload.py resume` re-launching `pipeline.py`).

`fcntl`/`pty`/`termios` and `winpty` are both optional imports; check
`PTY_SUPPORTED` first, `PTY_UNAVAILABLE_REASON` for why not.
"""
from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field

try:
    import fcntl
    import pty
    import struct
    import termios
    _POSIX_PTY_SUPPORTED = True
except ImportError:  # pragma: no cover - Windows
    _POSIX_PTY_SUPPORTED = False

winpty = None
_WINPTY_AVAILABLE = False
if sys.platform == "win32":
    try:
        import winpty  # type: ignore
        _WINPTY_AVAILABLE = True
    except Exception:  # pragma: no cover - pywinpty not installed, or its
        # native _winpty extension failed to load (e.g. missing VC++
        # runtime). Either way, degrade to "unsupported" rather than
        # letting an import-time exception blow up the whole dashboard.
        winpty = None
        _WINPTY_AVAILABLE = False

READ_CHUNK = 65536
DEFAULT_ROWS, DEFAULT_COLS = 30, 120

PTY_SUPPORTED = _POSIX_PTY_SUPPORTED or (sys.platform == "win32" and _WINPTY_AVAILABLE)

if PTY_SUPPORTED:
    PTY_UNAVAILABLE_REASON: str | None = None
elif sys.platform == "win32":
    PTY_UNAVAILABLE_REASON = (
        "Interactive sessions need the 'pywinpty' package on this Windows "
        "machine (pip install pywinpty) — it wraps the same ConPTY API "
        "Windows Terminal and VS Code's integrated terminal use. "
        "Alternatively, run the dashboard under WSL."
    )
else:  # pragma: no cover - some other, truly unsupported platform
    PTY_UNAVAILABLE_REASON = (
        "Interactive terminal sessions need a POSIX pty (Linux/macOS) or, "
        "on Windows, the 'pywinpty' package — neither is available here."
    )


@dataclass
class _PosixPtySession:
    """One live pty-backed child process, plus the plumbing to stream it
    to a WebSocket. One instance per browser terminal panel — sessions
    are never shared or reused across requests."""

    argv: list[str]
    cwd: str
    env: dict[str, str] | None = None

    pid: int = field(default=-1, init=False)
    fd: int = field(default=-1, init=False)
    _closed: bool = field(default=False, init=False)

    def start(self) -> None:
        """Fork the child under a new pty. Call once, before attach()."""
        if not _POSIX_PTY_SUPPORTED:
            raise RuntimeError(
                "Interactive terminal sessions need a POSIX pty and aren't "
                "available on this platform."
            )
        pid, fd = pty.fork()
        if pid == 0:
            # Child: pty.fork() already made the pty slave this process's
            # controlling terminal, so input()/safe_input() see a real tty.
            try:
                os.chdir(self.cwd)
                env = os.environ.copy()
                if self.env:
                    env.update(self.env)
                env.setdefault("TERM", "xterm-256color")
                os.execvpe(self.argv[0], self.argv, env)
            except Exception as exc:  # pragma: no cover - child path
                os.write(2, f"[dashboard] failed to launch: {exc!r}\n".encode())
                os._exit(1)

        # Parent.
        self.pid = pid
        self.fd = fd
        os.set_blocking(self.fd, False)
        self.resize(DEFAULT_ROWS, DEFAULT_COLS)

    def resize(self, rows: int, cols: int) -> None:
        if self.fd < 0 or not rows or not cols:
            return
        with contextlib.suppress(OSError):
            winsize = struct.pack("HHHH", rows, cols, 0, 0)
            fcntl.ioctl(self.fd, termios.TIOCSWINSZ, winsize)

    def write(self, data: bytes) -> None:
        """Forward browser keystrokes to the child's stdin, as if typed at
        a real terminal — how Y/I/Q answers reach pipeline.py's prompts."""
        if self.fd < 0 or self._closed:
            return
        with contextlib.suppress(OSError):
            os.write(self.fd, data)

    async def attach(self, on_output: Callable[[bytes], None]) -> int:
        """Register the pty fd with the running event loop and call
        `on_output(bytes)` for every chunk read, until the pty closes
        (child exited). Returns the child's exit code — best-effort;
        -signum if it died from a signal, -1 if undetermined."""
        loop = asyncio.get_running_loop()
        done: asyncio.Future[None] = loop.create_future()

        def _readable() -> None:
            try:
                data = os.read(self.fd, READ_CHUNK)
            except OSError:
                data = b""
            if data:
                on_output(data)
            else:
                # EOF: the child closed its end of the pty.
                with contextlib.suppress(ValueError):
                    loop.remove_reader(self.fd)
                if not done.done():
                    done.set_result(None)

        loop.add_reader(self.fd, _readable)
        try:
            await done
        finally:
            with contextlib.suppress(ValueError):
                loop.remove_reader(self.fd)

        return await self._wait_exit_code()

    async def _wait_exit_code(self) -> int:
        """Reap the child without blocking the event loop. Gives the OS
        up to ~10s to finish tearing the process down after EOF before
        giving up on an exact code."""
        loop = asyncio.get_running_loop()
        for _ in range(200):
            pid, status = await loop.run_in_executor(None, os.waitpid, self.pid, os.WNOHANG)
            if pid != 0:
                if os.WIFEXITED(status):
                    return os.WEXITSTATUS(status)
                if os.WIFSIGNALED(status):
                    return -os.WTERMSIG(status)
                return -1
            await asyncio.sleep(0.05)
        return -1

    def close(self) -> None:
        """Terminate the child if still alive and release the pty fd.
        Safe to call more than once (e.g. from both a WebSocket disconnect
        handler and a normal exit path)."""
        if self._closed:
            return
        self._closed = True
        if self.pid > 0:
            with contextlib.suppress(ProcessLookupError):
                os.kill(self.pid, signal.SIGTERM)
        if self.fd >= 0:
            with contextlib.suppress(OSError):
                os.close(self.fd)


@dataclass
class _WindowsPtySession:
    """Same interface as `_PosixPtySession`, backed by `winpty.PtyProcess`
    (a direct binding over ConPTY, Windows 10 1809+)."""

    argv: list[str]
    cwd: str
    env: dict[str, str] | None = None

    pid: int = field(default=-1, init=False)
    _proc: object = field(default=None, init=False, repr=False)
    _closed: bool = field(default=False, init=False)

    def start(self) -> None:
        if not _WINPTY_AVAILABLE:
            raise RuntimeError(PTY_UNAVAILABLE_REASON or "pywinpty is not available.")
        env = os.environ.copy()
        if self.env:
            env.update(self.env)
        env.setdefault("TERM", "xterm-256color")
        try:
            # spawn() takes an argv list and quotes it via
            # subprocess.list2cmdline() internally, same as stdlib Popen.
            self._proc = winpty.PtyProcess.spawn(  # type: ignore[union-attr]
                list(self.argv),
                cwd=self.cwd,
                env=env,
                dimensions=(DEFAULT_ROWS, DEFAULT_COLS),
            )
        except Exception as exc:
            raise RuntimeError(f"failed to launch {self.argv[0]!r}: {exc}") from exc
        self.pid = self._proc.pid or -1

    def resize(self, rows: int, cols: int) -> None:
        if self._proc is None or not rows or not cols:
            return
        with contextlib.suppress(Exception):
            self._proc.setwinsize(rows, cols)

    def write(self, data: bytes) -> None:
        if self._proc is None or self._closed:
            return
        with contextlib.suppress(Exception):
            # pywinpty's write() takes str, not bytes — browser keystrokes
            # arrive as UTF-8 text over the WebSocket in either case.
            self._proc.write(data.decode("utf-8", errors="ignore"))

    async def attach(self, on_output: Callable[[bytes], None]) -> int:
        """pywinpty's read() blocks (backed by its own reader thread), so
        it's driven from a thread-pool executor rather than an
        event-driven fd callback."""
        loop = asyncio.get_running_loop()
        while True:
            try:
                chunk = await loop.run_in_executor(None, self._proc.read, READ_CHUNK)
            except EOFError:
                break
            except OSError:
                break
            if chunk:
                on_output(chunk.encode("utf-8", errors="replace"))
        return await self._wait_exit_code()

    async def _wait_exit_code(self) -> int:
        loop = asyncio.get_running_loop()
        for _ in range(200):
            status = self._proc.exitstatus
            if status is not None:
                return int(status)
            if not await loop.run_in_executor(None, self._proc.isalive):
                # Died without an exit status ever being posted.
                return -1
            await asyncio.sleep(0.05)
        return -1

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.pid > 0:
            # terminate() only reaches the one process winpty holds a
            # handle to; taskkill /T walks the whole tree so a nested
            # pipeline.py (via `resume`) is never left orphaned.
            with contextlib.suppress(Exception):
                subprocess.run(
                    ["taskkill", "/T", "/F", "/PID", str(self.pid)],
                    capture_output=True, timeout=5,
                )
        if self._proc is not None:
            with contextlib.suppress(Exception):
                self._proc.close(force=True)


# dashboard.py imports this name and never needs to know which backend it got.
def PtySession(*, argv: list[str], cwd: str, env: dict[str, str] | None = None):
    if sys.platform == "win32":
        return _WindowsPtySession(argv=argv, cwd=cwd, env=env)
    return _PosixPtySession(argv=argv, cwd=cwd, env=env)
