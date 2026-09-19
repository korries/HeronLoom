# Contributing to HeronLoom

Thanks for considering a contribution.

## Development setup

```bash
git clone https://github.com/Korries/HeronLoom.git
cd HeronLoom

python -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

```bash
python -m pip install -U pip
python -m pip install -r requirements-dev.txt
cp .env.example .env
```

That installs the runtime dependencies plus `pytest` and `ruff`. Nothing else is
installed: HeronLoom runs from the folder itself. Run its commands from the
repository root, and see
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md#source-layout-and-imports) for how
`src/` reaches the import path.

Before opening a PR:

```bash
ruff check .
pytest
```

Both run in CI on Ubuntu (3.11, 3.13) and Windows (3.11).

## Code style

- Python 3.11+, type hints on new public functions.
- Match the existing docstring style (see `pipeline.py`) for new stage modules.
- Stage modules go in `src/`, entry points stay at the repository root.
- Use `from utils.logger import get_logger` directly — no `try`/`except
  ImportError` guard around it. The path is guaranteed at runtime and in tests.

## Adding a dependency

Pin new dependencies in `requirements.txt` (or `requirements-dev.txt` for
tooling only). CI checks every dependency's licence with `pip-licenses`
against the allow list in `.github/workflows/ci.yml`. If a new dependency's
licence isn't already on that list:

- Confirm it's compatible with the AGPL-3.0 before adding it.
- Add the exact licence string `pip-licenses --partial-match` reports for it
  to the allow list.
- If it carries any obligation beyond attribution (a copyleft licence, for
  example), record it in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) in
  the same PR.

`requirements.txt` must not pull in PyTorch, directly or through another
package: it stays an optional install (see the
[README](README.md#3-install-dependencies)).

## Reporting a security issue

See [SECURITY.md](SECURITY.md) — please don't open a public issue for a
vulnerability.
