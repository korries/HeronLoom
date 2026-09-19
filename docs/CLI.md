# CLI reference

Four entry points: `pipeline.py` (compute), `reload.py` (browse and replay), `search.py` (analysis), `dashboard.py` (web UI). All four read `config/config.yaml` natively, merged with `config/advanced.yaml`; there is no flag to point them at a different file.

---

## `pipeline.py`

```
python pipeline.py [options]
```

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--input` | path | `data/raw/` | Raw input directory. Reads `.json`, `.csv`, `.tsv`, `.txt`, `.md`, `.pdf`; a `.txt`, `.md`, or `.pdf` file becomes one post. |
| `--limit` | int | — | Truncate input to the first N posts, for a quick smoke test. |
| `--resume-from` | stage name | — | Force-resume from a specific stage, bypassing checkpoint detection. See [stage semantics](#--resume-from-stage-semantics). |
| `--run-mode` | `full` \| `clustering_only` | *(config value)* | Override `run_mode` for this run. `clustering_only` skips Nova and ADEPT only; edges and 3D layout still run. |
| `--content-type` | `social_media` \| `document` | *(config value)* | Override for a **new** run only. Baked into the frozen config snapshot at creation, so it has no effect on `--resume-from` or a resumed active run. |
| `--use-k-cache` | flag | off | Load `k_optimal` from `k_cache.pkl` and skip the K-estimation stage entirely. |
| `--set-k` | int | — | Force `k=N` through the GMM-BIC sweep (equivalent to `--k-range N N`). Persists K and seed labels to `k_cache.pkl`. |
| `--k-range` | `MIN MAX` | — | Debug: bypass the HDBSCAN pivot and sweep GMM-BIC directly over `[MIN, MAX]`. |
| `--k-fast` | flag | off | Not recommended. Skip the GMM-BIC sweep and use HDBSCAN's own cluster count as K (fast, less refined). Same as `clustering.k_estimation_method: hdbscan`. Ignored when `--set-k`, `--k-range`, or `--use-k-cache` already bypasses K estimation; a warning is logged. |
| `--skip-naming` | flag | off | Skip the naming stage; the label column is left `None`. |
| `--force-nova` | flag | off | Ignore the Nova reuse cache (`nova_assignments.parquet` / `nova_edges.parquet` at the run root) and recompute Nova from scratch. This is **not** the `stages/nova/` resume checkpoint; see [`ARCHITECTURE.md`](ARCHITECTURE.md#what-a-run-folder-contains). |
| `--render` | `none` \| `threejs` | `threejs` | Write `render.html` at the end of the run. Never opens a browser tab, since this also runs under cron. View it afterward with `reload.py show <run_id> --render threejs`. |
| `--fresh` | flag | off | Always start a brand-new run, no prompt, never resume. Any in-progress active run is left untouched and still resumable later. |
| `-y`, `--yes` | flag | off | Auto-confirm the resume prompt, resuming an in-progress active run without asking. |
| `--debug` | flag | off | Show DEBUG-level logs in the console. The log *file* always captures DEBUG regardless. |

A render failure does not invalidate the run: the data is already marked `complete`, the error is logged, and the process exits non-zero so automation notices. Re-render with `reload.py show`.

#### `--resume-from` stage semantics

- `nova` — reload posts and `nova_edges` exactly as they stood right after Nova finished, then jump into ADEPT. No `--force-nova` needed.
- `adept` — reload posts and `adept_edges` exactly as they stood right after ADEPT finished, then jump into edges.
- `adr`, `naming`, `edges`, `fa2` — reload that stage's own checkpoint and continue from there.

**Examples**

```bash
python pipeline.py
python pipeline.py --input data/raw/ --limit 500
python pipeline.py --run-mode clustering_only
python pipeline.py --set-k 42 --skip-naming
python pipeline.py --k-fast
python pipeline.py --resume-from nova
```

---

## `reload.py`

```
python reload.py <command> [options]
```

| Command | Positional args | Key options | What it does |
|---|---|---|---|
| `list` | — | `--trash` | List every active run, or every trashed run. |
| `show` | `RUN_ID` | `--render {none,threejs}` (default `none`), `--no-browser` | Render a run's saved state, no recompute. `--no-browser` writes `render.html` without opening a tab (used internally by the dashboard). |
| `open` | `RUN_ID` | — | Open a run's existing `render.html`. No recompute; fails if it does not exist yet. |
| `search` | `RUN_ID?` `QUERY` | `--run-mode`, `--no-decompose` | Ask a question against a run's saved output. `RUN_ID` is optional; omit it to use the active run. Thin wrapper around `search.py`. |
| `resume` | `RUN_ID` | `--render`, `--resume-from` (`run_store.STAGE_ORDER`), + [override flags](#override-flags) | Continue a run from its furthest checkpoint, or force a specific stage, e.g. `--resume-from naming --run-mode full` to add Nova and ADEPT to a run that finished `clustering_only`. Works even on an already-`complete` run. |
| `restart` | `RUN_ID` | `--render`, + [override flags](#override-flags) | Full re-run under a **new** `run_id`, always from stage 1. No `--resume-from`, since there is nothing to resume. |
| `relabel` | `RUN_ID` | `--force-skip-llm` | Redo naming only, in place. Uses the LLM if reachable; `--force-skip-llm` forces c-TF-IDF-only keywords, mirroring the naming stage's own automatic fallback. |
| `sync-models` | `RUN_ID` | `--dry-run` | Patch a run's frozen `config_snapshot.yaml` with the *live* `config.yaml`'s module routing only (`models.heavy` / `models.light`; `nova` / `adept` / `naming` → `model_slot`; and `nova` / `adept` → `parallel_clusters`). `--dry-run` prints the diff without writing. |
| `trash` | `RUN_ID` | — | Move a run to `runs/_trash/` (reversible). |
| `restore` | `RUN_ID` | — | Move a run back out of `runs/_trash/`. |
| `purge` | `RUN_ID?` | `--all` | Permanently delete one trashed run, or empty the whole trash with `--all`. |

### Override flags

`resume` and `restart` replay the run's originally recorded CLI invocation (`cli_args`, saved by `pipeline.py` on every launch). These flags override individual fields for **this launch only**; anything left unset falls back to the recorded value.

| Flag | Notes |
|---|---|
| `--input` | |
| `--run-mode` | |
| `--limit` | |
| `--set-k` | |
| `--k-range` | |
| `--k-fast` | Boolean, can only be forced **on**. No way to un-set a `true` already recorded. |
| `--use-k-cache` | Boolean, can only be forced **on**. |
| `--skip-naming` | Boolean, can only be forced **on**. |
| `--force-nova` | Boolean, can only be forced **on**. |

**Examples**

```bash
python reload.py list
python reload.py show a1b2c3 --render threejs
python reload.py resume a1b2c3 --resume-from nova
python reload.py resume a1b2c3 --resume-from naming --run-mode full   # add Nova/ADEPT
python reload.py restart a1b2c3 --input data/raw_v2/
python reload.py relabel a1b2c3 --force-skip-llm
python reload.py search a1b2c3 "what drove the June spike?"
python reload.py sync-models a1b2c3 --dry-run
```

---

## `search.py`

```
python search.py "<query>" [options]
```

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `query` | positional | *(required)* | The question to answer. |
| `--run` | run id | — | Resolve `output/` and that run's own `config_snapshot.yaml` automatically. Same resolution as `reload.py search`. |
| `--run-mode` | `full` \| `clustering_only` | auto-detected | Force the prompt shape. `full` sends the `NARRATIVE_CHAIN` / `SEMANTIC_POOL` / `ISOLATED_POST` prompt; `clustering_only` sends the post-only prompt. Auto-detection reads the graph itself and is usually right. |
| `--floor-ratio` | float | `0.80` | Relative relevance floor: keep candidates scoring ≥ `floor_ratio × best_score` for that question. |
| `--budget` | int | `3000` | Total post budget across all retrieved groups. Components are never split; the component that crosses the budget is kept whole and retrieval stops right after it. |
| `--no-decompose` | flag | off | Skip LLM query decomposition and retrieve on the original query text only. |
| `--dry-run` | flag | off | Run retrieval and context assembly as usual, print per-candidate diagnostics, then stop **before calling the LLM**. |
| `--dry-run-out` | path | `dry_run_prompt.md` | Where to write the full system + user prompt in `--dry-run` mode. |
| `--dry-run-retrieval-out` | path | `dry_run_retrieval.md` | Where to write per-candidate vector diagnostics (score, floor status, sub-question) in `--dry-run` mode. |

**`--run` is effectively required.** Without it, search reads `storage.processed_dir` from the config, and that key ships in neither `config.yaml` nor `advanced.yaml` — `run_store.py` only sets it at runtime, per run. Omitting `--run` therefore fails with an explicit error unless you have added `storage.processed_dir` to your own config by hand.

**Examples**

```bash
python search.py "what drove the spike in engagement around June?" --run a1b2c3
python search.py "summarize the main narrative threads" --run a1b2c3 --dry-run
```

---

## `dashboard.py`

```
python dashboard.py [options]
```

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--host` | address | `127.0.0.1` | Bind address. Anything other than `127.0.0.1` / `localhost` / `::1` prints an explicit warning: there is no authentication layer. |
| `--port` | int | `8000` | Bind port. |
| `--no-browser` | flag | off | Do not auto-open a browser tab. |
| `--allow-origin` | hostname, repeatable | `[]` | Additional hostname allowed past the Origin check for WebSockets and mutating POST actions. **Required** alongside `--host` when serving on your network: every WebSocket and mutating request is otherwise rejected as cross-origin, including from the dashboard's own address, which is never inferred automatically. |

**Examples**

```bash
python dashboard.py
python dashboard.py --port 8080
python dashboard.py --host 0.0.0.0 --allow-origin 192.168.1.5   # read DASHBOARD.md first
```

## See also

- [`CONFIGURATION.md`](CONFIGURATION.md) — the config values these flags override
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — what each program does and what it writes
- [`SEARCH.md`](SEARCH.md) — what `search.py`'s flags actually change
- [`DASHBOARD.md`](DASHBOARD.md) — the same actions from the web UI
