# Dashboard

The web UI for launching, monitoring, resuming, relabelling, browsing, and querying runs.

The dashboard computes nothing itself. Anything that mutates a run launches `pipeline.py` or `reload.py` as a subprocess in a terminal the browser can watch and type into.

## Launch

```bash
python dashboard.py
```

Available at `http://127.0.0.1:8000` by default, and it opens a browser tab automatically.

| Flag | What it does |
|---|---|
| `--port <n>` | Use a different port (default `8000`). |
| `--host <address>` | Bind somewhere other than `127.0.0.1`. Read [Security](#security) first. |
| `--no-browser` | Do not open a browser tab automatically. |
| `--allow-origin <host>` | Additional hostname allowed alongside `--host`. Required when binding to anything else. |

## The two tabs

- **Active runs** — every run with its status, current stage, and how much has been processed. Updates automatically as runs progress.
- **Trash** — runs removed from the active list.

## Run status

| Status | Meaning |
|---|---|
| `complete` | Finished successfully. |
| `in_progress` | Currently running. |
| `crashed` | Recorded as in-progress, but nothing is running it anymore. Resume to pick up from the last checkpoint. |
| `unreadable` | The run's saved state could not be read. Trash, restore, and purge still work. |

**★** next to a run id marks the active run. **locked** means another process currently holds it, so Resume, Relabel, Sync models, and Trash are hidden until it is released.

## Starting a new run

Click **+ New run**.

| Field | What it does |
|---|---|
| **Run mode** | **Full** — clustering, naming, LLM cluster analysis, edges, 3D layout. **Clustering only** — the same without the LLM analysis step; edges and layout still run. |
| **Content type** | **Social media** for posts (ordered chronologically), **Document** for other text (ordered thematically). Fixed for the life of the run; Resume cannot change it. |
| **Input folder** | Blank uses the default (`data/raw/`), or pick one with **Browse…**. Supported files: `.json`, `.csv`, `.tsv`, `.txt`, `.md`, `.pdf`. A `.txt`, `.md`, or `.pdf` file becomes one post. |
| **Number of clusters** | Blank auto-detects the count, or enter a value to force one. **Fast** skips the thorough GMM-BIC sweep for a quicker, less refined count; not recommended, and ignored once a custom count is entered. |

A box below the fields shows the exact command this will run. Click **Launch**.

## Watching a run

A panel shows live output as the run progresses. You can type into it to answer interactive prompts, exactly like a real terminal. The pipeline asks for input in three situations:

- duplicate ids in the source data (`y` to auto-suffix them and continue, anything else aborts);
- the naming LLM is unreachable (`Y` to fall back to c-TF-IDF labels, `R` to retry);
- Nova or ADEPT did not pass its validation gate (`C` to continue to the next stage anyway, `Q` to abort).

All three fail closed: with no terminal attached, the answer comes back empty and the pipeline takes the safe branch.

- **Clear** — clears the visible output. The full log stays available for download.
- **Download log** — save everything as a text file.
- **Stop** — stop the run. `Ctrl-C` is disabled here on purpose, since it is too easy to kill a long run by accident.

Drag the bar at the top of the panel to resize it, or double-click to reset. **Console**, near the bottom of the page, shows or hides the panel without stopping whatever is running inside it.

On systems where interactive sessions are not supported (rare, mainly Windows without an extra package installed), New run, Resume, Relabel, Sync models, Ask a question, and Generate 3D view are greyed out. The run table and trash, restore, and purge keep working.

## Working with a run

- **Run details** — click a run's id for its stage, post and edge counts, key timestamps, input folder, and config snapshot.
- **Resume** — pick up a paused or interrupted run, or add the LLM analysis stage to a clustering-only run. The caret lets you resume from a specific stage or with a different run mode.
- **Relabel** (⋮ menu) — redo just the cluster names, nothing else.
- **Relabel without LLM** (⋮ menu) — the same, using c-TF-IDF keywords only.
- **Sync models** (⋮ menu) — update the run's frozen model routing to match your current `config.yaml`.
- **Ask a question** — run a search against the run's output in plain language. The caret lists past questions, which can be reopened or deleted. Each one is also saved to `runs/<run_id>/searches/`; see [`SEARCH.md`](SEARCH.md).
- **Open 3D view** — once a run is complete, opens its interactive 3D visualisation in a new tab. See [`RENDER.md`](RENDER.md).

## Trash

- **Move to trash** — remove a run from the active list without deleting it.
- **Restore** — move it back.
- **Delete permanently** — irreversible. Emptying the whole trash at once requires typing back the exact number of runs being deleted.

## Documentation

**Guide** in the top bar opens this page inline, without leaving the dashboard. **Docs** lists every project document; picking one opens it in a new tab with its own sidebar and page outline.

## Security

There is no authentication layer. Read [`SECURITY.md`](../SECURITY.md) before exposing the dashboard beyond your own machine.

`--host 0.0.0.0` also requires `--allow-origin <your-host-ip>`, or WebSockets and every state-changing action (new run, resume, trash, restore, purge) are rejected as cross-origin. The dashboard's own address is never inferred automatically.

```bash
python dashboard.py --host 0.0.0.0 --allow-origin 192.168.1.5
```

## See also

- [`CLI.md`](CLI.md#dashboardpy) — the same actions from the command line
- [`ARCHITECTURE.md`](ARCHITECTURE.md#components) — how the dashboard relates to the other programs
- [`RENDER.md`](RENDER.md) — the 3D view it opens
- [`SEARCH.md`](SEARCH.md) — what "Ask a question" runs
