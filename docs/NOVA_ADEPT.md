# Nova & ADEPT

Nova and ADEPT are the two LLM analysis stages. They run once per cluster and
only in `full` mode, after Naming and before edge construction. This page
is their operational contract — prompt design, failure handling,
checkpoints, output schemas — and deliberately does **not** restate what other pages own: the ADEPT
density-peak math is in
[`ALGORITHMS.md`](ALGORITHMS.md#adept--density-peak-clustering), the
`nova.*` / `adept.*` config keys in [`CONFIGURATION.md`](CONFIGURATION.md),
call volume and cost in [`PERFORMANCE.md`](PERFORMANCE.md#llm-analysis-nova--adept),
and the CLI flags in [`CLI.md`](CLI.md).

## Nova: narrative subtopic chains

For each cluster, a single LLM call groups posts into coherent
subtopics — threads connected by an explicit causal, chronological, or
argumentative link — ordered as a chain (`pioneer` → `offspring`).

### Prompt design

`content_type` is the single source of truth for all prompt wording
(`nova.py:_build_prompt_params`):

| | `social_media` | `document` |
|---|---|---|
| unit words | posts | documents |
| ordering | chronological (`timeline`) | thematic (`thematic`) |
| valid grouping evidence | shared names, **exact dates**, specific claims | shared names, specific claims, verifiable facts |

Every call includes the following hard rules: no broad categorization by location/
institution/keyword alone; reasoning must cite explicit, observable evidence;
the sentiment arc must describe the actual evolution of the conversation,
not a fabricated narrative.

Posts are rendered as `<post id="..." date="...">text</post>` blocks
(`date` only when the timestamp exists), truncated at a word boundary with
an ellipsis, and angle brackets in the content itself are replaced with
lookalikes (`‹`/`›`) so a post cannot spoof a tag boundary.

### Response handling

The returned JSON passes through a repair-then-validate chain
(`_extract_json_from_text` → `_repair_subtopics` → `_validate_subtopics`):

- Root key and `post_ids` key are normalized (`topics`/`groups`/…, `posts`/`ids`/…).
- Post IDs are matched strictly against the cluster first, with a
  substring-match fallback only if nothing matched exactly.
- Each post is assigned to **one** subtopic (first-seen wins); subtopics
  below `nova.min_posts_per_subtopic` are dropped.
- "No groupable subtopics" is a **valid result** (the cluster's posts all
  go to ADEPT), not an error. Proposing subtopics that all fail validation
  *does* trigger a retry.

In chronological mode, only posts with a valid timestamp are re-sorted;
undated posts keep the position returned by the LLM.

## ADEPT: orphan pools

ADEPT receives only Nova's orphans for each cluster:

1. **Pool formation** — Density Peak Clustering on the LDA embeddings
   (see [`ALGORITHMS.md`](ALGORITHMS.md#adept--density-peak-clustering)
   for ρ/δ/γ and the pool-size bounds). Each pool gets a hub plus
   `pioneer`/`latest`/`member` roles from timestamps and engagement.
2. **LLM review** — one call per cluster reviews *all* pools at once: for
   each pool, the LLM may exclude candidates that don't genuinely belong
   (its `hub_id` is echoed back as `group_id` so explanations can't be
   misattributed by list position). **The hub can never be excluded**;
   unknown ids in `excluded_candidates` are ignored rather than trusted.
   Excluded posts become orphans again.
3. Each pool ends with a title (`Entity + State/Condition`, ≤ 8 words)
   and a one-sentence reasoning.

ADEPT emits `adept_spoke` (hub → member) edges, and nothing else. Grafting
pools back onto the Nova graph is **not** ADEPT's job: `adept_graft` edges are
built later by `edges.py` (`compute_adept_graft_edges`, gated by
`edges.adept_graft.graft_min_cosine`) — see [`ALGORITHMS.md`](ALGORITHMS.md#edges).

## Failure handling: both stages

- Each LLM call is retried up to `max_retries` times (default 3) with exponential
  backoff. ADEPT classifies failures as `network` (HTTP-level) vs `format`
  (unparseable response) for logging; the retry policy is the same.
- **Sequential mode** (`parallel_clusters: 1`): an exhausted cluster opens
  a Y/I/Q prompt — `Y` retry this cluster, `I` ignore + auto-skip all
  further failures this run, `Q` abort. With no terminal available
  (automation, closed stdin), the default is `Q` — fail-safe, never a
  silent skip.
- **Parallel mode** (`parallel_clusters > 1`): no prompts exist across
  threads; exhausted clusters are auto-skipped and reported at the end. A
  heuristic warns if the configured endpoint looks local (`localhost` /
  `127.0.0.1`), where a single model instance usually can't serve
  concurrent requests.
- A failed cluster is **never** added to `completed_clusters`. It stays in
  the resume set and is retried automatically on the next run — a failed
  pass never looks like a completed one.

### Validation gates

Each stage returns a `<stage>_validated` flag (plus `failed_clusters` and
`llm_abandoned`). `all_clusters_done` alone is *not* sufficient — it is
also true when clusters were skipped after exhausting retries. Callers must
gate on the validated flag:

- `nova_validated=False` → the pipeline stops before ADEPT.
- `adept_validated=False` → the pipeline stops before Edges.

In interactive use the gate asks once whether to continue anyway with the
failed clusters left as orphans (fails closed outside a terminal). There is
no degraded "LLM-free" mode — clustering-only exists for that.

## Checkpoints

`nova_checkpoint.json` / `adept_checkpoint.json` live in the run's
`processed_dir`, are written after **every** cluster, and are archived to
`checkpoints_archive/` **only when the whole pass validated** — a live
checkpoint sitting next to the parquet outputs means the last attempt did
not pass the gate, and the pipeline recomputes rather than reusing it.

- Nova's checkpoint also stores `llm_raw_order`: the post order that the LLM
  originally returned per subtopic, captured before chronological re-sort.
  For debugging and auditing only; never written to the parquet outputs.
- `--force-nova` ignores a validated Nova result and recomputes from
  scratch (there is no `--force-adept`: resume from the `adept` stage
  instead, see [`CLI.md`](CLI.md#reloadpy)).

## Outputs

Written to the run's `processed_dir/` (see
[`ARCHITECTURE.md`](ARCHITECTURE.md#what-a-run-folder-contains) for the
full run-folder layout):

| File | Producer | Contents |
|---|---|---|
| `nova_assignments.parquet` | Nova | `id`, `cluster_id`, `nova_role`, `nova_parent`, `nova_depth`, `nova_symbol`, `nova_force_score`, `subtopic_id`, `subtopic_label` |
| `nova_edges.parquet` | Nova | `source`, `target`, `type="nova"`, `force`, `nova_role` |
| `nova_metadata.parquet` | Nova | one row per subtopic: `subtopic_id`, `cluster_id`, `cluster_label`, `title`, `label`, `reasoning`, `sentiment_arc`, `post_count` |
| `adept_assignments.parquet` | ADEPT | `id`, `adept_role`, `adept_symbol`, `eng_norm` |
| `adept_edges.parquet` | ADEPT | `source`, `target`, `type="adept_spoke"`, `force` |
| `adept_metadata.parquet` | ADEPT | per-cluster orphan/pool/edge counts |
| `pool_explanations.parquet` | ADEPT | per pool: `hub_id`, `cluster_id`, `cluster_label`, `pool_title`, `pool_reasoning`, `member_count`, `excluded_count` |
| `adept_exclusions.parquet` | ADEPT | one row per LLM-excluded candidate |

### Column dictionary

| Column | Values | Meaning |
|---|---|---|
| `nova_role` | `pioneer` / `offspring` / `null` | First post of a subtopic chain vs. chained members; `null` = Nova orphan |
| `nova_parent` | post id / `null` | Previous post in the subtopic chain |
| `nova_depth` | int | Position in the chain (pioneer = 0) |
| `nova_force_score` | float | Chain edge weight before per-type multipliers (see `fa2.edge_weights.*`) |
| `subtopic_id` / `subtopic_label` | string | Scoped as `<cluster_id>__<st_id>` / the subtopic's technical label |
| `adept_role` | `hub` / `pioneer` / `latest` / `member` / `null` | Role inside an ADEPT pool; `null` = orphan |
| `eng_norm` | float 0–1 | Engagement normalized within the pool |

`nova_role` and `adept_role` never overlap: ADEPT only ever sees posts whose
`nova_role` is null, so a post carries at most one of the two. A post with a
`nova_role` always has a null `adept_role`, and vice versa.

## See also

- [`ALGORITHMS.md`](ALGORITHMS.md) — the math: ADR, LDA-GO, DPC, edges
- [`CONFIGURATION.md`](CONFIGURATION.md) — every `nova.*` / `adept.*` key
- [`PERFORMANCE.md`](PERFORMANCE.md) — LLM call volume and levers
- [`CLI.md`](CLI.md) — `--resume-from nova|adept`, `--force-nova`
