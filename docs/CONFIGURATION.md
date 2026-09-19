# Configuration reference

Every key in `config/config.yaml` and `config/advanced.yaml`, with the value actually shipped. If a key is set in both files, **`config.yaml` wins** (see `run_store.load_config()`), and the merged result is what every program reads.

- [`config.yaml`](#configyaml) — what you change per project: input format, field mapping, which LLM each module uses, run mode.
- [`advanced.yaml`](#advancedyaml) — algorithm internals: embedding, clustering, ADR, edges, ForceAtlas2, render. Rarely touched; it ships with working defaults.
- [Runtime-only keys](#runtime-only-keys) — set by the code, not by you.

A run freezes its config into `runs/<run_id>/config_snapshot.yaml` at launch. Editing the live files afterward does not affect that run, except through `reload.py sync-models`.

---

## `config.yaml`

### Input

| Key | Shipped value | Meaning |
|---|---|---|
| `content_type` | `"social_media"` | `"social_media"` (chronological Nova ordering) or `"document"` (thematic ordering). Also selects the embedding instruction prefix and the posts/documents wording in every prompt. Fixed for the life of a run; `--content-type` only affects a brand-new run, never a resume. |
| `field_mapping.content` | *(blank)* | Source column for post text. Blank = auto-detect among `content`, `body`, `text`, `selftext`, `message`. |
| `field_mapping.id` | *(blank)* | Source column for a stable post id. There is no alias list: auto-detection only matches a column literally named `id`. Posts with no id get a content hash. |
| `field_mapping.title` | *(blank)* | Source column for a title. Blank = auto-detect among `title`, `headline`, `subject`. Concatenated with content when both exist. |
| `field_mapping.timestamp` | *(blank)* | Source column for the timestamp. Blank = auto-detect among `timestamp`, `created_at`, `published_at`, `posted_at`. Numeric values are read as Unix epochs. |
| `field_mapping.engagement` | *(blank)* | Source column for an engagement metric. Blank = auto-detect among `engagement`, `score`. Values are then percentile-normalised per file to `[1, 10]`, see [`ALGORITHMS.md`](ALGORITHMS.md#ingestion). |

Setting a `field_mapping` key renames that column before detection runs, so it is the way to handle any name outside the alias lists (Reddit's `created`, for instance).

### LLM models

| Key | Shipped value | Meaning |
|---|---|---|
| `models.heavy.*` | OpenRouter, `z-ai/glm-5.3-flash` | The "heavy" slot, used by **Nova, ADEPT, and `search.py`**. Shipped with `timeout: 180`, `max_retries: 3`, `num_ctx: 1048576`, `max_tokens: 131072`, and `extra_params` pinning the provider (`reasoning.effort: high`, `provider.order: [Novita]`, `allow_fallbacks: false`). Every field is provider-dependent; swap the whole block to switch provider. OpenAI and Anthropic templates ship commented out in the file. |
| `models.light.*` | local Ollama, `qwen3.5:9b` | The "light" slot, used by cluster naming. Shipped with `timeout: 320`, `max_retries: 1`, `num_ctx: 60000`, `max_tokens: 30000`, `extra_params: {think: false}`. Same shape as `models.heavy`. |
| `models.<slot>.api_key` | *(blank)* | **Never fill this in.** Leave it blank and set `<SLOT>_API_KEY` in `.env` instead: default slots map to `HEAVY_API_KEY` / `LIGHT_API_KEY`, a custom slot named `anthropic` maps to `ANTHROPIC_API_KEY`. See `.env.example` and `_llm.py:get_llm_cfg`. |

You can define additional slots under `models.` and point a module at one with its `model_slot` key.

### Run mode and module routing

| Key | Shipped value | Meaning |
|---|---|---|
| `run_mode` | `"full"` | `"full"` = clustering → naming → Nova → ADEPT → edges → 3D layout. `"clustering_only"` = the same **minus Nova and ADEPT**; edge construction and 3D layout still run. Overridable per run with `--run-mode`. |
| `nova.model_slot` | `heavy` | Which LLM slot Nova uses. |
| `adept.model_slot` | `heavy` | Which LLM slot ADEPT uses. |
| `naming.model_slot` | `light` | Which LLM slot cluster naming uses. |
| `nova.parallel_clusters` | `1` | Clusters processed concurrently by Nova. `1` (shipped default) is sequential — the safe, precautionary starting point, and it's required for local models, which usually can't serve concurrent requests well. |
| `adept.parallel_clusters` | `1` | Same, for ADEPT. |

>[!TIP]
>Shipping 1 as the default is deliberate — it's the only value guaranteed to work with a local model, so new setups can't accidentally overload one. If you're on a fast API model this is the first thing to raise: 20-50 is a common parallel_clusters value and can turn hours of sequential Nova/ADEPT calls into minutes.
>
>Changing either value in config.yaml only affects new runs — an existing run keeps the value from its original configuration snapshot. To change the setting for a run that has already started, first edit config.yaml. In the dashboard, open the ⋮ (three-dot) menu on the right of the run's toolbar, select Sync models, then click Resume to continue the run with the updated setting.
>
> Alternatively, use the CLI:  
> `reload.py sync-models <run_id>`  
> `reload.py resume <run_id> --resume-from nova` (or `adept`).


The search feature does not have a configurable `model_slot`. Search always uses the `heavy` slot, defined by `SEARCH_MODEL_SLOT` in `search.py`. To use a different model slot for search, change that constant.

---

## `advanced.yaml`

### Embedding

| Key | Shipped value | Meaning |
|---|---|---|
| `embedding.mode` | `"ollama"` | Embedding backend. |
| `embedding.ollama_url` | `http://localhost:11434/api/embed` | Ollama's embedding endpoint. |
| `embedding.ollama_model` | `qwen3-embedding:8b` | Model to embed with. Must be pulled first: `ollama pull qwen3-embedding:8b`. |
| `embedding.batch_size` | `64` | Posts per embedding request. |
| `embedding.num_ctx` | `8192` | Context window, matching Qwen3-Embedding-8B's maximum. Used by the embedding stage; `search.py` sends `8192` for its own query embedding regardless. |
| `embedding.truncate` | `true` | Apply MRL truncation to `raw_dimensions`. |
| `embedding.raw_dimensions` | `1024` | Truncation target. If you change it, it must match the fallback baked into `embedding.py` and `search.py`, or query and document vectors land in different spaces. |
| `embedding.instruct_prompt` | *(blank)* | Instruction prefix for the **clustering** store. Blank = auto-derived from `content_type`. |

Two further keys are read by `search.py` but do not ship in the file. Add them only to override the defaults:

| Key | Default when absent | Meaning |
|---|---|---|
| `embedding.retrieval_instruct` | derived from `content_type` | The fixed task instruction wrapped around a search query. Never reworded per query. |
| `embedding.retrieval_parquet` | `embeddings_retrieval.parquet` | Filename of the retrieval store inside the run folder. |

### Dimensionality reduction

| Key | Shipped value | Meaning |
|---|---|---|
| `umap_cluster.n_components` | `5` | UMAP target dimensions before HDBSCAN. 5D is the BERTopic-recommended setting for short social-media text. |
| `umap_cluster.n_neighbors` | `15` | UMAP neighbourhood size. |
| `umap_cluster.min_dist` | `0.0` | Packs points tightly, favouring HDBSCAN density detection over visual spread. |
| `umap_cluster.metric` | `"cosine"` | Distance metric for pre-clustering UMAP. |
| `umap_cluster.random_state` | `42` | UMAP seed. |
| `pacmap_layout.n_components` | `3` | PaCMAP target dimensions. This, not UMAP, is what seeds the 3D layout. |
| `pacmap_layout.n_neighbors` | `15` | PaCMAP neighbourhood size. |
| `pacmap_layout.MN_ratio` / `FP_ratio` | `0.5` / `2.0` | Mid-near and further-pair ratios, controlling local versus global structure preservation. |
| `pacmap_layout.metric` | `"euclidean"` | Distance metric for the layout projection. |
| `pacmap_layout.random_state` | `42` | PaCMAP seed. |
| `pacmap_layout.output_scale` | `50` | Output coordinate scale before FA2 takes over. |
| `pacmap_layout.scale_mode` | `"per_axis"` | `per_axis` normalises X/Y/Z independently (spherical universe, recommended); `uniform` preserves geometry but can make Z visually shorter. |
| `pacmap_layout.layout_space` | `"lda"` | Project from `embedding_lda` (ADR's discriminant space) or `"raw"` for `embedding_raw`. |

### Clustering — K estimation

| Key | Shipped value | Meaning |
|---|---|---|
| `clustering.k_estimation_method` | `"gmm_bic"` | `"gmm_bic"` (UMAP + GMM + penalised BIC sweep) or `"hdbscan"` (use HDBSCAN's own count directly). `--k-fast` forces `hdbscan` for one run. If the key is missing entirely, the code falls back to `hdbscan`, so do not delete it expecting the documented default. |
| `clustering.gmm_n_jobs` | `-1` | Parallel jobs for the sweep (`-1` = all cores). |
| `clustering.bic_penalty_alpha` | `0.5` | Penalty weight discouraging over-fragmentation in the sweep. |
| `clustering.clustering_float` | `float64` | Precision for the clustering stage. |
| `clustering.gmm_input_space` | `"raw"` | `"raw"` or `"soft_zca"` (apply Soft-ZCA whitening before the sweep). |
| `clustering.gmm_zca_eps` | `0.01` | Soft-ZCA eigenvalue regulariser ε (paper: 0.01 or 0.1). |
| `hdbscan.min_cluster_size` | `15` | HDBSCAN's core density parameter. |
| `hdbscan.min_samples` | `7` | Conservativeness; higher means more points labelled noise. |
| `hdbscan.metric` | `"euclidean"` | Distance metric, applied post-UMAP. |
| `hdbscan.method` | `"leaf"` | Cluster selection method. |
| `hdbscan.prediction_data` | `true` | Keep the data needed for soft cluster membership. |
| `hdbscan.cluster_selection_epsilon` | `0.0` | Merge clusters closer than this distance. |
| `hdbscan.soft_zca_input_space` | `false` | Apply the same whitening as `gmm_input_space` before HDBSCAN, for consistency with a ZCA sweep. |
| `hdbscan.soft_zca_eps` | `0.01` | Soft-ZCA ε for HDBSCAN's input. |

### ADR — Adaptive Discriminant Refinement

| Key | Shipped value | Meaning |
|---|---|---|
| `adr.max_iter` | `30` | Maximum LDA ↔ GMM refinement passes. |
| `adr.random_state` | `42` | Seed for the refinement loop. |
| `adr.covariance_type` | `"diag"` | GMM covariance type, `"diag"` or `"full"`. |
| `adr.n_dims` | `auto` | Discriminant subspace dimensionality; `auto` = K − 1. |
| `adr.gmm_repetitions` | `10` | GMM restarts per pass, best-of-N by log-likelihood. |
| `adr.soft_zca_input_space` | `true` | Apply Soft-ZCA whitening before the ADR loop. |
| `adr.soft_zca_eps` | `0.01` | Soft-ZCA ε for ADR's input. |
| `adr.ldago_lr` | `0.1` | LDA-GO gradient learning rate. |
| `adr.ldago_max_iter` | `30000` | LDA-GO maximum gradient iterations. |
| `adr.ldago_tol` | `1e-3` | Convergence threshold on ‖∇L‖_F. |
| `adr.ldago_force_ce` | `true` | Always take the cross-entropy objective, skipping the structural diagnostic. Set `false` to let the data choose. |
| `adr.ldago_ce_optimizer` | `"sgd"` | `"sgd"` (paper default) or `"adam"`. |
| `adr.ldago_random_state` | `42` | LDA-GO's own seed. |
| `adr.ldago_device` | `"cuda"` | `null` (NumPy/CPU), `"cuda"`, or `"cpu"`. The shipped value is `"cuda"`; set it to `"cpu"` or `null` on a machine without a CUDA-enabled build. |
| `adr.ldago_stop_on_loss_increase` | `true` | Early-stop LDA-GO if the loss rises between iterations. |
| `adr.adr_float` | `"float64"` | `"float64"` (recommended) or `"float32"` (faster, less numerically stable). |

### Edges

| Key | Shipped value | Meaning |
|---|---|---|
| `edges.temporal_influence.semantic_min` | `0.95` | Minimum cosine similarity to validate a `temporal_influence` link. |
| `edges.temporal_influence.tau_hours` | `6` | Exponential decay window, in hours. |
| `edges.temporal_influence.tmax_hours` | `48` | Absolute temporal ceiling, in hours. |
| `edges.temporal_influence.threshold_percentile` | `0` | Minimum accepted edge-strength percentile. |
| `edges.temporal.*` | mirrors `temporal_influence` | Same four knobs for the engagement-agnostic `temporal` edge type. |
| `edges.temporal.pass2_max_out` / `pass2_max_in` | `2` / `1` | Per-node emission and reception quota in the second b-matching pass. |
| `edges.semantic_inter.threshold` | `0.95` | Minimum cosine for a greedy 1-to-1 inter-cluster bridge. |
| `edges.adept_graft.graft_min_cosine` | `0.9` | Minimum cosine to graft an ADEPT hub onto its nearest Nova node. |
| `edges.adept_graft.force_graft` | `0.0` | FA2 force for graft edges. At `0.0` the graft exists in the graph but exerts no pull. |
| `edges.pruning.max_component_size` | `500` | Final connected-component size cap, enforced once after every edge type is built. Oversized components are split by cutting the least-disposable edge first. |
| `edges.pruning.isolation_guard.enabled` | `false` | Reject a candidate bridge whose endpoints are both isolated singletons. |
| `edges.pruning.isolation_guard.dynamic` | `true` | When the guard is enabled, a node unblocked by an earlier accepted bridge in the same pass becomes a valid anchor for later candidates. |

### Nova and ADEPT tuning

Model routing and parallelism for these two modules live in `config.yaml`, above.

| Key | Shipped value | Meaning |
|---|---|---|
| `nova.min_posts_per_subtopic` | `2` | Minimum posts to form a Nova subtopic chain. |
| `nova.max_text_chars` | `500` | Post text truncation before it enters the Nova prompt. `search.py` hardcodes the same 500 for its own context and does not read this key, so changing it here does not change search. |
| `nova.max_retries` | `3` | Overrides `models.heavy.max_retries` for Nova specifically. ADEPT has no equivalent key and uses its slot's `max_retries`. |
| `nova.checkpoint_enabled` | `true` | Checkpoint Nova's output independently. ADEPT checkpoints unconditionally and has no equivalent key. |
| `adept.apply_forces` | `1` | `1` = ADEPT edges influence the FA2 layout, `0` = visual only. |
| `adept.min_orphans_for_event` | `2` | Minimum Nova orphans required to run ADEPT on a cluster at all. |
| `adept.pool_min_size` / `pool_max_size` | `3` / `20` | Pool size bounds (hub + members). Excess candidates stay orphan; `pool_max_size: null` removes the cap. |
| `adept.dpc_dc_percentile` | `2.0` | Percentage of point pairs used to set the DPC cutoff distance `dc` (paper: 1–2%). |
| `adept.dpc_membership_percentile` | `95` | δ percentile above which a point stays orphan instead of joining a distant pool. |
| `adept.dpc_gamma_percentile` | `60` | γ = ρ·δ percentile threshold to declare a density hub. |
| `adept.force_spoke` | `0.1` | Base FA2 edge force, hub → member, before `fa2.edge_weights.adept_spoke` multiplies it. |

### ForceAtlas2 layout

| Key | Shipped value | Meaning |
|---|---|---|
| `fa2.scaling_ratio` | `2.0` | Repulsion strength. Scaled by `0.1` in `fa2_layout.py` before being handed to ForceAtlas2, so the effective value is `0.2`. |
| `fa2.gravity` | `1.0` | Pull toward the centre. |
| `fa2.max_iter` | `100` | Simulation iterations per cluster, capped at `200` in code. |
| `fa2.edge_weight_influence` | `1.0` | How strongly edge weights affect attraction, ForceAtlas2's `edgeWeightInfluence`. |
| `fa2.jitter_tolerance` | `0.2` | Speed versus precision trade-off. |
| `fa2.lin_log_mode` | `false` | Log-scaled attraction; tighter clusters when enabled. |
| `fa2.strong_gravity_mode` | `false` | Stronger, more literal pull to the centre. |
| `fa2.local_scale` | `150.0` | Expands internal cluster structure, avoiding a compact blob. |
| `fa2.universe_scale` | `4000.0` | Spreads cluster centroids apart, avoiding inter-cluster overlap. |
| `fa2.nova_straighten.*` | enabled, `residual_factor: 0.1`, `min_chain_len: 3` | Straightens folded Nova chains after FA2 converges. `0` = perfectly straight, `1` = untouched. |
| `fa2.node_spacing.*` | enabled, `margin: 3.0` | Minimum spacing between nodes, scaled to the actual render size (`render.node_size_mult`) rather than an arbitrary distance. |
| `fa2.block_reorganize.*` | enabled, `margin: 1.15` | Treats each Nova chain or ADEPT pool as one rigid block and repositions overlapping blocks within a cluster. |
| `fa2.cluster_overlap_removal.*` | enabled, `margin: 1.2`, `max_iter: 200` | Hard-threshold, mass-weighted repulsion between overlapping cluster spheres. |
| `fa2.cluster_declump.*` | enabled, `strength: 100.0`, `iterations: 200` | Continuous, distance-decaying repulsion between cluster centroids. Pushes hardest where crowded, barely touches already-isolated clusters. Runs before `cluster_overlap_removal`. |

#### `fa2.edge_weights`

Per-edge-type multiplier on FA2's attraction force.

| Edge type | Shipped weight | Paired force key |
|---|---|---|
| `nova` | `1.0` | — |
| `adept_spoke` | `3.0` | `adept.force_spoke` = `0.1` |
| `adept_graft` | `1.0` | `edges.adept_graft.force_graft` = `0.0` |
| `temporal_influence` | `0.0` | — |
| `temporal` | `0.0` | — |
| `semantic_inter` | `0.0` | — |

Three of the six are shipped at `0.0`, and the graft's paired force is `0.0` too, so in the default configuration only Nova chains and ADEPT spokes exert attraction. The other edges are still built, stored, drawn in the 3D view, and traversed by search; the layout simply ignores them. Raise their weights if you want them to pull clusters together.

Every other `fa2.*` key not listed above (per-pass iteration counts, damping factors, anchor strengths, mass-weighting modes) is a smaller tuning constant documented by the inline comments in `advanced.yaml`.

### Render

The render block is mostly visual fine-tuning: label fonts, offsets, fade speed, dash patterns, node colours and shapes per role. The keys that matter most when customising the look:

| Key | Shipped value | Meaning |
|---|---|---|
| `render.node_size_mult` | `10.0` | Global node size multiplier. Also feeds `fa2.node_spacing`. |
| `render.mega_pct` / `big_pct` | `0.95` / `0.80` | Engagement percentile thresholds for "mega" and "big" node sizing. |
| `render.nav_margin_ratio` | `0.20` | Camera bounding-box margin, as a fraction of the diagonal. |
| `render.edge_colors.*` | per edge type | Hex colour per edge type. |
| `render.node_roles.*` | per role | Shape and colour per node role: `pioneer`, `hub`, `amplifier`, `latest`, `member`, `offspring`, `orphan`. |
| `render.label_max_visible.cluster` | `12` | Hard cap on simultaneously visible cluster labels; nearest to camera wins. `0` = unlimited. |
| `render.label_nudge.*` | enabled, `max_px: 80` | Declutters by nudging conflicting labels apart on screen instead of hiding them. |
| `render.cluster_min_radius` | `3000` | Minimum world-unit radius a cluster always occupies, so titles do not overflow tiny clusters. Also read by `fa2_layout.py` for spacing. |
| `render.nova_spline.*` | enabled, `samples: 10`, `tension: 0.2` | Renders Nova chains as smooth Catmull-Rom splines instead of straight segments. |

Every other `render.*` key is a smaller visual constant (edge opacity and width, dash sizes, label offsets and backdrop padding, fade speed). The inline comments in `advanced.yaml` are detailed enough to tune those by eye.

---

## Runtime-only keys

`storage.*` appears in neither YAML file. It is set at runtime, per run.

| Key | Meaning | Fallback when absent |
|---|---|---|
| `storage.processed_dir` | The run's own folder, `runs/<run_id>/`. Everything outside `output/` is read from and written to here, including `searches/`. | None. `search.py` raises an explicit error. |
| `storage.clusters_file` | Name of the clusters table `search.py` opens inside `output/`. Only the filename part is used. | `clusters.parquet` |
| `storage.edges_file` | Name of the edge table `search.py` opens inside `output/`. Only the filename part is used. | `edges.parquet` |

This is why `search.py` needs `--run`: it resolves `processed_dir` itself from the run id. Without it, and without the key added to `config.yaml` by hand, the command exits with an error.

## See also

- [`ALGORITHMS.md`](ALGORITHMS.md) — what each of these knobs actually controls
- [`CLI.md`](CLI.md) — the flags that override config values per run
- [`PERFORMANCE.md`](PERFORMANCE.md) — which keys are the real cost levers
- [`ARCHITECTURE.md`](ARCHITECTURE.md#what-a-run-folder-contains) — where the frozen snapshot lives
