# Algorithms and citations

The mathematical detail behind each stage, with the exact published sources. Every config key named here is documented in full in [`CONFIGURATION.md`](CONFIGURATION.md).

## Ingestion

Before any embedding, raw files are normalised into one table.

- **Supported formats** — `.json`, `.csv`, `.tsv`, `.txt`, `.md`, `.pdf`. A JSON file holds one record or a list of records; a CSV/TSV holds one record per row. A `.txt`, `.md`, or `.pdf` file becomes **one post** whose content is the whole file, so chunking, if needed, happens before ingestion.
- **Field detection** — columns are matched case-insensitively, first match wins: title (`title`, `headline`, `subject`), body (`content`, `body`, `text`, `selftext`, `message`), engagement (`engagement`, `score`), timestamp (`timestamp`, `created_at`, `published_at`, `posted_at`). When both a title and a body are found they are concatenated into `content`; the original columns are kept so the renderer can still show them separately. `field_mapping` in `config.yaml` overrides detection by renaming a column before this runs.
- **Timestamps** — only columns matching the timestamp aliases are auto-detected. A numeric value is read as a Unix epoch and converted to ISO 8601 UTC. Any other column name (Reddit's `created`, for instance) needs `field_mapping.timestamp`. Without a timestamp, temporal ordering is disabled for that file.
- **Ids** — `field_mapping.id` renames a source column; there is no alias list, so the column must be named `id` for auto-detection. A post with no id gets a deterministic one derived from its content (SHA-1, first 16 hex characters), so re-ingesting the same source data reproduces the same ids and naturally de-duplicates.
- **Engagement** — normalised **per file** to `[1, 10]` by percentile rank (`1 + 9 × rank`). A file with no variance is pinned at the ceiling. Posts with no value are filled with the global median across all files once they are concatenated, or `5.5` if the corpus has no engagement anywhere. This is why engagement is only ever a relative signal: it says where a post sits inside its own source file, not how many people actually reacted.

## Embedding

Posts are embedded with **Qwen3-Embedding-8B**, served locally through [Ollama](https://ollama.com/). The raw vector is truncated to `embedding.raw_dimensions` (default 1024) via **Matryoshka Representation Learning (MRL)**: the model was trained so a prefix of its full embedding is itself a valid embedding, so truncating trades a little accuracy for a proportional cut in memory and downstream compute. Embeddings are cached to disk (`diskcache`) keyed on content, so re-runs never re-embed unchanged posts.

The stage writes two stores, both in the run folder:

| Store | Built from | Used by |
|---|---|---|
| `embeddings.parquet` | text with the `content_type` instruction prefix (`social_media` or `document`) | clustering, ADR, layout |
| `embeddings_retrieval.parquet` | raw text, no prefix | `search.py` |

The split exists because Qwen3-Embedding's retrieval convention is asymmetric: the query is wrapped in a task instruction, the document is not. Reusing the clustering store for search would wrap both sides and degrade retrieval. See [`SEARCH.md`](SEARCH.md#retrieval).

## Whitening — Soft-ZCA

High-dimensional sentence embeddings are typically **anisotropic**: cosine similarity is dominated by a handful of high-variance directions shared by almost every point, which compresses genuine semantic differences into a narrow range and hurts both clustering and nearest-neighbour search. **Soft-ZCA whitening** corrects this by rescaling the embedding space along its own principal axes with a regularised inverse-square-root of the covariance matrix, controlled by a single eigenvalue regulariser ε (`clustering.gmm_zca_eps`, `hdbscan.soft_zca_eps`, `adr.soft_zca_eps`; all default to `0.01`) interpolating between the raw space (ε → ∞) and full ZCA whitening (ε → 0). It is "soft" because ε is tuned rather than fixed, which avoids the numerical instability of a literal matrix inverse on near-zero eigenvalues. It can be applied before K estimation (`clustering.gmm_input_space`, `hdbscan.soft_zca_input_space`) and before the ADR loop (`adr.soft_zca_input_space`).

> Andor Diera, Lukas Galke, Ansgar Scherp. **"Isotropy Matters: Soft-ZCA Whitening of Embeddings for Semantic Code Search."** arXiv:[2411.17538](https://arxiv.org/abs/2411.17538), 2024. Presented at ESANN 2025.

## K estimation

Two methods, selected by `clustering.k_estimation_method`.

- **HDBSCAN pivot** — [HDBSCAN](https://github.com/scikit-learn-contrib/hdbscan) runs on a UMAP-reduced embedding (`umap_cluster.*`: 5D, cosine metric, the BERTopic-recommended setting for short social-media text) and its own non-noise cluster count becomes the pivot.
  > Ricardo J. G. B. Campello, Davoud Moulavi, Jörg Sander. **"Density-Based Clustering Based on Hierarchical Density Estimates."** PAKDD 2013.
  > Leland McInnes, John Healy, Steve Astels. **hdbscan: Hierarchical density based clustering.** *Journal of Open Source Software* 2(11), 2017.
- **GMM-BIC sweep** (shipped default) — a Gaussian Mixture Model is fit for a range of K around the pivot, each scored with a **penalised Bayesian Information Criterion** (`clustering.bic_penalty_alpha` discourages fragmenting into many small clusters), and the best-scoring K is kept.

Whichever path runs, it also produces the seed labels ADR starts from. `--k-fast` (`clustering.k_estimation_method: hdbscan`) skips the sweep and takes the HDBSCAN pivot directly as K, then runs a single GMM fit at that K purely to generate those seed labels. `--set-k N` and `--k-range MIN MAX` go through the sweep over the given bounds. Every path except `--use-k-cache`, which reads it back, persists K and the seed labels to `k_cache.pkl`.

K estimation is usually the smaller of the two compute-heavy clustering stages. See [`PERFORMANCE.md`](PERFORMANCE.md#clustering-cluster-count-estimation-and-adr).

## ADR — Adaptive Discriminant Refinement

```text
GMM-BIC seed partition
        ↓
LDA-GO — discriminant subspace from current assignments
        ↓
GMM — re-cluster inside that subspace
        ↓
labels stable / loss check
        ↓
      repeat
```

The core clustering-quality step: an iterative loop alternating between fitting a discriminant projection given the current labels and re-clustering in that projection. `adr_refiner.py` is adapted from **TopiCLEAR**'s ADR loop, replacing its closed-form LDA step with **LDA-GO** and keeping its GMM re-clustering step. TopiCLEAR's own ADR scheme is itself an application, with GMM in place of k-means, of the general Adaptive Dimension Reduction concept.

> **On the acronym.** The source literature reads ADR as *Adaptive Dimension Reduction* (Ding & Li, 2007). Here it refers to *Adaptive Discriminant Refinement*: the reduction step is specifically a discriminant projection, and it is refined over several passes rather than computed once.

> Chris Ding, Tao Li. **"Adaptive dimension reduction using discriminant analysis and k-means clustering."** ICML 2007.

> Aoi Fujita, Taichi Yamamoto, Yuri Nakayama, Ryota Kobayashi. **"TopiCLEAR: Adaptive embedding clustering for interpretable topic discovery from short texts."** *Expert Systems with Applications* 333, 133996, 2027. https://doi.org/10.1016/j.eswa.2026.133996. Preprint: arXiv:[2512.06694v2](https://arxiv.org/abs/2512.06694v2) (titled differently on arXiv). Code (MIT licence, adapted with attribution, see [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md)): [github.com/aoi8716/TopiCLEAR](https://github.com/aoi8716/TopiCLEAR).

### LDA-GO

Classical Linear Discriminant Analysis needs the within-class scatter matrix to be invertible, which fails outright in the high-dimensional, small-sample regime typical of embeddings (more dimensions than posts per cluster). **LDA-GO** replaces the closed-form eigen-solution with a directly optimised, low-rank discriminant projection `L` (shape *p × d*), avoiding that inversion entirely. `lda_go.py` is a from-scratch implementation of the paper's Algorithm 1:

1. **Within-class standardisation** — features are rescaled by their within-class standard deviation before optimisation; the inverse transform is stored for inference on new data.
2. **Precision parametrisation** — the inverse covariance is written as Σ⁻¹ = LLᵀ + σ²I. σ² is set via Oracle Approximating Shrinkage for the likelihood path and fixed to 0 for the cross-entropy path.
3. **Path selection via structural diagnostics, overridable** — signal sparsity *r = D_eff / p* and excess kurtosis *κ* are computed on the data; the rule *r < 0.10 OR κ > 10 → cross-entropy path*, otherwise the negative-log-likelihood path. `adr.ldago_force_ce` (`true` in the shipped config) skips the diagnostic and always takes the cross-entropy path; set it to `false` to let the data decide.
4. **Two optimisation paths** — NLL: Gaussian negative log-likelihood via Adam. CE: cross-entropy via SGD or Adam (`adr.ldago_ce_optimizer`), with no *p×p* intermediate matrix, which keeps it cheap at high dimensionality.
5. **Initialisation** — L⁰ = I_{p×d} + ε, ε ~ 𝒩(0, 0.01²), per the paper.

> Cencheng Shen, Yuexiao Dong. **"Linear Discriminant Analysis with Gradient Optimization."** arXiv:[2506.06845v2](https://arxiv.org/abs/2506.06845v2), 2025.

### The refinement loop

The loop is seeded with the partition from the K-estimation stage, falling back to a cold GMM(K) fit if no seed labels are supplied. Each pass: LDA-GO projects the embeddings into its discriminant subspace given the current labels, then a GMM (`adr.gmm_repetitions` restarts, best-of-N by log-likelihood) re-clusters in that subspace and produces new labels.

The loop stops as soon as any of these holds:

- **Fixed point** — the GMM's new labels exactly reproduce the labels LDA-GO started from this pass (NMI = 1.0).
- **Stalled projection** — the next pass's LDA-GO starting loss is higher than this pass's, meaning the projection stopped improving. Checked before that next pass's GMM fit runs, so its cost is only paid when there is a reason to.
- **GMM collapse** — fewer than K active components (safety net).

Otherwise the loop runs up to `adr.max_iter` passes and keeps the last one completed, not a scored "best". The NMI between consecutive passes is logged throughout, making progress toward a fixed point visible.

This loop, not K estimation, is why clustering time grows sharply with corpus size: every pass fits a projection sized to K (`adr.n_dims`, default `K − 1`) over the whole corpus, then re-fits a K-component GMM `adr.gmm_repetitions` times, repeated for up to `adr.max_iter` passes. Both a larger corpus and a higher cluster count make each pass more expensive. See [`PERFORMANCE.md`](PERFORMANCE.md#clustering-cluster-count-estimation-and-adr).

## Naming

Two signals feed each cluster's name: **c-TF-IDF** keywords, via [BERTopic](https://github.com/MaartenGr/BERTopic)'s `ClassTfidfTransformer` (TF-IDF computed treating each *cluster* as one document rather than each post, so a term's weight reflects how much it distinguishes this cluster from every other), and one LLM call (`models.light` by default) turning those keywords plus a sample of representative posts into a human-readable label. If the naming LLM is unreachable, the stage falls back to c-TF-IDF-only labels.

> Maarten Grootendorst. **"BERTopic: Neural topic modeling with a class-based TF-IDF procedure."** 2022.

## Nova — narrative subtopic chains

Within each cluster, an LLM identifies **coherent subtopics**, groups of posts connected by an explicit causal or chronological thread, and orders them into a chain (`pioneer` → `offspring`). Posts it cannot confidently place are excluded and forwarded to ADEPT. Ordering is chronological for `content_type: social_media` and thematic for `content_type: document`. This stage is LLM-driven; there is no closed-form algorithm to cite beyond the clustering that precedes it. One LLM call per cluster.

## ADEPT — Density Peak Clustering

Nova's orphan posts, the ones no narrative chain claimed, are pooled around density hubs using **Density Peak Clustering (DPC)**. All of a cluster's pools are then labelled together in a single LLM call. For each orphan *i*:

- **Local density** ρᵢ = number of other points within a cutoff distance `dc` (`adept.dpc_dc_percentile`, 1–2% of all pairwise distances per the paper's guidance).
- **Distance to a denser point** δᵢ = min distance to any point *j* with ρⱼ > ρᵢ. For the single densest point overall, δ is the max distance to any other point instead.
- **Decision value** γᵢ = ρᵢ · δᵢ. Points with both high density and unusual distance from anything denser become pool hubs; the threshold is `adept.dpc_gamma_percentile`.
- Every non-hub point joins the pool of the higher-density neighbour defining its δ, walked recursively back to a hub, **except** that any single hop with δ above `adept.dpc_membership_percentile` breaks the chain and leaves the point orphan rather than forcing it into a distant pool.

Pools below `adept.pool_min_size` dissolve back to orphan status; pools above `adept.pool_max_size` keep only their closest candidates by δ. ADEPT emits `adept_spoke` (hub → member) edges within each pool. It does **not** emit the `adept_graft` edges that reconnect a pool to the main narrative graph: those are built afterwards by the edge stage (see [Edges](#edges) below). One LLM call per cluster.

> Alex Rodriguez, Alessandro Laio. **"Clustering by fast search and find of density peaks."** *Science* 344(6191), 2014.

## Edges

Beyond Nova and ADEPT's own edges, three types connect the graph:

- **`temporal_influence`** — directed, from an older, higher-engagement post to a semantically similar (`edges.temporal_influence.semantic_min`), younger one, with an exponential time decay (`tau_hours`) capped at `tmax_hours`. Falls back to an engagement-only ordering if timestamps are absent.
- **`temporal`** — the same idea without engagement weighting, resolved via a two-pass **b-matching** (each node capped at `pass2_max_out` outgoing and `pass2_max_in` incoming edges in the second pass) so no single post accumulates a disproportionate share of the temporal graph. b-matching runs under Numba, which ships in `requirements.txt`. If the import fails for any reason the pass falls back to plain NumPy and logs a warning, so the stage degrades in speed rather than breaking.
- **`semantic_inter`** — a greedy 1-to-1 bridge between two different clusters whenever cosine similarity clears `edges.semantic_inter.threshold`, exempt from b-matching.
- **`adept_graft`** — built here, not by ADEPT: each ADEPT hub is linked to the nearest Nova node **inside its own cluster** when cosine similarity clears `edges.adept_graft.graft_min_cosine` (`compute_adept_graft_edges` in `edges.py`). This is what reattaches an orphan pool to the narrative graph. Exempt from b-matching.

A final **pruning** pass (`prune_cross_component_bridges` in `edges.py`) deduplicates redundant bridges by connected component: a candidate is dropped once its endpoints already share a component, walked via a single shared union-find over all three bridge types in priority order (`temporal_influence` > `temporal` > `semantic_inter`). That ordering is fixed, not configurable. Optionally, a bridge between two otherwise-isolated singleton nodes can be rejected (`edges.pruning.isolation_guard`). Separately, as a final safety net after every edge type is built, any component still above `edges.pruning.max_component_size` is split by cutting its least-disposable edge first.

**These three bridge types carry no layout force in the shipped configuration.** `fa2.edge_weights` sets `temporal`, `temporal_influence`, and `semantic_inter` to `0.0`, and `edges.adept_graft.force_graft` is `0.0`. Those edges are still built, stored, drawn in the 3D view, and traversed by search; they just do not pull nodes together until you raise the weights. Nova chains and ADEPT spokes are the only edges with a non-zero weight.

## Layout — ForceAtlas2 in 3D

The graph is laid out with **ForceAtlas2**, run natively in three dimensions rather than projected from a 2D result, seeded from a **PaCMAP** projection of the discriminant (or raw) embedding space rather than from random positions.

> Mathieu Jacomy, Tommaso Venturini, Sebastien Heymann, Mathieu Bastian. **"ForceAtlas2, a Continuous Graph Layout Algorithm for Handy Network Visualization Designed for the Gephi Software."** *PLOS ONE* 9(6), 2014.
> Yingfan Wang, Haiyang Huang, Cynthia Rudin, Yaron Shaposhnik. **"Understanding How Dimension Reduction Tools Work: An Empirical Approach to Deciphering t-SNE, UMAP, TriMap, and PaCMAP for Data Visualization."** *JMLR* 22, 2021.

On top of the physics simulation, four post-processing passes keep large graphs readable, each with its own keys in [`CONFIGURATION.md`](CONFIGURATION.md#forceatlas2-layout): Nova chain straightening, size-aware node spacing, intra-cluster block reorganisation, and a two-stage cluster-overlap fix (continuous declumping repulsion between centroids, then a hard-threshold cleanup).

## Search retrieval

`search.py` decomposes the query into sub-questions with one LLM call, then runs dense retrieval per sub-question, ranking candidates by cosine similarity against `embeddings_retrieval.parquet`. Candidates are de-duplicated by connected component, so a hit inside an already-selected Nova chain or ADEPT pool does not count twice, trimmed to `--floor-ratio × best_score` for that sub-question, merged across sub-questions, and expanded to whole connected components by BFS before being handed to the answering LLM. Full detail, including the prompt contract, is in [`SEARCH.md`](SEARCH.md).

## See also

- [`CONFIGURATION.md`](CONFIGURATION.md) — every key referenced above
- [`PERFORMANCE.md`](PERFORMANCE.md) — what each stage costs in time and API calls
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — stage order and on-disk layout
- [`SEARCH.md`](SEARCH.md) — the retrieval and prompt contract in full
