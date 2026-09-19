# Performance and cost

What drives wall-clock time and LLM call volume as corpus size and cluster count grow, and which levers control each one.

> The runtime figures below are indicative, not benchmarks. Actual performance depends on CPU/GPU, corpus characteristics, provider latency and rate limits, and the configured `nova.parallel_clusters` / `adept.parallel_clusters` values.

## Clustering: cluster-count estimation and ADR

Two compute-heavy stages run before any LLM analysis.

- **Cluster-count estimation** (`clustering.k_estimation_method`, shipped `gmm_bic`) fits a Gaussian Mixture Model for each candidate K in the sweep around the HDBSCAN pivot.
- **ADR**, the LDA-GO and GMM refinement loop, is typically the larger of the two. Each pass operates on the full corpus, and several passes may be needed before convergence, so cost grows with corpus size and can also grow with the target cluster count. See [`ALGORITHMS.md`](ALGORITHMS.md#adr--adaptive-discriminant-refinement) for the stopping criteria.

**Typical runtime.** On the currently tested hardware, clustering (estimation + ADR) takes a few minutes for around 10,000 posts. At around 50,000 posts it can take several hours. These figures are workload- and hardware-dependent and are not guarantees.

| Lever | Effect |
|---|---|
| `--k-fast` | Skip the GMM-BIC sweep. Faster estimation, less refined K. |
| `--set-k N` | Skip automatic estimation entirely and force a cluster count. |
| `--use-k-cache` | Reuse a previous run's K and seed labels from `k_cache.pkl`. |
| `--limit N` | Cap the input at N posts. Useful for smoke-testing the rest of the pipeline before the full corpus. |
| `adr.max_iter` | Ceiling on refinement passes. The loop usually stops earlier on its own. |
| `adr.gmm_repetitions` | GMM restarts per pass. Lower is faster, at some risk of a worse local optimum. |
| `adr.ldago_device` | `"cuda"` in the shipped config. Set it to `"cpu"` or `null` on a machine without a CUDA-enabled build. |

## LLM analysis: Nova & ADEPT

In `full` mode both stages run per cluster.

- **Nova** makes one call per cluster to build that cluster's subtopic chain.
- **ADEPT** makes at most one call per cluster, grouping all of that cluster's orphan pools into a single request. A cluster with too few orphans to form a pool is skipped and makes no call.

A 100-cluster `full` run therefore costs at most 200 heavy-slot calls for these two stages, plus 100 light-slot naming calls. `clustering_only` skips both stages and makes zero Nova/ADEPT calls.

> This is a call count, not a cost estimate. Actual cost also depends on input and output tokens per request and on your provider's pricing. With the shipped `models.heavy` (`num_ctx: 1048576`, `max_tokens: 131072`, reasoning effort high), individual requests can be large.

| Lever | Effect |
|---|---|
| `--run-mode clustering_only` | Skip Nova and ADEPT entirely; no analysis-stage LLM calls. |
| [`nova.parallel_clusters`, `adept.parallel_clusters`](CONFIGURATION.md#run-mode-and-module-routing) | How many clusters are processed concurrently. Shipped default is `1` (sequential) — the safe starting point, and required for local models. On a fast API model with many clusters, raising it (`20`-`50` is common) reduces wall-clock time, but increases contention and can trigger rate limits. |
| `adept.min_orphans_for_event` | Raising it skips ADEPT on clusters with few orphans, cutting calls. |
| `nova.max_text_chars` | Post truncation inside the Nova prompt; the main input-token lever. |

## Cluster naming

Naming is a separate stage and also makes one LLM call per cluster, when the configured naming model is reachable.

By default the `light` slot is the local Ollama model `qwen3.5:9b`, so naming adds no API cost in the shipped configuration. If that model is unreachable, naming falls back to c-TF-IDF-only labels and makes no call at all. See [`ALGORITHMS.md`](ALGORITHMS.md#naming).

`--skip-naming` skips the stage; `reload.py relabel` redoes it alone, without recomputing anything else.

## Search

`search.py` is the only recurring API cost after a run finishes, and it is charged to the **heavy** slot.

Two calls per question: one to decompose the query, one to generate the report. A third is spent only when the first report fails contract validation, since `analyze_with_llm` then resends the same prompt once (see [`SEARCH.md`](SEARCH.md#validation-and-retry)).

Prompt size is driven by `--budget` (default 3000 posts) at up to 500 characters per post, so a large budget on a dense graph produces a large request. `--dry-run` reports the exact prompt size in characters without spending a call, which is the cheapest way to check before running a costly query.

| Lever | Effect |
|---|---|
| `--budget` | Caps total posts in the prompt. Components are never split, so the real total can exceed it slightly. |
| `--floor-ratio` | Raising it keeps fewer seeds, so fewer components are expanded. |

## See also

- [`CONFIGURATION.md`](CONFIGURATION.md) — every key named above
- [`ALGORITHMS.md`](ALGORITHMS.md) — why each stage costs what it costs
- [`CLI.md`](CLI.md) — the flags used as levers here
- [`SEARCH.md`](SEARCH.md) — what search spends its calls on
