"""
Univers-Graphe v6.1+ADR — Package src

Modules du pipeline (dans l'ordre d'execution):
  _1_embedding  : Qwen3 -> embeddings 384d/4096d + cache Redis
  _2_adr        : Adaptive Discriminant Refinement (ADR) — LDA + GMM
  _2_hdbscan    : Clustering semantique HDBSCAN sur PCA 64d
  _3_leiden     : Creation des galaxies (Jour 0 uniquement)
  _4_fastlpa     : Maintenance incrementale des galaxies (Jour N+1)
  _5_edges       : Calcul O(n^2) des 3 types d'edges
  _6_fa2         : Layout 3D + memoire des positions
  _7_render      : Plotly (MVP) ou fastplotlib (GPU)
"""
