"""Attach optional NOVA / ADEPT-pool description columns to posts_df.

Reads nova_metadata.parquet and pool_explanations.parquet from
cfg["storage"]["processed_dir"] if present. Missing files are not an
error — the columns stay None.
"""

from pathlib import Path

import pandas as pd

from utils.logger import get_logger

from .coerce import ss

log = get_logger(__name__)


def attach_metadata(posts_df: pd.DataFrame, edges_df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    posts_df = posts_df.copy()
    posts_df["adept_pool_label"] = None
    posts_df["adept_pool_hub_id"] = None
    posts_df["nova_title"] = None
    posts_df["nova_reasoning"] = None
    posts_df["nova_sentiment_arc"] = None
    posts_df["pool_reasoning"] = None

    processed_dir = Path(cfg["storage"].get("processed_dir", "data/processed"))

    try:
        nova_meta_path = processed_dir / "nova_metadata.parquet"
        if nova_meta_path.exists():
            nova_meta = pd.read_parquet(nova_meta_path)
            if "subtopic_id" in nova_meta.columns:
                nova_meta = nova_meta.copy()
                nova_meta["subtopic_id"] = nova_meta["subtopic_id"].astype(str)
                nova_by_id = nova_meta.drop_duplicates("subtopic_id").set_index("subtopic_id").to_dict("index")
                if "subtopic_id" in posts_df.columns:
                    key = posts_df["subtopic_id"].astype(str)
                    posts_df["nova_title"] = key.map(lambda k: nova_by_id.get(k, {}).get("title"))
                    posts_df["nova_reasoning"] = key.map(lambda k: nova_by_id.get(k, {}).get("reasoning"))
                    posts_df["nova_sentiment_arc"] = key.map(lambda k: nova_by_id.get(k, {}).get("sentiment_arc"))
                log.info("NOVA metadata loaded from %s", nova_meta_path.name)
            else:
                log.warning("%s found but subtopic_id column is missing — ignored", nova_meta_path.name)
        else:
            log.info("%s missing — no NOVA description", nova_meta_path.name)
    except Exception as e:
        log.warning("nova_metadata.parquet ignored (%s: %s)", type(e).__name__, e, exc_info=True)

    try:
        pool_exp_path = processed_dir / "pool_explanations.parquet"
        if pool_exp_path.exists():
            pool_exp = pd.read_parquet(pool_exp_path)
            if "hub_id" in pool_exp.columns:
                pool_exp = pool_exp.copy()
                pool_exp["hub_id"] = pool_exp["hub_id"].astype(str)
                pool_by_hub = pool_exp.drop_duplicates("hub_id").set_index("hub_id").to_dict("index")
                hub_to_title = {h: r.get("pool_title") for h, r in pool_by_hub.items()}
                hub_to_reasoning = {h: r.get("pool_reasoning") for h, r in pool_by_hub.items()}

                adept_adj = {}
                for _, er in edges_df.iterrows():
                    if ss(er.get("type", "")) != "adept_spoke":
                        continue
                    a, b = str(er["source"]), str(er["target"])
                    adept_adj.setdefault(a, set()).add(b)
                    adept_adj.setdefault(b, set()).add(a)

                # id -> adept_role, precomputed once (avoids repeated
                # posts_df.loc[...] scans and the ambiguous-Series-truth
                # crash that used to happen when a component's hub wasn't
                # a key in pool_by_hub, e.g. a hub not yet LLM-explained).
                id_to_role = dict(
                    zip(posts_df["id"].astype(str), posts_df["adept_role"].astype(str), strict=False)
                ) if "adept_role" in posts_df.columns else {}

                id_to_hub = {}
                seen = set()
                for start in adept_adj:
                    if start in seen:
                        continue
                    comp = {start}
                    queue = [start]
                    seen.add(start)
                    for cur in queue:
                        for nxt in adept_adj.get(cur, set()):
                            if nxt not in seen:
                                seen.add(nxt)
                                comp.add(nxt)
                                queue.append(nxt)
                    hubs = [x for x in comp if x in pool_by_hub]
                    if not hubs:
                        hubs = [x for x in comp if id_to_role.get(x) == "hub"]
                    if hubs:
                        hub = sorted(hubs)[0]
                        for x in comp:
                            id_to_hub[x] = hub

                posts_df["adept_pool_hub_id"] = posts_df["id"].astype(str).map(id_to_hub)
                posts_df["adept_pool_label"] = posts_df["adept_pool_hub_id"].map(hub_to_title)
                posts_df["pool_reasoning"] = posts_df["adept_pool_hub_id"].map(hub_to_reasoning)
                log.info("%d ADEPT pool explanations loaded from %s", len(pool_by_hub), pool_exp_path.name)
            else:
                log.warning("%s found but hub_id column is missing — ignored", pool_exp_path.name)
        else:
            log.info("%s missing — no ADEPT pool description", pool_exp_path.name)
    except Exception as e:
        log.warning("pool_explanations.parquet ignored (%s: %s)", type(e).__name__, e, exc_info=True)

    return posts_df
