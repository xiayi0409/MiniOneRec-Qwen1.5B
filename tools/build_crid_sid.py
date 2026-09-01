#!/usr/bin/env python3
"""Build a CRID-style SID: semantic prefix + within-cluster popularity rank.

The semantic prefix is copied from a supplied index (normally RQ-KMeans++).
Popularity is computed only from the training interaction split, avoiding
validation/test target leakage.  Rank 0 is the most frequently interacted
item in each semantic cluster; item id is a deterministic tie breaker.
"""
import argparse
import json
import os
from collections import Counter, defaultdict


def read_train_counts(path):
    counts = Counter()
    with open(path, encoding="utf-8") as f:
        next(f, None)
        for line in f:
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 3:
                continue
            _, history, target = fields
            if history.strip():
                counts.update(history.split())
            counts[target] += 1
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-index", required=True)
    ap.add_argument("--train-inter", required=True)
    ap.add_argument("--item-out", required=True)
    ap.add_argument("--metrics-out", required=True)
    args = ap.parse_args()

    with open(args.source_index, encoding="utf-8") as f:
        source = json.load(f)
    counts = read_train_counts(args.train_inter)
    clusters = defaultdict(list)
    for item_id, tokens in source.items():
        if len(tokens) < 2:
            raise ValueError(f"SID for item {item_id} has fewer than two semantic levels")
        clusters[(tokens[0], tokens[1])].append(item_id)

    crid = {}
    max_cluster = 0
    for prefix, ids in clusters.items():
        # Higher training frequency gets a smaller ordinal rank.
        ids.sort(key=lambda x: (-counts.get(str(x), 0), int(x) if str(x).isdigit() else str(x)))
        max_cluster = max(max_cluster, len(ids))
        for rank, item_id in enumerate(ids):
            crid[str(item_id)] = [prefix[0], prefix[1], f"<d_{rank}>"]

    os.makedirs(os.path.dirname(args.item_out), exist_ok=True)
    with open(args.item_out, "w", encoding="utf-8") as f:
        json.dump(crid, f, ensure_ascii=False, indent=2)
    unique = len(set(tuple(v) for v in crid.values()))
    collisions = len(crid) - unique
    metrics = {
        "method": "CRID-inspired popularity rank",
        "source_index": os.path.abspath(args.source_index),
        "train_inter": os.path.abspath(args.train_inter),
        "items": len(crid),
        "semantic_clusters": len(clusters),
        "max_cluster_size": max_cluster,
        "unique_final_sid": unique,
        "final_collision_rate": collisions / len(crid) if crid else 0.0,
        "rank_signal": "training interaction count; item_id tie-break",
        "leakage_policy": "train split only",
    }
    with open(args.metrics_out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
