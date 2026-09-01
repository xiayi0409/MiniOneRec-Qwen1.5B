#!/usr/bin/env python3
"""Evaluate category consistency of nearest-neighbor recall using SID embeddings.

Each item embedding is the mean of its learned SID token embeddings.  The script
recovers fine-grained Amazon categories by exact normalized-title matching
against the official UCSD 2018 metadata, then reports category Precision@K.
"""
import argparse
import gzip
import html
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoTokenizer


def normalize_title(value):
    value = html.unescape(str(value or ""))
    value = unicodedata.normalize("NFKC", value).casefold()
    return re.sub(r"[^a-z0-9]+", " ", value).strip()


def recover_categories(item_meta, amazon_meta_gz):
    wanted = defaultdict(list)
    for item_id, row in item_meta.items():
        wanted[normalize_title(row.get("title"))].append(item_id)
    candidates = defaultdict(list)
    with gzip.open(amazon_meta_gz, "rt", encoding="utf-8") as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = normalize_title(row.get("title"))
            if key in wanted:
                candidates[key].append(row)

    labels = {}
    for key, item_ids in wanted.items():
        rows = candidates.get(key, [])
        for item_id in item_ids:
            local_brand = normalize_title(item_meta[item_id].get("brand"))
            ranked = sorted(
                rows,
                key=lambda row: (
                    normalize_title(row.get("brand")) != local_brand,
                    not bool(row.get("category")),
                ),
            )
            if ranked and ranked[0].get("category"):
                category = ranked[0]["category"]
                if isinstance(category, list) and category:
                    labels[item_id] = [str(x) for x in category]
    return labels


def load_embedding_weight(checkpoint):
    model_path = Path(checkpoint) / "model.safetensors"
    with safe_open(str(model_path), framework="pt", device="cpu") as stream:
        return stream.get_tensor("model.embed_tokens.weight").float()


def random_same_label_probability(labels):
    counts = Counter(labels)
    total = len(labels)
    if total <= 1:
        return 0.0
    return sum(n * (n - 1) for n in counts.values()) / (total * (total - 1))


def evaluate(vectors, parent_labels, leaf_labels, ks):
    vectors = torch.nn.functional.normalize(vectors, dim=1)
    similarities = vectors @ vectors.T
    similarities.fill_diagonal_(-float("inf"))
    neighbors = similarities.topk(max(ks), dim=1).indices
    result = {}
    for label_name, labels in (("parent", parent_labels), ("leaf", leaf_labels)):
        label_match = []
        for row_index, neighbor_ids in enumerate(neighbors.tolist()):
            label_match.append([labels[j] == labels[row_index] for j in neighbor_ids])
        match = torch.tensor(label_match, dtype=torch.float32)
        result[label_name] = {
            "random_baseline": random_same_label_probability(labels),
            **{f"precision@{k}": match[:, :k].mean().item() for k in ks},
        }
    return result, neighbors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--item-meta", required=True)
    parser.add_argument("--amazon-meta-gz", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ks", default="1,5,10,20")
    args = parser.parse_args()

    item_meta = json.load(open(args.item_meta, encoding="utf-8"))
    sid_index = json.load(open(args.index, encoding="utf-8"))
    labels = recover_categories(item_meta, args.amazon_meta_gz)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    embedding_weight = load_embedding_weight(args.checkpoint)

    item_ids = sorted(set(sid_index) & set(labels), key=lambda x: int(x))
    all_vectors = []
    prefix_vectors = []
    for item_id in item_ids:
        tokens = sid_index[item_id]
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        if tokenizer.unk_token_id is not None and tokenizer.unk_token_id in token_ids:
            raise ValueError(f"Unknown SID token for item {item_id}: {tokens}")
        all_vectors.append(embedding_weight[token_ids].mean(dim=0))
        prefix_ids = token_ids[: min(2, len(token_ids))]
        prefix_vectors.append(embedding_weight[prefix_ids].mean(dim=0))

    parent_labels = [labels[item_id][1] if len(labels[item_id]) > 1 else labels[item_id][-1] for item_id in item_ids]
    leaf_labels = [labels[item_id][-1] for item_id in item_ids]
    ks = [int(x) for x in args.ks.split(",")]
    full_metrics, full_neighbors = evaluate(torch.stack(all_vectors), parent_labels, leaf_labels, ks)
    prefix_metrics, _ = evaluate(torch.stack(prefix_vectors), parent_labels, leaf_labels, ks)

    examples = []
    for row_index in range(min(10, len(item_ids))):
        query_id = item_ids[row_index]
        examples.append({
            "query_item_id": query_id,
            "query_title": item_meta[query_id].get("title", ""),
            "query_category": labels[query_id],
            "top5": [
                {
                    "item_id": item_ids[j],
                    "title": item_meta[item_ids[j]].get("title", ""),
                    "category": labels[item_ids[j]],
                }
                for j in full_neighbors[row_index, :5].tolist()
            ],
        })

    output = {
        "name": args.name,
        "items_total": len(item_meta),
        "items_with_category": len(item_ids),
        "category_coverage": len(item_ids) / len(item_meta),
        "parent_unique": len(set(parent_labels)),
        "leaf_unique": len(set(leaf_labels)),
        "full_sid_embedding": full_metrics,
        "semantic_prefix_2_embedding": prefix_metrics,
        "examples": examples,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as stream:
        json.dump(output, stream, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in output.items() if k != "examples"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
