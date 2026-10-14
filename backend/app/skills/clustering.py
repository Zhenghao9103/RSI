"""Historical capability vectors, deliberately separate from master embeddings."""

import math


def normalize(vector):
    values = [float(v) for v in vector]
    if not values or not all(math.isfinite(v) for v in values):
        raise ValueError("embedding must contain finite values")
    norm = math.sqrt(sum(v * v for v in values))
    if norm <= 1e-12:
        raise ValueError("embedding must be nonzero")
    return [v / norm for v in values]


def cosine(left, right):
    if len(left) != len(right):
        raise ValueError("embedding dimension mismatch")
    return sum(a * b for a, b in zip(left, right, strict=True))


def ranked_clusters(state, vector, threshold, limit, exclude=()):
    matches = []
    for cluster in state["clusters"].values():
        if cluster["status"] != "active" or cluster["id"] in exclude:
            continue
        score = cosine(vector, cluster["centroid"])
        if score >= threshold:
            matches.append((score, cluster["id"]))
    return sorted(matches, key=lambda pair: (-pair[0], pair[1]))[:limit]


def recompute_centers(state, cluster_ids):
    for cluster in state["clusters"].values():
        if cluster["status"] != "active" or cluster["id"] not in cluster_ids:
            continue
        vectors = [
            m["embedding"] for m in state["members"].values() if m["cluster_id"] == cluster["id"]
        ]
        if not vectors:
            cluster.update(status="archived", member_count=0, centroid=[], sum_vector=[])
            state["skills"][cluster["master_id"]]["status"] = "archived"
            continue
        dimension = len(vectors[0])
        if any(len(v) != dimension for v in vectors):
            raise ValueError("embedding dimension mismatch")
        total = [sum(v[i] for v in vectors) for i in range(dimension)]
        cluster.update(member_count=len(vectors), sum_vector=total, centroid=normalize(total))
