"""Query-route representations built from one immutable membership partition."""
from dataclasses import dataclass
import hashlib
import json
import numpy as np
from .vector_math import NumpyBackend, unit_rows, stable_topk


@dataclass
class RepresentationView:
    kind: str
    cluster_ids: list
    vectors: np.ndarray
    vector_cluster_ids: list
    representative_ids: dict
    member_skill_ids: dict
    membership_fingerprint: str
    index_budget: dict


def build_representation(membership, vectors, kind, backend=None):
    if kind not in ('centroid', 'medoid', 'rep3', 'rep5') or not membership:
        raise ValueError('Expected nonempty membership and a known representation')
    backend = backend or NumpyBackend()
    members = {}
    for sid, cid in sorted(membership.items()):
        if sid not in vectors or type(cid) is not int or cid < 0:
            raise ValueError('Invalid cluster member mapping')
        members.setdefault(cid, []).append(sid)
    selected, labels, representatives = [], [], {}
    for cid, ids in sorted(members.items()):
        matrix = unit_rows([vectors[sid] for sid in ids])
        total = matrix.sum(axis=0)
        if np.linalg.norm(total) < 1e-12:
            raise ValueError('Cluster center cancels to zero')
        center = unit_rows(total)[0]
        if kind == 'centroid':
            selected.append(center)
            labels.append(cid)
            continue
        # For unit vectors, maximizing total cosine equals maximizing dot(sum).
        medoid = stable_topk(backend.scores(center, matrix)[0], ids, 1)[0]
        chosen = [medoid]
        budget = int(kind[-1]) if kind.startswith('rep') else 1
        best = np.full(len(ids), -np.inf)
        while len(chosen) < min(budget, len(ids)):
            best = np.maximum(best, backend.scores(matrix[chosen[-1]], matrix)[0])
            remaining = [i for i in range(len(ids)) if i not in chosen]
            next_index = min(remaining, key=lambda i: (float(best[i]), ids[i]))
            chosen.append(next_index)
        representatives[cid] = [ids[i] for i in chosen]
        selected.extend(matrix[chosen])
        labels.extend([cid] * len(chosen))
    output = np.asarray(selected, dtype=backend.dtype)
    fingerprint = hashlib.sha256(json.dumps(sorted(membership.items())).encode()).hexdigest()
    return RepresentationView(kind, sorted(members), output, labels, representatives, members, fingerprint,
                              {'vector_count': len(output), 'vector_bytes': output.nbytes,
                               'score_dtype': backend.dtype.name, 'representation': kind})
