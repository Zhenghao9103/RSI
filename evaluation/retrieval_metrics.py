"""Gold metrics with explicit current-library eligibility and routing attribution."""
from collections import Counter
import math
import numpy as np


def evaluate_query(gold, trace, membership, available_ids, top_k, *, all_skill_ids=None):
    if type(top_k) is not int or top_k < 1:
        raise ValueError('Invalid final Top-K')
    metric_keys = [f'skill_hit_at_{top_k}', f'skill_recall_at_{top_k}', f'mrr_at_{top_k}', f'ndcg_at_{top_k}']
    row = dict(query_id=gold['query_id'], eligible=False, failure_type='unlabeled',
               metric_scope=gold.get('annotation_scope'), stage1_success=None, stage2_success=None,
               cluster_hit=None, cluster_coverage=None, routed_gold_recall=None,
               available_conditioned_recall=None, gold_states={}, **{k: None for k in metric_keys})
    if gold.get('annotation_status') != 'reviewed':
        return row
    targets = gold.get('gold_skill_ids')
    if not isinstance(targets, list) or not targets or len(set(targets)) != len(targets):
        raise ValueError('Reviewed Gold needs nonempty unique Skill IDs')
    targets, available_ids = set(targets), set(available_ids)
    returned = [h.skill_id for h in trace.hits[:top_k]]
    candidates = set(trace.candidate_skill_ids)
    if (len(set(returned)) != len(returned) or not set(returned) <= candidates or not candidates <= available_ids):
        raise ValueError('Result identities or candidate visibility are invalid')
    active = targets & available_ids
    row.update(available_gold_count=len(active), unavailable_gold_count=len(targets-active))
    if all_skill_ids is not None and not targets <= set(all_skill_ids):
        row['failure_type'] = 'mapping_error'
        return row
    if active:
        row['available_conditioned_recall'] = len(active & set(returned))/len(active)
    if not targets <= available_ids:
        row['failure_type'] = 'ineligible'
        return row
    if trace.selected_cluster_ids is not None and not targets <= set(membership):
        row['failure_type'] = 'mapping_error'
        return row
    hit = bool(targets & set(returned))
    row.update(eligible=True, failure_type='retrieval_success' if hit else
               ('stage2_miss' if targets & candidates else 'stage1_miss'),
               stage2_success=hit, routed_gold_recall=len(targets & candidates)/len(targets),
               gold_states={sid: ('returned' if sid in returned else
                                  'routed_but_not_returned' if sid in candidates else 'not_routed') for sid in sorted(targets)})
    row[metric_keys[0]] = int(hit)
    row[metric_keys[1]] = len(targets & set(returned))/len(targets)
    row[metric_keys[2]] = next((1/rank for rank, sid in enumerate(returned, 1) if sid in targets), 0.)
    if trace.selected_cluster_ids is not None:
        gold_clusters = {membership[sid] for sid in targets}
        selected = set(trace.selected_cluster_ids)
        row.update(gold_cluster_ids=sorted(gold_clusters), stage1_success=bool(selected & gold_clusters),
                   cluster_hit=int(bool(selected & gold_clusters)),
                   cluster_coverage=len(selected & gold_clusters)/len(gold_clusters))
    grades = gold.get('relevance_grades')
    if grades is not None:
        if set(grades) != targets or any(isinstance(v, bool) or not isinstance(v, (int, float)) or
                                        not math.isfinite(v) or not 0 < v <= 100 for v in grades.values()):
            raise ValueError('nDCG needs finite positive grades for every Gold Skill')
        gain = lambda value: 2**value-1
        dcg = sum(gain(grades.get(sid, 0))/math.log2(rank+1) for rank, sid in enumerate(returned, 1))
        ideal = sum(gain(g)/math.log2(rank+1) for rank, g in
                    enumerate(sorted(grades.values(), reverse=True)[:top_k], 1))
        row[metric_keys[3]] = dcg/ideal
    return row


def summarize_queries(rows):
    eligible = [r for r in rows if r.get('eligible')]
    output = {'queries': len(rows), 'eligible_queries': len(eligible),
              'failure_counts': dict(Counter(r['failure_type'] for r in rows))}
    keys = {'cluster_hit', 'cluster_coverage', 'routed_gold_recall'}
    keys.update(k for r in rows for k in r if k.startswith(('skill_hit_at_', 'skill_recall_at_', 'mrr_at_', 'ndcg_at_')))
    for key in sorted(keys):
        values = [r[key] for r in eligible if r.get(key) is not None]
        output[key] = float(np.mean(values)) if values else None
    return output
