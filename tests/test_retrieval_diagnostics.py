import numpy as np
import pytest


def data():
    ids = ['a', 'b', 'c', 'd']
    vectors = np.array([[1, 0], [.8, .6], [0, 1], [-1, 0]], dtype=np.float64)
    membership = {'a': 0, 'b': 0, 'c': 1, 'd': 2}
    return ids, vectors, membership


def test_backend_stable_ties_shapes_and_invalid_k():
    from core.vector_math import make_vector_backend, stable_topk
    backend = make_vector_backend({'backend': 'numpy', 'score_dtype': 'float64'})
    assert backend.scores([[1, 0]], [[1, 0], [0, 1]]).tolist() == [[1, 0]]
    assert stable_topk([1, 1, 0], ['z', 'a', 'b'], 2) == [1, 0]
    with pytest.raises(ValueError): stable_topk([1], ['a'], True)
    with pytest.raises(ValueError): backend.scores([[float('nan'), 0]], [[1, 0]])


def test_cuda_requirement_never_falls_back(monkeypatch):
    import torch
    from core.vector_math import make_vector_backend
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    with pytest.raises(RuntimeError, match='CUDA'):
        make_vector_backend({'backend': 'torch_cuda', 'device': 'cuda:0'})


@pytest.mark.parametrize('kind', ['centroid', 'medoid', 'rep3', 'rep5'])
def test_four_representations_share_membership_and_respect_current_members(kind):
    from core.cluster_representations import build_representation
    ids, vectors, membership = data()
    view = build_representation(membership, dict(zip(ids, vectors)), kind)
    center = build_representation(membership, dict(zip(ids, vectors)), 'centroid')
    assert view.membership_fingerprint == center.membership_fingerprint
    assert view.cluster_ids == [0, 1, 2]
    if kind != 'centroid':
        for cid, members in view.representative_ids.items():
            assert all(membership[sid] == cid for sid in members)
            assert len(members) <= min(int(kind[-1]) if kind.startswith('rep') else 1,
                                       sum(c == cid for c in membership.values()))
    assert view.index_budget['vector_bytes'] == view.vectors.nbytes


def test_cosine_medoid_matches_pairwise_definition_and_does_not_use_future_vectors():
    from core.cluster_representations import build_representation
    ids, vectors, _ = data()
    membership = {'a': 0, 'b': 0, 'c': 0}
    view = build_representation(membership, dict(zip(ids, vectors)), 'medoid')
    scores = vectors[:3] @ vectors[:3].T
    expected = ids[int(np.argmax(scores.sum(axis=1)))]
    assert view.representative_ids == {0: [expected]}
    assert 'd' not in view.member_skill_ids[0]


def test_hierarchical_queries_only_selected_clusters_and_oracle_covers_multiple_gold():
    from core.cluster_representations import build_representation
    from core.diagnostic_retrievers import DiagnosticRetriever
    ids, vectors, membership = data()
    view = build_representation(membership, dict(zip(ids, vectors)), 'rep3')
    r = DiagnosticRetriever(ids, vectors, method='hierarchical', view=view)
    result = r.search([1, 0], query_text='x', top_k=5, route_k=1)
    assert result.selected_cluster_ids == [0]
    assert result.candidate_skill_ids == ['a', 'b']
    assert [h.skill_id for h in result.hits] == ['a', 'b']
    oracle = DiagnosticRetriever(ids, vectors, method='oracle', view=view)
    result = oracle.search([1, 0], query_text='x', top_k=5, route_k=None, oracle_cluster_ids=[0, 1])
    assert set(result.candidate_skill_ids) == {'a', 'b', 'c'}
    assert result.index_budget['oracle'] is True


def test_flat_exact_and_hybrid_zero_lexical_matches_do_not_invent_evidence():
    from core.diagnostic_retrievers import DiagnosticRetriever
    ids, vectors, _ = data()
    dense = DiagnosticRetriever(ids, vectors)
    hybrid = DiagnosticRetriever(ids, vectors, method='flat_hybrid', texts=['heat', 'clean', 'cool', 'store'])
    left = dense.search([1, 0], query_text='unmatched', top_k=3)
    right = hybrid.search([1, 0], query_text='unmatched', top_k=3)
    assert [h.skill_id for h in left.hits] == ['a', 'b', 'c']
    assert [h.skill_id for h in right.hits] == ['a', 'b', 'c']
    match = hybrid.search([1, 0], query_text='store', top_k=4)
    assert match.hits[0].skill_id == 'd'
    assert left.selected_cluster_ids is None


def gold(ids, **changes):
    return dict(query_id='q', gold_skill_ids=ids, annotation_status='reviewed', reviewer='fixture',
                annotation_scope='exhaustive_within_library', **changes)


def trace(ids, candidates, clusters):
    from core.diagnostic_retrievers import RetrievalTrace
    from core.retriever import RetrievalHit
    return RetrievalTrace('fixture', [RetrievalHit(sid, sid, 1., '', rank) for rank, sid in enumerate(ids, 1)],
                          candidates, clusters)


def test_multigold_metrics_hand_calculation_and_routing_attribution():
    from evaluation.retrieval_metrics import evaluate_query
    t = trace(['x', 'a', 'y', 'b', 'z'], ['x', 'a', 'y', 'b', 'z'], [0])
    membership = {'a': 0, 'b': 0, 'c': 1, 'x': 0, 'y': 0, 'z': 0}
    result = evaluate_query(gold(['a', 'b', 'c']), t, membership, set(membership), 5)
    assert result['skill_hit_at_5'] == 1
    assert result['skill_recall_at_5'] == pytest.approx(2/3)
    assert result['mrr_at_5'] == .5
    assert result['cluster_coverage'] == .5
    assert result['routed_gold_recall'] == pytest.approx(2/3)
    assert result['gold_states']['c'] == 'not_routed'
    assert result['ndcg_at_5'] is None


@pytest.mark.parametrize('candidates,clusters,failure', [(['x'], [1], 'stage1_miss'),
                                                       (['a', 'x'], [0, 1], 'stage2_miss')])
def test_stage_specific_miss(candidates, clusters, failure):
    from evaluation.retrieval_metrics import evaluate_query
    r = evaluate_query(gold(['a']), trace(['x'], candidates, clusters), {'a': 0, 'x': 1}, {'a', 'x'}, 5)
    assert r['failure_type'] == failure and r['skill_hit_at_5'] == 0


def test_future_gold_is_ineligible_and_pending_proposal_is_never_scored():
    from evaluation.retrieval_metrics import evaluate_query
    t = trace(['a'], ['a'], [0])
    r = evaluate_query(gold(['a', 'future']), t, {'a': 0}, {'a'}, 5,
                       all_skill_ids={'a', 'future'})
    assert not r['eligible'] and r['skill_recall_at_5'] is None
    assert r['available_conditioned_recall'] == 1
    pending = {'query_id': 'q', 'gold_skill_ids': [], 'proposed_gold_skill_ids': ['a'], 'annotation_status': 'pending'}
    r = evaluate_query(pending, t, {'a': 0}, {'a'}, 5)
    assert r['failure_type'] == 'unlabeled' and r['skill_hit_at_5'] is None


def test_ndcg_and_macro_summary_exclude_ineligible_queries():
    from evaluation.retrieval_metrics import evaluate_query, summarize_queries
    t = trace(['a', 'b'], ['a', 'b'], None)
    r = evaluate_query(gold(['a', 'b'], relevance_grades={'a': 3, 'b': 1}), t,
                       {'a': 0, 'b': 0}, {'a', 'b'}, 5)
    assert r['ndcg_at_5'] == 1 and r['cluster_hit'] is None
    unknown = dict(r, eligible=False, skill_recall_at_5=None)
    summary = summarize_queries([r, unknown])
    assert summary['eligible_queries'] == 1 and summary['skill_recall_at_5'] == 1
