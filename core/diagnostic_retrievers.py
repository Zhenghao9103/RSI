"""Shared Top-K contract for direct retrieval diagnostics, independent of Agent caps."""
from collections import Counter
from dataclasses import dataclass, field
import math
import re
import time
import numpy as np
from .retriever import RetrievalHit
from .vector_math import NumpyBackend, stable_topk, unit_rows


@dataclass
class RetrievalTrace:
    method: str
    hits: list
    candidate_skill_ids: list
    selected_cluster_ids: list | None = None
    cluster_scores: list = field(default_factory=list)
    stage_times: dict = field(default_factory=dict)
    index_budget: dict = field(default_factory=dict)
    membership_fingerprint: str | None = None


class DiagnosticRetriever:
    def __init__(self, ids, vectors, *, method='flat_dense', texts=None, library=None, view=None,
                 backend=None, bm25_k1=1.2, bm25_b=.75, rrf_constant=60):
        if method not in ('flat_dense', 'flat_hybrid', 'hierarchical', 'oracle'):
            raise ValueError('Unknown diagnostic method')
        self.ids, self.method, self.view, self.library = list(ids), method, view, library
        if not self.ids or len(set(self.ids)) != len(self.ids):
            raise ValueError('Expected unique nonempty current Skill IDs')
        self.backend = backend or NumpyBackend()
        self.vectors = unit_rows(vectors)
        if len(self.vectors) != len(self.ids):
            raise ValueError('Vector identity count mismatch')
        self.positions = {sid: i for i, sid in enumerate(self.ids)}
        self.matrix = self.backend.prepare(self.vectors)
        self.route_matrix = None
        if method in ('hierarchical', 'oracle'):
            if view is None or set(sid for ids_ in view.member_skill_ids.values() for sid in ids_) != set(self.ids):
                raise ValueError('Route membership differs from current available Skills')
            self.route_matrix = self.backend.prepare(view.vectors)
        self.k1, self.b, self.rrf_constant = bm25_k1, bm25_b, rrf_constant
        if not (math.isfinite(bm25_k1) and bm25_k1 > 0 and 0 <= bm25_b <= 1 and rrf_constant > 0):
            raise ValueError('Invalid BM25/RRF parameters')
        if method == 'flat_hybrid':
            if texts is None or len(texts) != len(self.ids):
                raise ValueError('Hybrid needs the same current Skill text collection')
            self.tokens = [Counter(self.tokenize(text)) for text in texts]
            self.lengths = np.asarray([sum(t.values()) for t in self.tokens])
            self.avgdl = float(self.lengths.mean()) or 1.
            self.df = Counter(term for document in self.tokens for term in document)

    @staticmethod
    def tokenize(text):
        return re.findall(r'\w+', text.casefold(), flags=re.UNICODE)

    def _bm25(self, query):
        scores = np.zeros(len(self.ids))
        for term in set(self.tokenize(query)):
            df = self.df[term]
            if not df:
                continue
            idf = math.log(1 + (len(self.ids)-df+.5)/(df+.5))
            tf = np.asarray([doc[term] for doc in self.tokens])
            scores += idf * tf * (self.k1+1) / (tf+self.k1*(1-self.b+self.b*self.lengths/self.avgdl))
        return scores

    def search(self, query_vector, *, query_text, top_k=5, route_k=None, oracle_cluster_ids=None):
        if type(top_k) is not int or top_k < 1:
            raise ValueError('top_k must be positive')
        query = unit_rows(query_vector)
        if query.shape != (1, self.vectors.shape[1]):
            raise ValueError('Expected one query in the same vector space')
        self.backend.synchronize()
        started = time.perf_counter()
        selected, cluster_scores = None, []
        if self.method == 'hierarchical':
            if type(route_k) is not int or route_k < 1:
                raise ValueError('route_k must be positive')
            route_scores = self.backend.scores(query, self.route_matrix)[0]
            by_cluster = {cid: max(float(route_scores[i]) for i, c in enumerate(self.view.vector_cluster_ids) if c == cid)
                          for cid in self.view.cluster_ids}
            order = stable_topk([by_cluster[c] for c in self.view.cluster_ids], self.view.cluster_ids, route_k)
            selected = [self.view.cluster_ids[i] for i in order]
            cluster_scores = [by_cluster[c] for c in selected]
        elif self.method == 'oracle':
            if oracle_cluster_ids is None or any(c not in self.view.cluster_ids for c in oracle_cluster_ids):
                raise ValueError('Oracle needs valid Gold cluster IDs')
            selected = sorted(set(oracle_cluster_ids))
        self.backend.synchronize()
        route_seconds = time.perf_counter()-started
        second = time.perf_counter()
        candidates = self.ids if selected is None else sorted({sid for c in selected for sid in self.view.member_skill_ids[c]})
        indices = [self.positions[sid] for sid in candidates]
        scores = (self.backend.scores(query, self.backend.take(self.matrix, indices))[0]
                  if candidates else np.empty(0))
        if self.method == 'flat_hybrid':
            dense_order = stable_topk(scores, candidates, len(candidates))
            lexical = self._bm25(query_text)
            scores = np.zeros(len(candidates))
            for rank, i in enumerate(dense_order, 1):
                scores[i] += 1/(self.rrf_constant+rank)
            lexical_order = [i for i in stable_topk(lexical, candidates, len(candidates)) if lexical[i] > 0]
            for rank, i in enumerate(lexical_order, 1):
                scores[i] += 1/(self.rrf_constant+rank)
        hits = []
        for rank, i in enumerate(stable_topk(scores, candidates, top_k), 1):
            sid = candidates[i]
            name = self.library.get(sid).name if self.library else sid
            path = str(self.library.file(sid)) if self.library else ''
            hits.append(RetrievalHit(sid, name, float(scores[i]), path, rank, self.method,
                                    {'score_type': 'rrf' if self.method == 'flat_hybrid' else 'cosine'}))
        self.backend.synchronize()
        budget = dict(self.view.index_budget) if self.view else {'vector_count': 0, 'vector_bytes': 0}
        budget.update(oracle=self.method == 'oracle', candidate_count=len(candidates),
                      skill_vector_count=len(self.ids),
                      skill_vector_bytes=len(self.ids)*self.vectors.shape[1]*self.backend.dtype.itemsize,
                      vector_count_definition='first_level_route_vectors_only',
                      retrieved_cluster_count=len(selected) if selected is not None else None)
        return RetrievalTrace(self.method, hits, list(candidates), selected, cluster_scores,
                              {'stage1_seconds': route_seconds, 'stage2_seconds': time.perf_counter()-second,
                               'total_retrieval_seconds': time.perf_counter()-started}, budget,
                              self.view.membership_fingerprint if self.view else None)
