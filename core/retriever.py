"""Library-independent retrieval result contract around official GoS calls."""
from dataclasses import dataclass
import json
import shutil
import time
from pathlib import Path
from .storage import atomic_json, data_disk_path


@dataclass(frozen=True)
class RetrievalHit:
    skill_id: str
    name: str
    score: float
    source_path: str
    rank: int = 0
    retriever_type: str = ''
    metadata: dict | None = None


class NoSkillRetriever:
    async def retrieve(self, query, top_k=8): return []


class GoSRetriever:
    def __init__(self, engine, library, *, mode='gos', workspace=None, provider=None, trace_path=None):
        if mode not in ('gos','vector'): raise ValueError('Unknown retrieval mode')
        self.engine, self.library, self.mode = engine, library, mode
        self.last_result = None
        self.workspace = data_disk_path(workspace) if workspace else None
        self.provider = provider
        self.trace_path = data_disk_path(trace_path) if trace_path else None
        self.trace = []
        self.invalid_reason = None

    @classmethod
    def load(cls, workspace, provider, library, *, mode='gos', trace_path=None, **options):
        from baselines.gos_adapter import load_engine
        engine = load_engine(workspace,provider,library,**options)
        return cls(engine,library,mode=mode,workspace=workspace,provider=provider,trace_path=trace_path)

    @classmethod
    async def build(cls, source, workspace, provider, library, *, mode='gos',
                    graph_policy='preserve', llm_service=None, **load_options):
        from baselines.gos_adapter import build_workspace
        await build_workspace(source,workspace,provider,library,
                              graph_policy=graph_policy,llm_service=llm_service)
        return cls.load(workspace,provider,library,mode=mode,**load_options)

    def save(self, target):
        """Copy a verified, persisted official workspace without overwriting."""
        from baselines.gos_adapter import validate_index
        if not self.workspace or not self.provider:
            raise ValueError('save requires a retriever created by build/load')
        self._check_valid()
        validate_index(self.workspace,self.provider,self.library.version)
        target = data_disk_path(target)
        if target.is_relative_to(self.workspace): raise ValueError('Cannot save a workspace inside itself')
        if target.exists(): raise FileExistsError(target)
        if any(p.is_symlink() and not p.resolve().is_relative_to(self.workspace) for p in self.workspace.rglob('*')):
            raise ValueError('Workspace symlink escapes source directory')
        shutil.copytree(self.workspace,target)
        validate_index(target,self.provider,self.library.version)
        return target

    def invalidate(self, reason='explicit invalidation'):
        self.invalid_reason = reason
        if self.workspace:
            atomic_json(self.workspace / 'invalidated.json',{'reason':reason})

    def _check_valid(self):
        if self.invalid_reason or (self.workspace and (self.workspace / 'invalidated.json').exists()):
            raise ValueError('Retriever/index invalidated; rebuild into a fresh workspace')

    async def retrieve(self, query, top_k=8):
        if top_k < 1: raise ValueError('top_k must be positive')
        self._check_valid()
        fn = self.engine.async_retrieve if self.mode == 'gos' else self.engine.async_retrieve_vector
        started = time.perf_counter()
        result = await fn(query, top_n=top_k)
        self.last_result = result
        hits = [RetrievalHit(self.library.get(s.name).skill_id,s.name,s.score,str(self.library.file(self.library.get(s.name).skill_id)),
                             rank,self.mode,s.model_dump(exclude={'payload'}))
                for rank,s in enumerate(result.skills,1)]
        event = {'query':query,'top_k':top_k,'retriever_type':self.mode,
                 'elapsed_seconds':time.perf_counter()-started,'hits':[h.__dict__ for h in hits]}
        self.trace.append(event)
        if self.trace_path:
            self.trace_path.parent.mkdir(parents=True,exist_ok=True)
            with self.trace_path.open('a') as out: out.write(json.dumps(event,ensure_ascii=False,allow_nan=False)+'\n')
        return hits


class VectorRetriever(GoSRetriever):
    def __init__(self, engine, library, *, mode='vector', **options):
        if mode != 'vector': raise ValueError('VectorRetriever requires vector mode')
        super().__init__(engine,library,mode=mode,**options)

    @classmethod
    def load(cls, workspace, provider, library, *, mode='vector', **options):
        if mode != 'vector': raise ValueError('VectorRetriever requires vector mode')
        return super().load(workspace,provider,library,mode=mode,**options)

    @classmethod
    async def build(cls, source, workspace, provider, library, *, mode='vector', **options):
        if mode != 'vector': raise ValueError('VectorRetriever requires vector mode')
        return await super().build(source,workspace,provider,library,mode=mode,**options)
