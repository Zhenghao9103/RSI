"""Append-only adaptive clusters with shared membership and two query routes."""
import hashlib
import json
import shutil
import time
import numpy as np
from .storage import data_disk_path,atomic_json,sha256_file


def positive_integer(value,name):
    if type(value) is not int or value<1:raise ValueError(f'{name} must be a positive integer')


class OnlineClusterIndex:
    def __init__(self,dimension,*,threshold,representatives,math_backend=None):
        positive_integer(dimension,'dimension');positive_integer(representatives,'representatives')
        if isinstance(threshold,bool) or not isinstance(threshold,(float,int)) or not np.isfinite(threshold) or not -1<=threshold<=1:
            raise ValueError('threshold must be finite and in [-1,1]')
        self.dimension,self.threshold,self.representatives=dimension,float(threshold),representatives
        self._vectors={};self._clusters=[];self._membership={};self._fingerprint=None
        self.provenance=None
        self.math_backend=math_backend

    @property
    def membership(self):return dict(self._membership)

    @property
    def cluster_count(self):return len(self._clusters)

    @property
    def representative_ids(self):return {i:list(c['representatives']) for i,c in enumerate(self._clusters)}

    def get_vector(self,skill_id):
        """Read a member vector without exposing mutable index state."""
        return self._vectors[skill_id].copy()

    def _unit(self,vector):
        value=np.asarray(vector,dtype=np.float64)
        if value.shape!=(self.dimension,) or not np.isfinite(value).all():
            raise ValueError('Invalid vector dimension or finite values')
        norm=float(np.linalg.norm(value))
        if not np.isfinite(norm) or norm<1e-12:raise ValueError('Vector norm must be nonzero')
        return value/norm

    def _representatives(self,members,vectors,center):
        ids=sorted(members);matrix=np.vstack([vectors[sid] for sid in ids])
        similarities=lambda v: (matrix@v if self.math_backend is None else self.math_backend.scores(v,matrix)[0])
        chosen=[int(np.argmax(similarities(center)))];dots=len(ids)
        best=np.full(len(ids),-np.inf)
        while len(chosen)<min(self.representatives,len(ids)):
            best=np.maximum(best,similarities(matrix[chosen[-1]]));dots+=len(ids)
            distance=1-best;distance[chosen]=-np.inf
            chosen.append(int(np.argmax(distance)))
        return [ids[i] for i in chosen],dots

    def add(self,skill_id,vector):
        if not isinstance(skill_id,str) or not skill_id.strip():raise ValueError('Skill ID must be nonempty')
        if skill_id in self._vectors:raise ValueError('Duplicate Skill ID')
        started=time.perf_counter();value=self._unit(vector)
        if self.math_backend is not None and self._clusters:
            raw=self.math_backend.scores(value,np.vstack([c['center'] for c in self._clusters]))[0]
            scores=[float(np.clip(s,-1,1)) for s in raw]
        else:scores=[float(np.clip(value@c['center'],-1,1)) for c in self._clusters]
        best=max(range(len(scores)),key=lambda i:(scores[i],-i)) if scores else None
        similarity=scores[best] if best is not None else None
        created=best is None or similarity<self.threshold
        cluster_id=len(self._clusters) if created else best
        previous=None if created else self._clusters[cluster_id]
        members=[] if created else list(previous['members'])
        members.append(skill_id)
        vector_sum=value.copy() if created else previous['sum']+value
        norm=float(np.linalg.norm(vector_sum))
        if norm<1e-12:raise ValueError('Cluster center cancelled to zero; insertion rejected')
        center=vector_sum/norm
        base_seconds=time.perf_counter()-started
        representative_start=time.perf_counter()
        proposed=dict(self._vectors);proposed[skill_id]=value
        representatives,dots=self._representatives(members,proposed,center)
        representative_seconds=time.perf_counter()-representative_start
        cluster={'members':members,'sum':vector_sum,'center':center,'representatives':representatives}
        # All validation and calculations precede the state mutation.
        self._vectors[skill_id]=value;self._membership[skill_id]=cluster_id
        if created:self._clusters.append(cluster)
        else:self._clusters[cluster_id]=cluster
        self._fingerprint=None
        return {'skill_id':skill_id,'cluster_id':cluster_id,'created_cluster':created,
            'best_existing_similarity':similarity,'clusters':self.cluster_count,'skills':len(self._vectors),
            'assignment_dot_products':len(scores),'representative_dot_products':dots,
            'centroid_update_seconds':base_seconds,'representative_update_seconds':representative_seconds,
            'elapsed_seconds':time.perf_counter()-started}

    def fingerprint(self):
        if self._fingerprint is None:
            state={'dimension':self.dimension,'threshold':self.threshold,'representatives':self.representatives,
                   'ids':list(self._vectors),'membership':self._membership,
                   'clusters':[{'members':c['members'],'representatives':c['representatives']} for c in self._clusters]}
            digest=hashlib.sha256(json.dumps(state,sort_keys=True,separators=(',',':')).encode())
            for vector in self._vectors.values():digest.update(vector.astype('<f8').tobytes())
            self._fingerprint=digest.hexdigest()
        return self._fingerprint

    def export_state(self):
        """Small resumable state referencing immutable external member vectors."""
        return {'version':1,'dimension':self.dimension,'threshold':self.threshold,
                'representatives':self.representatives,'skill_ids':list(self._vectors),
                'members':[list(c['members']) for c in self._clusters],
                'representative_ids':self.representative_ids,'fingerprint':self.fingerprint()}

    @classmethod
    def restore_state(cls,state,vectors,*,math_backend=None):
        if state.get('version')!=1:raise ValueError('Unknown runtime cluster state')
        index=cls(state['dimension'],threshold=state['threshold'],representatives=state['representatives'],
                  math_backend=math_backend)
        ids=state['skill_ids']
        if len(set(ids))!=len(ids) or any(sid not in vectors for sid in ids):
            raise ValueError('Runtime state references unknown or duplicate Skills')
        index._vectors={sid:index._unit(vectors[sid]) for sid in ids}
        for cid,members in enumerate(state['members']):
            if not members or len(set(members))!=len(members) or any(sid not in index._vectors or sid in index._membership for sid in members):
                raise ValueError('Invalid runtime member partition')
            total=np.zeros(index.dimension)
            for sid in members:total+=index._vectors[sid];index._membership[sid]=cid
            norm=float(np.linalg.norm(total))
            if norm<1e-12:raise ValueError('Invalid runtime center')
            reps=state['representative_ids'].get(cid,state['representative_ids'].get(str(cid)))
            if not reps or len(reps)!=len(set(reps)) or len(reps)>index.representatives or any(sid not in members for sid in reps):
                raise ValueError('Invalid runtime representative IDs')
            index._clusters.append({'members':list(members),'sum':total,'center':total/norm,'representatives':list(reps)})
        if set(index._membership)!=set(ids) or index.fingerprint()!=state['fingerprint']:
            raise ValueError('Runtime state fingerprint mismatch')
        return index

    def search(self,query_vector,*,route_clusters,top_k,representation):
        positive_integer(route_clusters,'route_clusters');positive_integer(top_k,'top_k')
        if representation not in ('centroid','representatives'):raise ValueError('Unknown route representation')
        query=self._unit(query_vector);signature=self.fingerprint()
        started=time.perf_counter();scores=[];dots=0
        for i,cluster in enumerate(self._clusters):
            if representation=='centroid':score=float(query@cluster['center']);dots+=1
            else:
                score=max(float(query@self._vectors[sid]) for sid in cluster['representatives'])
                dots+=len(cluster['representatives'])
            scores.append((i,score))
        selected=sorted(scores,key=lambda item:(-item[1],item[0]))[:route_clusters]
        route_seconds=time.perf_counter()-started
        started=time.perf_counter()
        candidates=[sid for cluster_id,_ in selected for sid in self._clusters[cluster_id]['members']]
        hits=[{'skill_id':sid,'score':float(np.clip(query@self._vectors[sid],-1,1))} for sid in candidates]
        hits=sorted(hits,key=lambda hit:(-hit['score'],hit['skill_id']))[:top_k]
        for rank,hit in enumerate(hits,1):hit['rank']=rank
        return {'representation':representation,'cluster_ids':[i for i,_ in selected],
            'cluster_scores':[score for _,score in selected],'candidate_count':len(candidates),
            'candidate_skill_ids':candidates,'hits':hits,'route_dot_products':dots,
            'within_cluster_dot_products':len(candidates),'route_seconds':route_seconds,
            'within_cluster_seconds':time.perf_counter()-started,'state_fingerprint':signature}

    def save(self,path,*,provenance=None):
        path=data_disk_path(path);path.mkdir(parents=True,exist_ok=False)
        try:
            ids=list(self._vectors)
            matrix=np.vstack(list(self._vectors.values())) if ids else np.empty((0,self.dimension))
            np.save(path/'vectors.npy',matrix,allow_pickle=False)
            manifest={'version':1,'dimension':self.dimension,'threshold':self.threshold,'representatives':self.representatives,
                'skill_ids':ids,'members':[c['members'] for c in self._clusters],
                'representative_ids':[c['representatives'] for c in self._clusters],
                'vectors_sha256':sha256_file(path/'vectors.npy'),'fingerprint':self.fingerprint()}
            manifest['provenance']=self.provenance if provenance is None else provenance
            atomic_json(path/'manifest.json',manifest)
            return manifest
        except BaseException:
            shutil.rmtree(path);raise

    @classmethod
    def load(cls,path):
        path=data_disk_path(path);manifest=json.loads((path/'manifest.json').read_text())
        vector_path=data_disk_path(path/'vectors.npy')
        if not vector_path.is_relative_to(path) or sha256_file(vector_path)!=manifest['vectors_sha256']:
            raise ValueError('Snapshot vector hash mismatch')
        if manifest.get('version')!=1:raise ValueError('Unsupported snapshot version')
        index=cls(manifest['dimension'],threshold=manifest['threshold'],representatives=manifest['representatives'])
        index.provenance=manifest.get('provenance')
        ids=manifest['skill_ids'];vectors=np.load(vector_path,allow_pickle=False)
        if (len(set(ids))!=len(ids) or any(not isinstance(sid,str) or not sid.strip() for sid in ids)
            or vectors.shape!=(len(ids),index.dimension) or not np.isfinite(vectors).all()
            or not np.allclose(np.linalg.norm(vectors,axis=1),1,atol=1e-10)):
            raise ValueError('Invalid snapshot vectors or Skill IDs')
        index._vectors={sid:vector.copy() for sid,vector in zip(ids,vectors)}
        for cluster_id,members in enumerate(manifest['members']):
            if not members or len(set(members))!=len(members) or any(sid not in index._vectors or sid in index._membership for sid in members):
                raise ValueError('Invalid snapshot cluster membership')
            vector_sum=np.zeros(index.dimension)
            for sid in members:vector_sum=vector_sum+index._vectors[sid];index._membership[sid]=cluster_id
            norm=np.linalg.norm(vector_sum)
            if norm<1e-12:raise ValueError('Invalid snapshot center')
            center=vector_sum/norm
            representatives,_=index._representatives(members,index._vectors,center)
            index._clusters.append({'members':list(members),'sum':vector_sum,'center':center,'representatives':representatives})
        if (set(index._membership)!=set(ids) or index.representative_ids!={i:r for i,r in enumerate(manifest['representative_ids'])}
            or index.fingerprint()!=manifest['fingerprint']):
            raise ValueError('Snapshot state fingerprint mismatch')
        return index
