import json
import numpy as np
import pytest


def make_index(**settings):
    from core.online_clusters import OnlineClusterIndex
    return OnlineClusterIndex(2,threshold=settings.get('threshold',0.8),representatives=settings.get('representatives',2))


def test_stream_forms_growing_clusters_and_single_membership():
    index=make_index()
    a=index.add('a',[1,0]);b=index.add('b',[1,.1]);c=index.add('c',[0,1])
    assert a['created_cluster'] and not b['created_cluster'] and c['created_cluster']
    assert index.membership=={'a':0,'b':0,'c':1}
    assert index.cluster_count==2 and a['assignment_dot_products']==0
    assert c['assignment_dot_products']==1


def test_two_routes_share_state_and_multicluster_results_use_global_score():
    index=make_index();index.add('a',[1,0]);index.add('c',[0,1]);index.add('b',[.99,.1])
    before=index.fingerprint()
    for representation in ['centroid','representatives']:
        result=index.search([1,0],route_clusters=10,top_k=10,representation=representation)
        assert len(result['cluster_ids'])==2 and result['candidate_count']==3
        assert [h['skill_id'] for h in result['hits']]==['a','b','c']
        assert result['hits'][0]['score']==pytest.approx(1)
        assert result['state_fingerprint']==before
    assert index.fingerprint()==before


def test_representatives_can_route_to_minority_member_missed_by_centroid():
    index=make_index(threshold=0.5,representatives=2)
    # x-majority cluster absorbs a 60-degree minority; a 90-degree cluster
    # attracts the minority query by centroid, but loses to its exact representative.
    for sid in ['a','b','c']:index.add(sid,[1,0])
    query=[.5,np.sqrt(.75)]
    index.add('y',query);index.add('z',[0,1])
    result=index.search(query,route_clusters=1,top_k=1,representation='representatives')
    centroid=index.search(query,route_clusters=1,top_k=1,representation='centroid')
    assert centroid['hits'][0]['skill_id']=='z' and centroid['cluster_ids']!=result['cluster_ids']
    assert result['hits'][0]['skill_id']=='y'
    assert index.representative_ids[0]==['a','y']


@pytest.mark.parametrize('vector',[[0,0],[float('nan'),1],[1,float('inf')],[1,0,0]])
def test_invalid_insert_does_not_change_existing_state(vector):
    index=make_index();index.add('a',[1,0]);before=index.fingerprint()
    with pytest.raises(ValueError):index.add('bad',vector)
    assert index.fingerprint()==before


def test_duplicate_and_cancelled_centroid_are_transactional():
    index=make_index(threshold=-1);index.add('a',[1,0]);before=index.fingerprint()
    with pytest.raises(ValueError,match='Duplicate'):index.add('a',[0,1])
    with pytest.raises(ValueError,match='center'):index.add('b',[-1,0])
    assert index.fingerprint()==before


def test_empty_query_and_stable_equal_score_order():
    index=make_index(threshold=.9)
    assert index.search([1,0],route_clusters=1,top_k=2,representation='centroid')['hits']==[]
    index.add('z',[1,0]);index.add('a',[1,0])
    assert [h['skill_id'] for h in index.search([1,0],route_clusters=1,top_k=2,representation='centroid')['hits']]==['a','z']


def test_snapshot_reload_preserves_membership_representatives_and_retrieval(tmp_path):
    from core.online_clusters import OnlineClusterIndex
    index=make_index();index.add('a',[1,.1]);index.add('b',[0,1]);index.add('c',[1,0])
    output=tmp_path/'snapshot';index.save(output)
    loaded=OnlineClusterIndex.load(output)
    assert loaded.fingerprint()==index.fingerprint()
    assert loaded.membership==index.membership and loaded.representative_ids==index.representative_ids
    for representation in ['centroid','representatives']:
        left=index.search([1,1],route_clusters=2,top_k=3,representation=representation)
        right=loaded.search([1,1],route_clusters=2,top_k=3,representation=representation)
        assert left['hits']==right['hits'] and left['cluster_ids']==right['cluster_ids']
    with pytest.raises(FileExistsError):index.save(output)
    (output/'vectors.npy').write_bytes(b'corrupted')
    with pytest.raises(ValueError,match='hash'):OnlineClusterIndex.load(output)


@pytest.mark.parametrize('setting',[{'dimension':0},{'threshold':float('nan')},{'threshold':1.1},{'representatives':0},{'representatives':True}])
def test_constructor_rejects_invalid_parameters(setting):
    from core.online_clusters import OnlineClusterIndex
    kwargs={'dimension':2,'threshold':.8,'representatives':2};kwargs.update(setting)
    with pytest.raises(ValueError):OnlineClusterIndex(**kwargs)


@pytest.mark.parametrize('change',[{'route_clusters':0},{'top_k':0},{'representation':'unknown'}])
def test_search_rejects_invalid_parameters(change):
    index=make_index();kwargs={'route_clusters':1,'top_k':1,'representation':'centroid'};kwargs.update(change)
    with pytest.raises(ValueError):index.search([1,0],**kwargs)
