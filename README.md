# Online clustering and two-stage Skill retrieval

This directory contains the vector-index code used by the no-external-Plan arm of the Skill Evolution experiments. The index assigns each incoming Skill to one cluster with a cosine threshold of 0.7. Retrieval compares a single task embedding with the cluster centroids, selects the top three clusters, and ranks their combined membership by Skill-level cosine similarity to return four hits.

## Quick start

Run the following commands from this directory.

```sh
python -m pip install -r requirements.txt
```

Use Skill vectors and task vectors from the same embedding model and configuration. This example uses small vectors so it runs without a model download or API call. The benchmark used BGE embeddings.

```python
from core.cluster_representations import build_representation
from core.diagnostic_retrievers import DiagnosticRetriever
from core.online_clusters import OnlineClusterIndex

index = OnlineClusterIndex(2, threshold=0.7, representatives=1)
index.add("object-placer", [1.0, 0.0])
index.add("object-picker", [0.9, 0.2])
index.add("cabinet-opener", [0.0, 1.0])
index.add("tool-locator", [-1.0, 0.0])

skill_ids = list(index.membership)
vectors = {skill_id: index.get_vector(skill_id) for skill_id in skill_ids}
centroids = build_representation(index.membership, vectors, "centroid")
retriever = DiagnosticRetriever(
    skill_ids,
    [vectors[skill_id] for skill_id in skill_ids],
    method="hierarchical",
    view=centroids,
)
result = retriever.search([1.0, 0.1], query_text="", route_k=3, top_k=4)
print([hit.skill_id for hit in result.hits])
```

Call `index.add` as new Skill embeddings arrive. Refresh the centroid view and retriever before querying the updated index. The example uses NumPy. The historical GPU evaluation used the optional PyTorch CUDA backend with float64 scoring.

## Historical ALFWorld comparison

At the 750-Skill checkpoint, the original-query two-stage arm matched the full-library vector ranking for all 140 tasks while scoring an average of 75.54 Skill vectors in its second stage. GoS and the no-Plan arm each completed 132 of 140 tasks. Adding the 500-to-750 maintenance batch to execution gives 5,459,494 tokens for GoS and 3,457,156 for the no-Plan arm, a 36.68% reduction under that accounting. GoS used dynamic retrieval, while the no-Plan arm used a fixed initial candidate pool.

The release accepts precomputed vectors and includes the index, ranker, and regression tests. Full benchmark reproduction also requires the separate BGE encoding, Skill library, ALFWorld, Agent, and GoS experiment assets. The inherited snapshot helpers expect a writable path on the experiment data disk under `/mnt`.

To run the original regression tests, install pytest and PyTorch in the same environment. PyTorch is used by the CUDA availability test. Set the temporary directory to a writable location supported by the snapshot helpers.

```sh
python -m pytest -q -p no:cacheprovider --basetemp /mnt/path-to-rsi-test-tmp tests
```
