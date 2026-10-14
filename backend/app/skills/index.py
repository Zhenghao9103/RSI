"""Version-filtered skill discovery on the existing Chroma/BGE-M3 infrastructure."""

import math
from pathlib import Path


class SkillIndex:
    def __init__(self, root: Path, embed=None):
        self.root = Path(root).resolve()
        self.embed = embed
        self._dimension = None
        self._collection_value = None

    @property
    def collection(self):
        if self._collection_value is None:
            from ..rag.vector_store import get_or_create_collection

            self._collection_value = get_or_create_collection(
                self.root / "chroma", "skill_pool_objects"
            )
            existing = self._collection_value.get(limit=1, include=["embeddings"])
            embeddings = existing.get("embeddings")
            if embeddings is not None and len(embeddings):
                stored_dimension = len(embeddings[0])
                if self._dimension is not None and self._dimension != stored_dimension:
                    raise ValueError("embedding dimension differs from stored index")
                self._dimension = stored_dimension
        return self._collection_value

    def _normalize(self, vector) -> list[float]:
        try:
            values = [float(value) for value in vector]
        except (TypeError, ValueError) as exc:
            raise ValueError("embedding must be a numeric vector") from exc
        norm = math.hypot(*values)
        if (
            not values
            or not all(math.isfinite(value) for value in values)
            or not math.isfinite(norm)
            or norm == 0
        ):
            raise ValueError("embedding must be finite and nonzero")
        if self._dimension is not None and len(values) != self._dimension:
            raise ValueError("embedding dimension changed")
        self._dimension = len(values)
        return [value / norm for value in values]

    def encode(self, text: str) -> list[float]:
        if self._collection_value is None and (self.root / "chroma").exists():
            _ = self.collection  # Restore the dimension before accepting a candidate.
        if self.embed is None:
            from ..rag.vector_store import embed_text

            vector = embed_text(text)
        else:
            vector = self.embed(text)
        return self._normalize(vector)

    @staticmethod
    def revision_key(kind: str, entity: dict) -> str:
        return f"{kind}:{entity['id']}:{entity['version']}"

    def sync(self, before: dict, after: dict) -> None:
        changes = []
        for kind, bucket in (("skill", "skills"), ("reference", "references")):
            for entity_id, entity in after[bucket].items():
                if entity.get("status") != "active":
                    continue
                previous = before[bucket].get(entity_id)
                if previous == entity:
                    continue
                if previous and previous["version"] == entity["version"]:
                    immutable = ("name", "description", "body", "embedding", "file_path")
                    if any(previous.get(key) != entity.get(key) for key in immutable):
                        raise ValueError("indexed content changed without a new version")
                    if previous.get("status") == "active":
                        continue  # Counters and provenance do not change retrieval vectors.
                changes.append((kind, entity))
        if not changes:
            return
        collection = self.collection
        vectors = [self._normalize(entity["embedding"]) for _, entity in changes]
        collection.upsert(
            ids=[self.revision_key(kind, entity) for kind, entity in changes],
            embeddings=vectors,
            documents=[entity["name"] + "\n" + entity["description"] for _, entity in changes],
            metadatas=[
                {
                    "revision_key": self.revision_key(kind, entity),
                    "kind": kind,
                    "entity_id": entity["id"],
                    "version": entity["version"],
                    "skill_id": entity.get("skill_id", entity["id"]),
                }
                for kind, entity in changes
            ],
        )

    def discover(
        self,
        state: dict,
        query: str,
        limit: int = 3,
        min_similarity: float = 0.65,
        skill_id: str | None = None,
        kind: str = "skill",
    ) -> list[dict]:
        if kind not in ("skill", "reference"):
            raise ValueError("kind must be skill or reference")
        if limit < 1:
            return []
        bucket = "skills" if kind == "skill" else "references"
        entities = {
            self.revision_key(kind, entity): entity
            for entity in state[bucket].values()
            if entity.get("status") == "active"
            and (skill_id is None or entity.get("skill_id", entity["id"]) == skill_id)
            and (
                kind == "skill"
                or state["skills"].get(entity.get("skill_id"), {}).get("status") == "active"
            )
        }
        if not entities:
            return []
        collection = self.collection
        results = collection.query(
            query_embeddings=[self.encode(query)],
            n_results=min(limit, len(entities)),
            where={"revision_key": {"$in": list(entities)}},
            include=["distances"],
        )
        from ..rag.vector_store import distance_to_similarity

        # Chroma 0.5.23 rejects the factory's newer configuration argument and
        # defaults to squared L2. Newer clients expose the actual configuration;
        # legacy clients expose hnsw:space in collection metadata instead.
        configuration = getattr(collection, "configuration", {}) or {}
        metadata = getattr(collection, "metadata", {}) or {}
        space = configuration.get("hnsw", {}).get("space", metadata.get("hnsw:space", "l2"))

        discovered = []
        for revision_key, distance in zip(results["ids"][0], results["distances"][0], strict=True):
            entity = entities.get(revision_key)
            score = distance_to_similarity(distance, space)
            if entity is not None and score >= min_similarity:
                discovered.append(
                    {key: entity[key] for key in ("id", "name", "description", "version")}
                )
                discovered[-1]["score"] = score
        return discovered
