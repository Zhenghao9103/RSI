from functools import lru_cache
from typing import Literal

from .config import get_settings


@lru_cache(maxsize=1)
def _model(path: str, device: str | None):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(path, device=device)
    model.max_seq_length = 512
    return model


def embed_text(
    text: str, input_type: Literal["document", "query"] = "document"
) -> list[float]:
    # BGE-M3 encodes queries and documents symmetrically.
    settings = get_settings()
    return (
        _model(settings.bge_model, settings.bge_device)
        .encode(text, normalize_embeddings=True)
        .tolist()
    )
