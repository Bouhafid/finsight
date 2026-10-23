"""Semantic search over 10-K chunks stored in pgvector."""

from functools import lru_cache

from finsight.config import COMPANY, EMBEDDING_MODEL, FORM_TYPE, PIPELINE_VERSION
from finsight.models import Source

# Top-k chunks per fiscal year, from the latest chunk version of each 10-K,
# so every requested year is represented in the evidence.
QUERY = """
    SELECT d.exercice, c.chunk_key, c.texte, c.kind, c.section,
           c.source_url, c.similarity
    FROM documents d
    JOIN LATERAL (
        SELECT id
        FROM chunk_versions
        WHERE document_id = d.id AND pipeline_version = %(version)s
        ORDER BY created_at DESC, id DESC
        LIMIT 1
    ) v ON TRUE
    JOIN LATERAL (
        SELECT chunk_key, texte, kind, section, source_url,
               1 - (embedding <=> %(vector)s::vector) AS similarity
        FROM structured_chunks
        WHERE version_id = v.id
        ORDER BY embedding <=> %(vector)s::vector
        LIMIT %(top_k)s
    ) c ON TRUE
    WHERE d.entreprise = %(company)s
      AND d.type = %(form)s
      AND d.exercice = ANY(%(years)s::int[])
    ORDER BY d.exercice, c.similarity DESC
"""


class RetrievalError(Exception):
    pass


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL, device="cpu")


def encode_query(text: str):
    """Embed a question as a normalized numpy vector."""
    return _model().encode(text, normalize_embeddings=True)


def embed_query(text: str) -> str:
    """Embed a question as a pgvector literal."""
    return "[" + ",".join(str(float(x)) for x in encode_query(text)) + "]"


def retrieve_documents(conn, question: str, years, top_k: int) -> list[Source]:
    """Return the top_k most similar chunks for each requested year."""
    params = {
        "version": PIPELINE_VERSION,
        "vector": embed_query(question),
        "top_k": top_k,
        "company": COMPANY,
        "form": FORM_TYPE,
        "years": list(years),
    }

    with conn.cursor() as cur:
        cur.execute(QUERY, params)
        rows = cur.fetchall()

    documents = [
        Source(
            id=f"D{year}_{key}",
            kind="DOCUMENT",
            label=f"FY{year} 10-K, {section} ({kind})",
            url=url,
            excerpt=text,
            score=float(score),
        )
        for year, key, text, kind, section, url, score in rows
    ]

    missing = set(years) - {row[0] for row in rows}
    if missing:
        raise RetrievalError(f"No indexed 10-K chunks for FY{sorted(missing)}")

    return documents
