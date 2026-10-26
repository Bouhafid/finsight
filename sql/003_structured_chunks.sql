CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunk_versions (
    id BIGSERIAL PRIMARY KEY,
    document_id INT NOT NULL REFERENCES documents(id),
    pipeline_version TEXT NOT NULL,
    model_name TEXT NOT NULL,
    chunks_sha256 TEXT NOT NULL,
    embeddings_sha256 TEXT NOT NULL,
    chunk_count INT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE (
        document_id,
        pipeline_version,
        chunks_sha256,
        embeddings_sha256
    )
);

CREATE TABLE IF NOT EXISTS structured_chunks (
    id BIGSERIAL PRIMARY KEY,
    version_id BIGINT NOT NULL
        REFERENCES chunk_versions(id),
    chunk_key TEXT NOT NULL,
    kind TEXT NOT NULL,
    section TEXT,
    source_block_ids JSONB NOT NULL,
    table_row_range JSONB,
    source_url TEXT NOT NULL,
    accession_number TEXT NOT NULL,
    texte TEXT NOT NULL,
    token_count INT NOT NULL
        CHECK (token_count BETWEEN 1 AND 256),
    embedding VECTOR(384) NOT NULL,

    UNIQUE (version_id, chunk_key)
);

CREATE INDEX IF NOT EXISTS idx_structured_version
ON structured_chunks (version_id);

CREATE INDEX IF NOT EXISTS idx_structured_vector
ON structured_chunks
USING HNSW (embedding vector_cosine_ops);
