
-- Table des rapports financiers
CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    entreprise TEXT NOT NULL,
    type TEXT,
    exercice INT,
    url TEXT,
    date_ingestion TIMESTAMPTZ DEFAULT NOW()
);

-- Table des passages pour le RAG
CREATE TABLE IF NOT EXISTS chunks (
    id SERIAL PRIMARY KEY,
    document_id INT REFERENCES documents(id),
    page INT,
    section TEXT,
    kind TEXT,
    texte TEXT NOT NULL,
    unite TEXT,
    periode TEXT,
    embedding VECTOR(384),
    tsv TSVECTOR GENERATED ALWAYS AS (
        to_tsvector('english', texte)
    ) STORED
);

-- Index de recherche textuelle
CREATE INDEX IF NOT EXISTS idx_chunks_text
ON chunks USING GIN (tsv);

-- Index de recherche vectorielle
CREATE INDEX IF NOT EXISTS idx_chunks_vector
ON chunks USING HNSW (embedding vector_cosine_ops);

-- Table des données financières structurées
CREATE TABLE IF NOT EXISTS financial_facts (
    id SERIAL PRIMARY KEY,
    document_id INT REFERENCES documents(id),
    entreprise TEXT,
    metrique TEXT,
    periode TEXT,
    valeur NUMERIC,
    unite TEXT,
    source_chunk_id INT REFERENCES chunks(id),
    origine TEXT
);
