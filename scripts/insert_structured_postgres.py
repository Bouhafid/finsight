import hashlib
import json
import os
from pathlib import Path

import numpy as np
import psycopg
from dotenv import load_dotenv
from psycopg.types.json import Jsonb

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data/processed/apple"

load_dotenv(ROOT / ".env")

YEARS = (2023, 2024, 2025)
PIPELINE_VERSION = "structured-v1"

config = {
    "host": os.getenv("DB_HOST", "127.0.0.1"),
    "port": int(os.getenv("DB_PORT", "5433")),
    "dbname": os.getenv("DB_NAME", "finsight"),
    "user": os.getenv("DB_USER", "finsight"),
    "password": os.getenv("DB_PASSWORD"),
}


def load_year(year):
    chunks_path = DATA / f"chunks_structured_{year}.jsonl"
    embeddings_path = DATA / f"embeddings_structured_{year}.npy"
    manifest_path = DATA / f"embeddings_manifest_{year}.json"

    raw = chunks_path.read_bytes()

    chunks = [
        json.loads(line)
        for line in raw.decode("utf-8-sig").splitlines()
        if line.strip()
    ]

    embeddings = np.load(
        embeddings_path,
        allow_pickle=False
    )

    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    chunks_hash = hashlib.sha256(raw).hexdigest()

    embeddings_hash = hashlib.sha256(
        embeddings_path.read_bytes()
    ).hexdigest()

    if chunks_hash != manifest["chunks_sha256"]:
        raise ValueError(f"SHA256 des chunks incorrect FY{year}")

    if manifest["year"] != year:
        raise ValueError("Annee du manifest incorrecte")

    if manifest["chunk_count"] != len(chunks):
        raise ValueError("Nombre de chunks incoherent")

    if manifest["model"] != (
        "sentence-transformers/all-MiniLM-L6-v2"
    ):
        raise ValueError("Modele d'embeddings inattendu")

    if manifest["dimensions"] != 384:
        raise ValueError("Dimensions du manifest invalides")

    if embeddings.shape != (len(chunks), 384):
        raise ValueError("Dimensions embeddings invalides")

    if not np.isfinite(embeddings).all():
        raise ValueError("Embedding NaN ou inf")

    norms = np.linalg.norm(embeddings, axis=1)

    if not np.allclose(norms, 1, atol=1e-4):
        raise ValueError("Embeddings non normalises")

    keys = [chunk["chunk_id"] for chunk in chunks]

    if len(keys) != len(set(keys)):
        raise ValueError("IDs de chunks dupliques")

    for chunk in chunks:
        if chunk["year"] != year:
            raise ValueError("Annee incorrecte dans un chunk")

        if not chunk["texte"].strip():
            raise ValueError("Chunk vide")

        if not 1 <= chunk["token_count"] <= 256:
            raise ValueError("Nombre de tokens invalide")

        if not chunk.get("source_url"):
            raise ValueError("Source manquante")

        if not chunk.get("accession_number"):
            raise ValueError("Accession manquant")

    return (
        chunks,
        embeddings,
        manifest,
        chunks_hash,
        embeddings_hash
    )


# Valider toutes les annees avant d'ecrire en base
datasets = {}

for year in YEARS:
    datasets[year] = load_year(year)
    print(
        f"FY{year} valide : "
        f"{len(datasets[year][0])} chunks"
    )


with psycopg.connect(**config) as conn:
    with conn.cursor() as cur:

        for year in YEARS:
            chunks, embeddings, manifest, chash, ehash = (
                datasets[year]
            )

            cur.execute(
                """
                SELECT id, url FROM documents
                WHERE entreprise = %s
                  AND exercice = %s
                  AND type = '10-K'
                """,
                ("Apple", year)
            )

            documents = cur.fetchall()

            if len(documents) != 1:
                raise RuntimeError(
                    f"Document FY{year} absent ou duplique"
                )

            document_id, document_url = documents[0]

            if any(
                c["source_url"] != document_url
                for c in chunks
            ):
                raise RuntimeError(
                    f"Sources incoherentes FY{year}"
                )

            cur.execute(
                """
                INSERT INTO chunk_versions (
                    document_id,
                    pipeline_version,
                    model_name,
                    chunks_sha256,
                    embeddings_sha256,
                    chunk_count
                )
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (
                    document_id,
                    pipeline_version,
                    chunks_sha256,
                    embeddings_sha256
                )
                DO NOTHING
                RETURNING id
                """,
                (
                    document_id,
                    PIPELINE_VERSION,
                    manifest["model"],
                    chash,
                    ehash,
                    len(chunks)
                )
            )

            inserted = cur.fetchone()

            if inserted:
                version_id = inserted[0]
            else:
                cur.execute(
                    """
                    SELECT id FROM chunk_versions
                    WHERE document_id = %s
                      AND pipeline_version = %s
                      AND chunks_sha256 = %s
                      AND embeddings_sha256 = %s
                    """,
                    (
                        document_id,
                        PIPELINE_VERSION,
                        chash,
                        ehash
                    )
                )
                version_id = cur.fetchone()[0]

            cur.execute(
                """
                SELECT COUNT(*)
                FROM structured_chunks
                WHERE version_id = %s
                """,
                (version_id,)
            )

            existing_count = cur.fetchone()[0]

            if existing_count == len(chunks):
                print(f"FY{year} : deja importe, ignore")
                continue

            if existing_count != 0:
                raise RuntimeError(
                    f"Import incomplet FY{year} : "
                    f"{existing_count} lignes"
                )

            rows = []

            for chunk, embedding in zip(chunks, embeddings):
                vector = "[" + ",".join(
                    str(float(x)) for x in embedding
                ) + "]"

                rows.append((
                    version_id,
                    chunk["chunk_id"],
                    chunk["kind"],
                    chunk.get("section"),
                    Jsonb(chunk["source_block_ids"]),
                    Jsonb(chunk.get("table_row_range")),
                    chunk["source_url"],
                    chunk["accession_number"],
                    chunk["texte"],
                    chunk["token_count"],
                    vector,
                ))

            cur.executemany(
                """
                INSERT INTO structured_chunks (
                    version_id,
                    chunk_key,
                    kind,
                    section,
                    source_block_ids,
                    table_row_range,
                    source_url,
                    accession_number,
                    texte,
                    token_count,
                    embedding
                )
                VALUES (
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s::vector
                )
                """,
                rows
            )

            print(
                f"FY{year} importe : "
                f"{len(rows)} chunks, version {version_id}"
            )

print("\nINDEXATION STRUCTUREE TERMINEE")
