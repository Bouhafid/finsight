import os
import json
from pathlib import Path

import numpy as np
import psycopg
from dotenv import load_dotenv

load_dotenv()

CHUNKS_FILE = Path("data/processed/apple/chunks_2025.jsonl")
EMBEDDINGS_FILE = Path("data/processed/apple/embeddings_2025.npy")

# Charger les chunks
with CHUNKS_FILE.open("r", encoding="utf-8") as f:
    chunks = [json.loads(line) for line in f if line.strip()]

# Charger les embeddings
embeddings = np.load(EMBEDDINGS_FILE, allow_pickle=False)

print("Chunks :", len(chunks))
print("Dimensions embeddings :", embeddings.shape)

# Verifier la coherence des donnees
if embeddings.shape != (len(chunks), 384):
    raise ValueError("Incoherence entre chunks et embeddings")

if not np.isfinite(embeddings).all():
    raise ValueError("Certains embeddings contiennent NaN ou inf")

# Connexion PostgreSQL
config = {
    "host": os.getenv("DB_HOST"),
    "port": int(os.getenv("DB_PORT", "5433")),
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
}

print("Connexion PostgreSQL...")

with psycopg.connect(**config) as conn:
    with conn.cursor() as cur:

        # Chercher si le rapport existe deja
        cur.execute(
            """
            SELECT id FROM documents
            WHERE entreprise = %s
              AND type = %s
              AND exercice = %s
            ORDER BY id DESC
            LIMIT 1
            """,
            ("Apple", "10-K", 2025)
        )

        resultat = cur.fetchone()

        if resultat:
            document_id = resultat[0]
            print("Document existant :", document_id)
        else:
            # Inserer le rapport dans documents
            cur.execute(
                """
                INSERT INTO documents
                    (entreprise, type, exercice, url)
                VALUES (%s, %s, %s, %s)
                RETURNING id
                """,
                (
                    "Apple",
                    "10-K",
                    2025,
                    "https://www.sec.gov/Archives/edgar/data/"
                    "320193/000032019325000079/aapl-20250927.htm"
                )
            )

            document_id = cur.fetchone()[0]
            print("Document cree :", document_id)

        # Eviter de dupliquer les chunks
        cur.execute(
            "SELECT COUNT(*) FROM chunks WHERE document_id = %s",
            (document_id,)
        )

        existing_count = cur.fetchone()[0]

        if existing_count:
            print("Document deja indexe :", existing_count, "chunks")
            print("Aucune nouvelle insertion effectuee.")
        else:
            print("Insertion des chunks...")

            for chunk, embedding in zip(chunks, embeddings):

                # Transformer le vecteur en format pgvector
                vector = "[" + ",".join(
                    str(float(x)) for x in embedding
                ) + "]"

                cur.execute(
                    """
                    INSERT INTO chunks
                        (document_id, texte, embedding, kind)
                    VALUES (%s, %s, %s::vector, %s)
                    """,
                    (
                        document_id,
                        chunk["texte"],
                        vector,
                        "text"
                    )
                )

            print("Chunks inseres :", len(chunks))

print("Transaction terminee avec succes !")
