import os
import sys

import psycopg
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

question = " ".join(sys.argv[1:]).strip()

if not question:
    question = "What are Apple's main business risks?"

print("Question :", question)

# 1. Charger le modele d'embeddings
model = SentenceTransformer(MODEL_NAME, device="cpu")

# 2. Transformer la question en vecteur
embedding = model.encode(
    question,
    normalize_embeddings=True
)

vector = "[" + ",".join(
    str(float(x)) for x in embedding
) + "]"

# 3. Se connecter a PostgreSQL
config = {
    "host": os.getenv("DB_HOST", "127.0.0.1"),
    "port": int(os.getenv("DB_PORT", "5433")),
    "dbname": os.getenv("DB_NAME", "finsight"),
    "user": os.getenv("DB_USER", "finsight"),
    "password": os.getenv("DB_PASSWORD"),
}

# 4. Rechercher les 5 chunks les plus proches
with psycopg.connect(**config) as conn:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                c.id,
                c.texte,
                1 - (c.embedding <=> %s::vector) AS similarity
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            WHERE d.entreprise = 'Apple'
              AND d.exercice = 2025
              AND c.embedding IS NOT NULL
            ORDER BY c.embedding <=> %s::vector
            LIMIT 5;
            """,
            (vector, vector)
        )

        results = cur.fetchall()

print("\n--- RESULTATS ---")

for rank, (chunk_id, texte, similarity) in enumerate(
    results, start=1
):
    print(f"\n=== RESULTAT {rank} ===")
    print("Chunk ID :", chunk_id)
    print("Similarite cosinus :", round(float(similarity), 4))
    print("Texte :", texte[:700])
    print("...")
