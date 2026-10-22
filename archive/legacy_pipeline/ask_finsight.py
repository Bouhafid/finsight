import os
import sys

import psycopg
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

METRICS = {
    "total net sales": "Total net sales",
    "revenue": "Total net sales",
    "chiffre d'affaires": "Total net sales",
    "net income": "Net income",
    "net profit": "Net income",
    "benefice net": "Net income",
    "operating income": "Operating income",
    "total assets": "Total assets",
    "total liabilities": "Total liabilities",
}

question = " ".join(sys.argv[1:]).strip()

if not question:
    raise ValueError("Veuillez fournir une question")

print("\nQuestion :", question)

config = {
    "host": os.getenv("DB_HOST", "127.0.0.1"),
    "port": int(os.getenv("DB_PORT", "5433")),
    "dbname": os.getenv("DB_NAME", "finsight"),
    "user": os.getenv("DB_USER", "finsight"),
    "password": os.getenv("DB_PASSWORD"),
}

# Routeur simple
metric = None
for keyword, metric_name in METRICS.items():
    if keyword in question.lower():
        metric = metric_name
        break

with psycopg.connect(**config) as conn:
    with conn.cursor() as cur:

        if metric:
            print("Route : SQL FINANCIER")

            if any(y in question for y in ("2023", "2024", "2026")):
                print("Seules les valeurs FY2025 sont chargees.")
                sys.exit(0)

            cur.execute(
                """
                SELECT valeur, unite, periode, origine
                FROM financial_facts
                WHERE entreprise = %s
                  AND metrique = %s
                  AND periode = %s
                LIMIT 1
                """,
                ("Apple", metric, "FY2025")
            )

            result = cur.fetchone()

            if result:
                valeur, unite, periode, origine = result

                print("Metrique :", metric)
                print("Periode :", periode)
                print("Valeur :", f"{valeur:,.0f}", unite)
                print("Source :", origine)
            else:
                print("Aucune valeur trouvee.")

        else:
            print("Route : RECHERCHE VECTORIELLE")

            model = SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2",
                device="cpu"
            )

            embedding = model.encode(
                question,
                normalize_embeddings=True
            )

            vector = "[" + ",".join(
                str(float(x)) for x in embedding
            ) + "]"

            cur.execute(
                """
                SELECT c.id, c.texte,
                       1 - (c.embedding <=> %s::vector)
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                WHERE d.entreprise = %s
                  AND d.exercice = %s
                  AND c.embedding IS NOT NULL
                ORDER BY c.embedding <=> %s::vector
                LIMIT 3
                """,
                (vector, "Apple", 2025, vector)
            )

            for chunk_id, texte, score in cur.fetchall():
                print("\nChunk :", chunk_id)
                print("Similarite :", round(float(score), 4))
                print("Texte :", texte[:500])
