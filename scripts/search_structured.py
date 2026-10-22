import argparse
import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

PIPELINE_VERSION = "structured-v1"

parser = argparse.ArgumentParser(
    description="Recherche semantique FinSight multi-annees"
)

parser.add_argument("question", type=str)
parser.add_argument(
    "--years",
    nargs="+",
    type=int,
    choices=[2023, 2024, 2025],
    default=[2023, 2024, 2025]
)
parser.add_argument(
    "--top-k",
    type=int,
    default=3
)

args = parser.parse_args()

if not 1 <= args.top_k <= 20:
    parser.error("--top-k doit etre entre 1 et 20")

years = sorted(set(args.years))

print("\nQuestion :", args.question)
print("Annees :", years)
print("Version :", PIPELINE_VERSION)

# Charger MiniLM et encoder la question une seule fois
print("\nGeneration de l'embedding...")

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device="cpu"
)

embedding = model.encode(
    args.question,
    normalize_embeddings=True
)

vector = "[" + ",".join(
    str(float(x)) for x in embedding
) + "]"

config = {
    "host": os.getenv("DB_HOST", "127.0.0.1"),
    "port": int(os.getenv("DB_PORT", "5433")),
    "dbname": os.getenv("DB_NAME", "finsight"),
    "user": os.getenv("DB_USER", "finsight"),
    "password": os.getenv("DB_PASSWORD"),
}

# On recherche separement dans chaque exercice.
# Cela evite qu'une seule annee occupe tous les resultats.

SQL = """
SELECT
    c.chunk_key,
    c.texte,
    c.kind,
    c.section,
    c.source_block_ids,
    c.table_row_range,
    c.source_url,
    d.exercice,
    1 - (c.embedding <=> %s::vector) AS similarity

FROM documents d

JOIN LATERAL (
    SELECT v.id
    FROM chunk_versions v
    WHERE v.document_id = d.id
      AND v.pipeline_version = %s
    ORDER BY v.created_at DESC, v.id DESC
    LIMIT 1
) selected_version ON TRUE

JOIN structured_chunks c
    ON c.version_id = selected_version.id

WHERE d.entreprise = %s
  AND d.type = '10-K'
  AND d.exercice = %s

ORDER BY c.embedding <=> %s::vector

LIMIT %s
"""

with psycopg.connect(**config) as conn:
    with conn.cursor() as cur:

        for year in years:
            cur.execute(
                SQL,
                (
                    vector,
                    PIPELINE_VERSION,
                    "Apple",
                    year,
                    vector,
                    args.top_k
                )
            )

            results = cur.fetchall()

            if not results:
                raise RuntimeError(
                    f"Aucun chunk structure trouve FY{year}"
                )

            print(f"\n========== APPLE FY{year} ==========")
            print("Resultats :", len(results))

            for rank, result in enumerate(results, 1):
                (
                    chunk_key,
                    texte,
                    kind,
                    section,
                    block_ids,
                    row_range,
                    url,
                    exercice,
                    similarity
                ) = result

                print(f"\n--- RESULTAT {rank} ---")
                print("Chunk :", chunk_key)
                print("Similarite :", round(float(similarity), 4))
                print("Type :", kind)
                print("Section :", section)
                print("Blocs source :", block_ids)

                if row_range is not None:
                    print("Lignes du tableau :", row_range)

                print("Document :", url)
                print("Texte :", texte[:700])
                print("...")

print("\nRECHERCHE MULTI-ANNEES TERMINEE")
