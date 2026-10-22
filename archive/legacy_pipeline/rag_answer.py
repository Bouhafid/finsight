import os
import sys

import psycopg
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from cited_answer import generate_cited_answer

load_dotenv()


def retrieve(question, top_k=5):
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

    config = {
        "host": os.getenv("DB_HOST", "127.0.0.1"),
        "port": int(os.getenv("DB_PORT", "5433")),
        "dbname": os.getenv("DB_NAME", "finsight"),
        "user": os.getenv("DB_USER", "finsight"),
        "password": os.getenv("DB_PASSWORD"),
    }

    with psycopg.connect(**config) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    c.id,
                    c.texte,
                    1 - (c.embedding <=> %s::vector)
                        AS similarity,
                    d.url
                FROM chunks c
                JOIN documents d
                    ON d.id = c.document_id
                WHERE d.entreprise = %s
                  AND d.exercice = %s
                  AND c.embedding IS NOT NULL
                ORDER BY c.embedding <=> %s::vector
                LIMIT %s
                """,
                (vector, "Apple", 2025, vector, top_k)
            )

            return cur.fetchall()


def main():
    question = " ".join(sys.argv[1:]).strip()

    if not question:
        question = "What are Apple's main business risks?"

    print("\nQUESTION :", question)
    print("\nRecherche des passages...")

    results = retrieve(question)

    if not results:
        print("Aucun passage trouve.")
        return

    print("Chunks retrouves :", len(results))
    print("Generation et validation...")

    answer = generate_cited_answer(question, results)

    print("\n========== REPONSE ==========\n")

    if answer["insufficient_context"]:
        print(
            "Les passages retrouves ne suffisent pas "
            "pour etablir une reponse fiable."
        )
        return

    for claim in answer["claims"]:
        refs = " ".join(
            f"[chunk:{i}]" for i in claim["chunk_ids"]
        )

        print(f"- {claim['text']} {refs}")

    print("\n========== SOURCES CITEES ==========\n")

    cited_ids = {
        i
        for claim in answer["claims"]
        for i in claim["chunk_ids"]
    }

    for chunk_id, texte, similarity, url in results:
        if chunk_id not in cited_ids:
            continue

        print(f"Chunk : {chunk_id}")
        print(f"Similarite : {float(similarity):.4f}")
        print(f"Document : {url}")
        print(f"Extrait : {texte[:350]}")
        print()


if __name__ == "__main__":
    main()
