import os
import json
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from claim_verifier import verify_claim

load_dotenv()

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
            "SELECT texte FROM chunks WHERE id = %s",
            (51,)
        )
        row = cur.fetchone()

if row is None:
    raise RuntimeError("Chunk 51 introuvable")

evidence = row[0]

phrase = "supply shortages and price increases"

if phrase.lower() not in evidence.lower():
    raise RuntimeError(
        "La phrase attendue est absente du chunk 51. "
        "Verifier le contenu avant l'evaluation."
    )

position = evidence.lower().find(phrase.lower())
print("Phrase trouvee dans le chunk 51 :")
print(evidence[max(0, position - 100):position + 150])

tests = [
    {
        "claim": (
            "Apple faces significant risks of "
            "supply shortages and price increases."
        ),
        "expected": "SUPPORTED"
    },
    {
        "claim": (
            "Apple faces no risks of "
            "supply shortages or price increases."
        ),
        "expected": "CONTRADICTED"
    },
    {
        "claim": (
            "Apple's European Vision Pro revenue "
            "was exactly 12 billion dollars in 2025."
        ),
        "expected": "INSUFFICIENT"
    }
]

results = []

for i, test in enumerate(tests, start=1):
    print(f"\n--- TEST {i} ---")
    print("Affirmation :", test["claim"])
    print("Attendu :", test["expected"])

    result = verify_claim(test["claim"], evidence)

    correct = result["verdict"] == test["expected"]

    print("Obtenu :", result["verdict"])
    print("Correct :", correct)
    print("Raison :", result["reason"])

    results.append({
        **test,
        "predicted": result["verdict"],
        "correct": correct,
        "reason": result["reason"]
    })

correct_count = sum(r["correct"] for r in results)

print("\n========== EVALUATION ==========")
print("Tests :", len(results))
print("Corrects :", correct_count)
print("Erreurs :", len(results) - correct_count)
print(
    "Accuracy exploratoire :",
    round(correct_count / len(results), 3)
)

output = Path("data/evaluation")
output.mkdir(parents=True, exist_ok=True)

with (output / "verifier_results.json").open(
    "w", encoding="utf-8"
) as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("Resultats sauvegardes.")
