import os
import re
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from claim_verifier import verify_claim

# Charger explicitement le fichier .env
PROJECT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_DIR / ".env")

# Connexion PostgreSQL
config = {
    "host": os.getenv("DB_HOST", "127.0.0.1"),
    "port": int(os.getenv("DB_PORT", "5433")),
    "dbname": os.getenv("DB_NAME", "finsight"),
    "user": os.getenv("DB_USER", "finsight"),
    "password": os.getenv("DB_PASSWORD"),
}

print("Connexion PostgreSQL...")

with psycopg.connect(**config) as conn:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT texte FROM chunks WHERE id = %s",
            (51,)
        )
        row = cur.fetchone()

if row is None:
    raise RuntimeError("Chunk 51 introuvable")

texte = row[0]

# Chercher une phrase precise concernant les risques
phrase = "supply shortages and price increases"

matches = list(
    re.finditer(
        r"[^.!?]*" + re.escape(phrase) + r"[^.!?]*[.!?]?",
        texte,
        flags=re.IGNORECASE
    )
)

if not matches:
    raise RuntimeError("Preuve introuvable dans le chunk 51")

# Extraire la phrase pertinente
evidence = matches[-1].group().strip()

print("\n========== PREUVE CIBLEE ==========")
print(evidence)

# Trois affirmations a tester
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

corrects = 0

# Executer les tests
for i, test in enumerate(tests, start=1):

    print(f"\n========== TEST {i} ==========")
    print("Affirmation :", test["claim"])
    print("Attendu :", test["expected"])

    result = verify_claim(
        test["claim"],
        evidence
    )

    correct = result["verdict"] == test["expected"]

    corrects += int(correct)

    print("Obtenu :", result["verdict"])
    print("Correct :", correct)
    print("Raison :", result["reason"])

# Afficher le score
print("\n========== RESULTAT FINAL ==========")
print("Tests :", len(tests))
print("Corrects :", corrects)
print("Erreurs :", len(tests) - corrects)
print(f"SCORE : {corrects}/{len(tests)}")
