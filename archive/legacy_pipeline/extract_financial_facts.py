import json
import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv

load_dotenv()

INPUT = Path("data/raw/apple/companyfacts.json")

ACCESSION = "0000320193-25-000079"
REPORT_END = "2025-09-27"
DOCUMENT_ID = 1

METRICS = {
    "Total net sales": "RevenueFromContractWithCustomerExcludingAssessedTax",
    "Net income": "NetIncomeLoss",
    "Operating income": "OperatingIncomeLoss",
    "Total assets": "Assets",
    "Total liabilities": "Liabilities",
}

data = json.loads(INPUT.read_text(encoding="utf-8"))
gaap = data["facts"]["us-gaap"]

facts_to_insert = []

for metric_name, tag in METRICS.items():
    concept = gaap.get(tag)

    if concept is None:
        print("Concept absent :", tag)
        continue

    observations = concept["units"].get("USD", [])

    candidates = [
        f for f in observations
        if f.get("accn") == ACCESSION
        and f.get("form") == "10-K"
        and f.get("end") == REPORT_END
        and f.get("fp") == "FY"
        and (
            "start" not in f
            or (
                300 <= (
                    __import__("datetime").date.fromisoformat(f["end"])
                    - __import__("datetime").date.fromisoformat(f["start"])
                ).days <= 380
            )
        )
    ]

    if not candidates:
        print("Valeur introuvable :", metric_name)
        continue

    values = {f["val"] for f in candidates}

    if len(values) != 1:
        raise ValueError(f"Valeurs contradictoires : {metric_name}")

    value = candidates[0]["val"]

    facts_to_insert.append((
        DOCUMENT_ID,
        "Apple",
        metric_name,
        "FY2025",
        value,
        "USD",
        f"SEC_XBRL:us-gaap:{tag}:{ACCESSION}"
    ))

    print(f"{metric_name}: {value:,.0f} USD")

if "Total net sales" not in [f[2] for f in facts_to_insert]:
    raise RuntimeError("Chiffre d'affaires annuel introuvable")

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
            SELECT id FROM documents WHERE id = %s
            """,
            (DOCUMENT_ID,)
        )

        if cur.fetchone() is None:
            raise RuntimeError("Document PostgreSQL introuvable")

        # Permet de relancer le script sans doublons XBRL.
        cur.execute(
            """
            DELETE FROM financial_facts
            WHERE document_id = %s
              AND origine LIKE 'SEC_XBRL:%%'
            """,
            (DOCUMENT_ID,)
        )

        for fact in facts_to_insert:
            cur.execute(
                """
                INSERT INTO financial_facts
                (document_id, entreprise, metrique,
                 periode, valeur, unite, origine)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                fact
            )

print("Donnees inserees :", len(facts_to_insert))
print("Insertion terminee !")

